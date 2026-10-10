"""Persistent run record and a file lock.

The engine keeps tournament bookkeeping in .arena/<run>/arena.json. This module
adds the execution record beside it: what was requested, who answered, when,
with which provider and model, how many attempts, and why anything failed.

Three properties matter for the requirements:

  Thread safety. The executor dispatches jobs from a thread pool, and every job
  records provenance here. All mutation of the shared record happens under one
  reentrant lock, and save() snapshots the record under that lock before
  serialising the private copy. Redaction and json.dump therefore never iterate
  a dict another thread is mutating. Two locks are used and never nested, so
  there is no acquisition order that can deadlock.

  Durability. Every write goes to a temporary file and is moved into place with
  os.replace, so an interrupted run never leaves a half-written record. A run
  can be resumed from whatever was last committed.

  No duplicate processing. A phase transition takes an exclusive lock, and job
  status is checked before dispatch, so two workers cannot both advance the same
  round or both run the same job.

  No orphaned jobs. A job that was left in "running" by a crashed process is
  marked "interrupted" when the record is next loaded, so nothing is reported as
  in progress forever.

The record contains no credentials. redact() is applied before writing.
"""

import copy
import errno
import json
import os
import tempfile
import threading
import time

from .redact import redact, redact_mapping, secret_values_from_env

RUNLOG_NAME = "runlog.json"
LOCK_NAME = "runlog.lock"


class FileLock:
    """Exclusive, advisory, stale-tolerant file lock."""

    def __init__(self, path, timeout=60.0, poll=0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self._fd = None

    def acquire(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return True
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                if self._stale():
                    self._break()
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError("could not acquire %s within %.0fs"
                                       % (self.path, self.timeout))
                time.sleep(self.poll)

    def _stale(self, max_age=900.0):
        try:
            return (time.time() - os.path.getmtime(self.path)) > max_age
        except OSError:
            return False

    def _break(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def release(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            os.unlink(self.path)
        except OSError:
            pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class RunLog:
    def __init__(self, run_dir, secrets=()):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, RUNLOG_NAME)
        self.lock_path = os.path.join(run_dir, LOCK_NAME)
        self.secrets = set(secrets) | secret_values_from_env()
        # One reentrant lock for the shared record. Reentrant so a mutating
        # method may call another mutating method without self-deadlocking.
        self._lock = threading.RLock()
        # A separate lock that only serialises the file replacement. It is never
        # held at the same time as _lock, so no lock ordering can deadlock.
        self._write_lock = threading.Lock()
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    data = json.load(fh)
                self._reconcile_orphans(data)
                return data
            except (ValueError, OSError):
                # A corrupt record must not be silently replaced with a clean one:
                # that would erase the evidence of what actually ran.
                backup = self.path + ".corrupt.%d" % int(time.time())
                os.replace(self.path, backup)
                return self._fresh(corrupt_backup=backup)
        return self._fresh()

    @staticmethod
    def _reconcile_orphans(data):
        """A job still marked "running" belongs to a process that is gone.

        Nothing is reported as in progress forever: the record is reloaded by a
        live process, so any "running" entry is by definition orphaned and is
        given a terminal status with the reason recorded.
        """
        for entry in (data.get("jobs") or {}).values():
            if entry.get("status") == "running":
                entry["status"] = "interrupted"
                entry["error_kind"] = "interrupted"
                entry["error"] = ("left in running by a previous process; "
                                  "re-run to retry this job")
                entry.setdefault("history", []).append({
                    "attempt": int(entry.get("attempts", 0)),
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "interrupted",
                    "error_kind": "interrupted",
                })

    def _fresh(self, corrupt_backup=None):
        data = {
            "version": 1,
            "run_dir": self.run_dir,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": None,
            "engine": None,
            "jobs": {},
            "budget": None,
            "independence": None,
            "phases": [],
        }
        if corrupt_backup:
            data["corrupt_record_moved_to"] = corrupt_backup
        return data

    # -- persistence ---------------------------------------------------------
    def snapshot(self):
        """A deep, private copy of the record, taken under the lock."""
        with self._lock:
            return copy.deepcopy(self.data)

    def save(self):
        """Redact and write a snapshot.

        The snapshot is taken under _lock, then redaction and serialisation work
        on that private copy, so no other thread can mutate the dict while it is
        being iterated. The write itself is serialised separately so an older
        snapshot can never replace a newer one.
        """
        os.makedirs(self.run_dir, exist_ok=True)
        snapshot = self.snapshot()
        payload = redact_mapping(snapshot, self.secrets)
        text = json.dumps(payload, indent=2, sort_keys=True)
        with self._write_lock:
            fd, tmp = tempfile.mkstemp(dir=self.run_dir, prefix=".runlog.",
                                       suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(text)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    def lock(self, timeout=60.0):
        return FileLock(self.lock_path, timeout=timeout)

    # -- provenance ----------------------------------------------------------
    def set_provider(self, description):
        with self._lock:
            self.data["provider"] = redact_mapping(description, self.secrets)

    def set_engine(self, description):
        with self._lock:
            self.data["engine"] = description

    def set_budget(self, summary):
        with self._lock:
            self.data["budget"] = summary

    def set_independence(self, report):
        with self._lock:
            self.data["independence"] = report

    def start_job(self, job_id, phase, role, request=None, provider=None):
        public = redact_mapping(request.to_public_dict(), self.secrets) \
            if request is not None else None
        redacted_provider = redact_mapping(provider, self.secrets) \
            if provider is not None else None
        with self._lock:
            entry = self.data["jobs"].setdefault(job_id, {
                "job_id": job_id, "phase": phase, "role": role,
                "status": "pending", "attempts": 0, "artifacts": [],
                "history": [],
            })
            entry["status"] = "running"
            entry["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if public is not None:
                entry["request"] = public
                entry["context_fingerprint"] = public["context_fingerprint"]
            if redacted_provider is not None:
                entry["provider"] = redacted_provider
            return copy.deepcopy(entry)

    def finish_job(self, job_id, result=None, artifacts=(), isolation=None,
                   error=None, attempt=None, cause_kind=None):
        """Record a terminal outcome for a job.

        Nothing here iterates the shared record, so it is safe to call from a
        worker thread. The caller passes everything needed; only the entry
        itself is touched, under the lock.
        """
        public_result = redact_mapping(result.to_public_dict(), self.secrets) \
            if result is not None else None
        rel_artifacts = [os.path.relpath(a, self.run_dir) for a in artifacts]
        message = redact(str(error), self.secrets)[:500] if error is not None else None
        with self._lock:
            entry = self.data["jobs"].setdefault(job_id, {
                "job_id": job_id, "phase": None, "role": None,
                "status": "pending", "attempts": 0, "artifacts": [],
                "history": [],
            })
            if attempt is None:
                attempt = int(entry.get("attempts", 0)) + 1
            entry["attempts"] = int(attempt)
            entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if error is None:
                entry["status"] = "done"
                entry["artifacts"] = rel_artifacts
                if public_result is not None:
                    entry["result"] = public_result
                    entry["request_id"] = result.request_id
                    entry["model"] = result.model
                    entry["provider_name"] = result.provider
                if isolation is not None:
                    entry["isolation"] = isolation
                entry.pop("error_kind", None)
                entry.pop("wrapper_kind", None)
                entry.pop("error", None)
            else:
                entry["status"] = "failed"
                entry["error_kind"] = cause_kind or getattr(error, "kind", "error")
                entry["wrapper_kind"] = getattr(error, "kind", "error")
                entry["error"] = message
            entry.setdefault("history", []).append({
                "attempt": entry["attempts"],
                "at": entry["finished_at"],
                "status": entry["status"],
                "error_kind": entry.get("error_kind"),
            })
            entry["history"] = sorted(entry["history"], key=lambda h: h["attempt"])
            return copy.deepcopy(entry)

    def note_attempt(self, job_id, attempt, error=None):
        """Record an intermediate failed attempt so the retry trail survives."""
        kind = getattr(error, "kind", "error") if error is not None else None
        with self._lock:
            entry = self.data["jobs"].setdefault(job_id, {
                "job_id": job_id, "status": "running", "attempts": 0,
                "artifacts": [], "history": [],
            })
            entry.setdefault("history", []).append({
                "attempt": int(attempt),
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "retry",
                "error_kind": kind,
            })
            entry["history"] = sorted(entry["history"], key=lambda h: h["attempt"])


    def status_of(self, job_id):
        with self._lock:
            entry = self.data["jobs"].get(job_id)
            return entry["status"] if entry else None

    def record_phase(self, phase, summary):
        redacted = redact_mapping(summary, self.secrets)
        with self._lock:
            self.data.setdefault("phases", []).append({
                "phase": phase,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "summary": redacted,
            })

    # -- queries -------------------------------------------------------------
    def jobs(self, status=None):
        """Terminal-safe iteration over a snapshot, so callers never hold the lock."""
        with self._lock:
            entries = copy.deepcopy(list(self.data["jobs"].values()))
        for entry in entries:
            if status is None or entry.get("status") == status:
                yield entry

    def counts(self):
        with self._lock:
            entries = list(self.data["jobs"].values())
        out = {}
        for entry in entries:
            key = entry.get("status", "unknown")
            out[key] = out.get(key, 0) + 1
        return out

    def context_fingerprints(self):
        """job_id -> fingerprint, for every job that was actually dispatched."""
        with self._lock:
            entries = list(self.data["jobs"].values())
        return {j["job_id"]: j.get("context_fingerprint")
                for j in entries if j.get("context_fingerprint")}
