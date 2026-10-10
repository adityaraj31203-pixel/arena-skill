"""Persistent run record and a file lock.

The engine keeps tournament bookkeeping in .arena/<run>/arena.json. This module
adds the execution record beside it: what was requested, who answered, when,
with which provider and model, how many attempts, and why anything failed.

Two properties matter for the requirements:

  Durability. Every write goes to a temporary file and is moved into place with
  os.replace, so an interrupted run never leaves a half-written record. A run
  can be resumed from whatever was last committed.

  No duplicate processing. A phase transition takes an exclusive lock, and job
  status is checked before dispatch, so two workers cannot both advance the same
  round or both run the same job.

The record contains no credentials. redact() is applied before writing.
"""

import errno
import json
import os
import tempfile
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
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as fh:
                    return json.load(fh)
            except (ValueError, OSError):
                # A corrupt record must not be silently replaced with a clean one:
                # that would erase the evidence of what actually ran.
                backup = self.path + ".corrupt.%d" % int(time.time())
                os.replace(self.path, backup)
                return self._fresh(corrupt_backup=backup)
        return self._fresh()

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
    def save(self):
        os.makedirs(self.run_dir, exist_ok=True)
        payload = redact_mapping(self.data, self.secrets)
        fd, tmp = tempfile.mkstemp(dir=self.run_dir, prefix=".runlog.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
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
        self.data["provider"] = redact_mapping(description, self.secrets)

    def set_engine(self, description):
        self.data["engine"] = description

    def set_budget(self, summary):
        self.data["budget"] = summary

    def set_independence(self, report):
        self.data["independence"] = report

    def start_job(self, job_id, phase, role, request=None, provider=None):
        entry = self.data["jobs"].setdefault(job_id, {
            "job_id": job_id, "phase": phase, "role": role,
            "status": "pending", "attempts": 0, "artifacts": [],
            "history": [],
        })
        entry["status"] = "running"
        entry["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if request is not None:
            public = redact_mapping(request.to_public_dict(), self.secrets)
            entry["request"] = public
            entry["context_fingerprint"] = public["context_fingerprint"]
        if provider is not None:
            entry["provider"] = redact_mapping(provider, self.secrets)
        return entry

    def finish_job(self, job_id, result=None, artifacts=(), isolation=None,
                   error=None, attempt=None, cause_kind=None):
        entry = self.data["jobs"][job_id]
        if attempt is None:
            attempt = int(entry.get("attempts", 0)) + 1
        entry["attempts"] = int(attempt)
        entry["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if error is None:
            entry["status"] = "done"
            entry["artifacts"] = [os.path.relpath(a, self.run_dir) for a in artifacts]
            if result is not None:
                entry["result"] = redact_mapping(result.to_public_dict(), self.secrets)
                entry["request_id"] = result.request_id
                entry["model"] = result.model
                entry["provider_name"] = result.provider
            if isolation is not None:
                entry["isolation"] = isolation
            entry.pop("error_kind", None)
            entry.pop("error", None)
        else:
            entry["status"] = "failed"
            entry["error_kind"] = cause_kind or getattr(error, "kind", "error")
            entry["wrapper_kind"] = getattr(error, "kind", "error")
            entry["error"] = redact(str(error), self.secrets)[:500]
        entry["history"].append({
            "attempt": entry["attempts"],
            "at": entry["finished_at"],
            "status": entry["status"],
            "error_kind": entry.get("error_kind"),
        })
        entry["history"] = sorted(entry["history"], key=lambda h: h["attempt"])
        return entry

    def note_attempt(self, job_id, attempt, error=None):
        """Record an intermediate failed attempt so the retry trail survives."""
        entry = self.data["jobs"][job_id]
        entry["history"].append({
            "attempt": int(attempt),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "retry",
            "error_kind": getattr(error, "kind", "error") if error else None,
        })
        entry["history"] = sorted(entry["history"], key=lambda h: h["attempt"])

    def status_of(self, job_id):
        entry = self.data["jobs"].get(job_id)
        return entry["status"] if entry else None

    def record_phase(self, phase, summary):
        self.data["phases"].append({
            "phase": phase,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": redact_mapping(summary, self.secrets),
        })

    # -- queries -------------------------------------------------------------
    def jobs(self, status=None):
        for entry in self.data["jobs"].values():
            if status is None or entry.get("status") == status:
                yield entry

    def counts(self):
        out = {}
        for entry in self.data["jobs"].values():
            out[entry.get("status", "unknown")] = out.get(entry.get("status", "unknown"), 0) + 1
        return out

    def context_fingerprints(self):
        """job_id -> fingerprint, for every job that was actually dispatched."""
        return {j["job_id"]: j.get("context_fingerprint")
                for j in self.data["jobs"].values() if j.get("context_fingerprint")}
