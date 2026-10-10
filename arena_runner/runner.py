"""The independent job executor.

This replaces the Claude Code Agent tool. Every job the engine schedules becomes
one stateless model API request with its own isolated context. The engine
(skills/arena/bracket.py) still owns the bracket, the cards, the pairing, the
rubric arithmetic and the elimination logic; this module only dispatches work and
records what happened.

Guarantees this module is responsible for:

  Isolation       each request is built from one job's declared inputs only.
                  A competitor's spawn prompt cannot contain another
                  competitor's draft, because that path is not in its input set.
  No fabrication  a failed job stays failed. It is recorded with an error kind
                  and is never rewritten as a plausible success.
  No self-judging the controller never chooses a winner. bracket.collect reads
                  the judge's own verdict file. If no valid verdict exists, the
                  round does not advance and the run stops with an error.
  Idempotence     jobs whose outputs already exist are skipped, so a resumed run
                  does not redo or double-charge completed work.
  Bounded work    concurrency, attempts, backoff, calls, tokens, characters and
                  elapsed time are all capped, and the run stops safely at a limit.
"""

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from .anonymize import anonymize, scrub
from .bracket import (RUBRIC_PATH, advance, collect, load_state, load_strategies,
                      missing_jobs, next_action, new_state, save_state, write_prompts)
from .budget import Budget
from .errors import (ArenaRunnerError, BudgetExceeded, JobFailed,
                     MalformedResponse, ProviderError, ProviderTimeout, RateLimited)
from .materialize import materialize
from .redact import redact, secret_values_from_env
from .runlog import RunLog
from .validation import multi_output_directive, validate

#: Overrides the "reply with this one line" receipt the original briefs ask for.
#: A Claude Code sub-agent wrote files and then sent a receipt; an API request
#: must return the artifact itself. This is the one intentional behavioural
#: difference from the Claude Code path, documented in docs/MIGRATION.md.
RECEIPT_OVERRIDE = (
    "\n\nIgnore any instruction above that tells you to reply with a one-line "
    "receipt such as DONE, ATTACKED, DEFENDED, WINNER or FINAL. You cannot write "
    "files, so a receipt would leave no artifact behind. Return the artifact "
    "itself, in the exact format the brief specifies."
)

RETRYABLE = (RateLimited, ProviderTimeout)

INDEPENDENCE_LABELS = {
    0: "Simulated: one executor imitates multiple roles",
    1: "Isolated requests: multiple stateless API requests with separate contexts",
    2: "Independently orchestrated: separate jobs, isolated state, validated "
       "outputs, controlled concurrency, reliable recovery",
    3: "Multi-provider evaluation: independent orchestration plus configured "
       "provider diversity",
}


def _atomic_write(path, text):
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".arena-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class _Args:
    """Stand-in for bracket's argparse namespace. Only .dir is used."""

    def __init__(self, run_dir):
        self.dir = run_dir


class TournamentRunner:
    def __init__(self, state, provider, budget=None, concurrency=2, max_attempts=3,
                 backoff=0.5, runlog=None, secrets=(), on_event=None):
        self.state = state
        self.provider = provider
        self.budget = budget or Budget()
        self.concurrency = max(1, int(concurrency))
        self.max_attempts = max(1, int(max_attempts))
        self.backoff = backoff
        self.secrets = set(secrets) | secret_values_from_env()
        self.runlog = runlog or RunLog(state["dir"], secrets=self.secrets)
        self.on_event = on_event or (lambda event: None)
        self.runlog.set_provider(provider.describe())
        self.runlog.set_engine({
            "engine": "skills/arena/bracket.py",
            "modified_by_migration": False,
            "seed": state.get("seed"),
            "agents_n": state.get("agents_n"),
        })

    # -- prompt construction -------------------------------------------------
    def build_prompt(self, job, brief_text):
        """Materialise inputs, blind the judge, append the API-mode directives."""
        prompt, isolation = materialize(self.state, job, brief_text)

        if job["kind"] == "judge":
            prompt, _ = anonymize(prompt, job["a"], job["b"])
        if job["kind"] in ("judge", "final"):
            if job["kind"] == "final":
                others = sorted(self.state["agents"])
            else:
                others = [a for a in sorted(self.state["agents"])
                          if a not in (job.get("a"), job.get("b"))]
            prompt = scrub(prompt, others,
                           provider_names=[self.provider.name],
                           model_names=[getattr(self.provider, "model", None)])

        prompt += RECEIPT_OVERRIDE
        prompt += multi_output_directive(job["outputs"])
        return prompt, isolation

    def build_request(self, job, prompt):
        from .providers import CompletionRequest
        return CompletionRequest(
            job_id=job["id"],
            role=job["kind"],
            prompt=prompt,
            system=("You are one competitor, attacker, defender or judge in a "
                    "single-elimination tournament. Follow the brief exactly. "
                    "You have no tools and no memory of any other request."),
            max_tokens=self.budget.max_output_tokens_per_call or 4096,
            temperature=0.9 if job["kind"] in ("competitor", "attacker") else 0.2,
            model=None,
            metadata={"phase": job["kind"], "round": job.get("round"),
                      "match": job.get("match")},
        )

    # -- single job ----------------------------------------------------------
    def execute_job(self, job):
        """Run one job end to end. Returns a per-job summary dict."""
        job_id = job["id"]
        brief_path = job["prompt"]
        if not os.path.exists(brief_path):
            raise ArenaRunnerError("brief missing for %s: %s" % (job_id, brief_path))
        with open(brief_path, encoding="utf-8") as fh:
            brief_text = fh.read()

        prompt, isolation = self.build_prompt(job, brief_text)
        request = self.build_request(job, prompt)

        try:
            self.budget.check_before_call(len(prompt), request.max_tokens)
        except BudgetExceeded as exc:
            self.runlog.start_job(job_id, job["kind"], job["kind"], request,
                                  self.provider.describe())
            self.runlog.finish_job(job_id, error=exc)
            self.runlog.save()
            return {"job_id": job_id, "status": "failed", "attempts": 0,
                    "error_kind": exc.kind, "error": str(exc)}

        self.runlog.start_job(job_id, job["kind"], job["kind"], request,
                              self.provider.describe())
        self.runlog.save()

        last_error = None
        attempt = 0
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self.provider.generate(request)
                self.budget.record_call(result)
                self.budget.check_estimated_cost()
                artifacts = validate(job, result.text,
                                     first=job.get("a"), second=job.get("b"))
                for path, content in artifacts.items():
                    _atomic_write(path, content)
                isolation_record = {
                    "inputs_inlined": sorted(isolation["inlined"]),
                    "missing_inputs": isolation["missing_inputs"],
                    "context_fingerprint": request.context_fingerprint(),
                }
                self.runlog.finish_job(job_id, result=result,
                                       artifacts=sorted(artifacts),
                                       isolation=isolation_record,
                                       attempt=attempt)
                self.runlog.save()
                self.on_event({"type": "job_done", "job_id": job_id, "attempt": attempt})
                return {"job_id": job_id, "status": "done", "attempts": attempt,
                        "artifacts": sorted(artifacts), "model": result.model,
                        "request_id": result.request_id,
                        "output_tokens": result.output_tokens}
            except BudgetExceeded:
                raise
            except RETRYABLE as exc:
                last_error = exc
            except (ProviderError, MalformedResponse) as exc:
                last_error = exc
                if not getattr(exc, "retryable", True):
                    break

            self.runlog.note_attempt(job_id, attempt, last_error)
            if attempt < self.max_attempts:
                try:
                    self.budget.check_before_retry()
                except BudgetExceeded:
                    break
                self.budget.record_retry()
                delay = getattr(last_error, "retry_after", None)
                if delay is None:
                    delay = self.backoff * (2 ** (attempt - 1))
                time.sleep(min(delay, 30.0))

        error = JobFailed(job_id, attempt, last_error)
        self.runlog.finish_job(job_id, error=error, attempt=attempt,
                               cause_kind=getattr(last_error, "kind", "error"))
        self.runlog.save()
        self.on_event({"type": "job_failed", "job_id": job_id,
                       "error": redact(str(last_error), self.secrets)})
        return {"job_id": job_id, "status": "failed", "attempts": attempt,
                "error_kind": getattr(last_error, "kind", "error"),
                "error": redact(str(last_error), self.secrets)}

    # -- phase ---------------------------------------------------------------
    def run_phase(self, phase):
        """Dispatch every outstanding job in one phase, with bounded concurrency."""
        jobs = write_prompts(self.state, phase)
        outstanding = missing_jobs(jobs)
        outstanding_ids = {j["id"] for j in outstanding}
        skipped = [j["id"] for j in jobs if j["id"] not in outstanding_ids]

        if not outstanding:
            summary = {"phase": phase, "dispatched": 0, "done": 0, "failed": 0,
                       "failures": [], "skipped_already_complete": skipped}
            self.runlog.record_phase(phase, summary)
            self.runlog.save()
            return summary

        done, failed = [], []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.execute_job, job): job for job in outstanding}
            for future in futures:
                job = futures[future]
                try:
                    outcome = future.result()
                except BudgetExceeded as exc:
                    outcome = {"job_id": job["id"], "status": "failed",
                               "error_kind": exc.kind, "error": str(exc)}
                except Exception as exc:        # one job must never kill the phase
                    outcome = {"job_id": job["id"], "status": "failed",
                               "error_kind": "unexpected",
                               "error": redact(str(exc), self.secrets)}
                (done if outcome["status"] == "done" else failed).append(outcome)

        summary = {"phase": phase, "dispatched": len(outstanding), "done": len(done),
                   "failed": len(failed), "failures": failed,
                   "skipped_already_complete": skipped}
        self.runlog.record_phase(phase, summary)
        self.runlog.set_budget(self.budget.summary())
        self.runlog.save()
        self.on_event({"type": "phase_done", "phase": phase,
                       "dispatched": summary["dispatched"], "done": summary["done"],
                       "failed": summary["failed"]})
        return summary

    # -- engine transitions --------------------------------------------------
    def _reload(self):
        self.state = load_state(_Args(self.state["dir"]))
        return self.state

    def _collect(self):
        with self.runlog.lock():
            self._reload()
            results = collect(self.state)
            save_state(self.state)
        bad = [(mid, status, note) for mid, status, note in results
               if status != "recorded"]
        if bad:
            raise ArenaRunnerError(
                "collect could not record %d verdict(s): %s. The controller will "
                "not choose a winner in the judge's place."
                % (len(bad), "; ".join("%s %s %s" % b for b in bad)))
        return results

    def _advance(self):
        with self.runlog.lock():
            self._reload()
            survivors = advance(self.state)
            save_state(self.state)
        return survivors

    # -- whole tournament ----------------------------------------------------
    def run_to_completion(self):
        """Drive the engine to a champion. Stops safely at any budget limit."""
        steps = []
        while True:
            phase, _left, _all_jobs = next_action(self.state)
            if phase == "done":
                break
            if phase == "collect":
                results = self._collect()
                steps.append({"phase": "collect", "recorded": len(results)})
                continue
            if phase == "advance":
                survivors = self._advance()
                steps.append({"phase": "advance", "alive": sorted(survivors)})
                self.on_event({"type": "round_advanced", "alive": len(survivors)})
                continue

            summary = self.run_phase(phase)
            steps.append({k: v for k, v in summary.items() if k != "failures"})
            if summary["failed"]:
                raise ArenaRunnerError(
                    "phase %s had %d failed job(s): %s"
                    % (phase, summary["failed"],
                       ", ".join(f["job_id"] for f in summary["failures"])))

        self._reload()
        self.runlog.set_budget(self.budget.summary())
        self.runlog.set_independence(independence_report(self.runlog, self.provider))
        self.runlog.save()
        return {"steps": steps, "champion": self.state.get("champion"),
                "final": self.state.get("final"),
                "budget": self.budget.summary(),
                "independence": self.runlog.data["independence"]}


def independence_report(runlog, provider):
    """Classify the independence actually achieved, from recorded evidence only."""
    jobs = list(runlog.jobs())
    fingerprints = {j.get("context_fingerprint") for j in jobs if j.get("context_fingerprint")}
    competitors = [j for j in jobs
                   if j.get("role") == "competitor" and j.get("status") == "done"]
    competitor_contexts = {j.get("context_fingerprint") for j in competitors}
    models = {j.get("model") for j in jobs if j.get("model")}

    level = 0
    evidence = []
    caveats = []

    if jobs:
        level = 1
        evidence.append("%d separate stateless requests were dispatched" % len(jobs))
        evidence.append("%d distinct context fingerprints recorded" % len(fingerprints))

    if len(competitors) > 1:
        if len(competitor_contexts) == len(competitors):
            evidence.append("all %d competitor requests had distinct contexts"
                            % len(competitors))
        else:
            level = 0
            caveats.append("only %d distinct contexts for %d competitors, so these "
                           "are not independent" % (len(competitor_contexts),
                                                    len(competitors)))

    if competitors and all(j.get("isolation", {}).get("inputs_inlined") is not None
                           for j in competitors):
        evidence.append("each competitor's inputs were checked against a declared "
                        "allowlist before dispatch")
    else:
        caveats.append("input allowlisting was not recorded for every competitor")

    if len(models) <= 1:
        caveats.append("all requests used one model (%s): these are separate "
                       "contexts, NOT different underlying models"
                       % (sorted(models)[0] if models else "unknown"))

    if not getattr(provider, "live", False):
        caveats.append("provider %r is an offline test double, so no real model "
                       "call has been exercised" % provider.name)

    caveats.append("Level 2 additionally requires separately orchestrated worker "
                   "processes verified under real provider failures; that has not "
                   "been exercised here.")
    caveats.append("Level 3 additionally requires two or more explicitly configured "
                   "providers; that is not configured here.")

    return {"level": level, "label": INDEPENDENCE_LABELS.get(level, "unknown"),
            "requests": len(jobs), "distinct_contexts": len(fingerprints),
            "competitor_jobs": len(competitors),
            "distinct_competitor_contexts": len(competitor_contexts),
            "models": sorted(models), "live_provider": bool(getattr(provider, "live", False)),
            "evidence": evidence, "caveats": [c for c in caveats if c]}


def init_run(run_dir, agents, seed=None, task_file=None, baseline_file=None,
             strategies=None, wave=10):
    """Create a run using the original engine's own init semantics.

    Mirrors bracket.cmd_init: the task is copied in, the rubric is frozen into
    the run directory, and new_state deals the cards.
    """
    data = strategies or load_strategies()
    if seed is None:
        seed = int(time.time()) % 1000000
    run_dir = os.path.abspath(run_dir)
    if os.path.exists(os.path.join(run_dir, "arena.json")):
        raise ArenaRunnerError("%s already has an arena in it" % run_dir)
    os.makedirs(run_dir, exist_ok=True)

    with open(task_file, encoding="utf-8") as fh:
        task = fh.read().strip()
    if not task:
        raise ArenaRunnerError("the task file is empty")
    with open(os.path.join(run_dir, "task.md"), "w", encoding="utf-8") as fh:
        fh.write(task + "\n")

    has_baseline = False
    if baseline_file:
        with open(baseline_file, encoding="utf-8") as fh:
            baseline = fh.read().strip()
        if not baseline:
            raise ArenaRunnerError("the baseline file is empty")
        with open(os.path.join(run_dir, "baseline.md"), "w", encoding="utf-8") as fh:
            fh.write(baseline + "\n")
        has_baseline = True

    shutil.copyfile(RUBRIC_PATH, os.path.join(run_dir, "rubric.md"))

    state = new_state(agents, seed, data, run_dir, wave=wave, has_baseline=has_baseline)
    save_state(state)
    return state
