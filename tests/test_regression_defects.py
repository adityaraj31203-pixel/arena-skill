"""Regression tests for the defects found in the pre-merge audit of PR #1.

Each test is written to fail against the code as it stood before the fix, so a
reintroduction is caught rather than assumed away.

  D1  RunLog concurrent-write race          (HIGH)
  D2  Uncapped provider Retry-After sleep   (HIGH)
  D3  Over-broad redaction of ARENA_*       (HIGH)
  D4  base_url input silently ignored       (MEDIUM)
  D5  Live-smoke call count misstated       (MEDIUM)
"""

import json
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

from helper import TempWorkspace, make_state  # noqa: F401  (sys.path setup)

try:
    import yaml
    HAS_YAML = True
except ImportError:                                    # pragma: no cover
    yaml = None
    HAS_YAML = False

from arena_runner.bracket import load_state, write_prompts
from arena_runner.budget import Budget
from arena_runner.errors import ArenaRunnerError, RateLimited
from arena_runner.http_provider import (MAX_RETRY_DELAY, OpenAICompatibleProvider,
                                        bounded_delay, parse_retry_after)
from arena_runner.providers import CompletionRequest, ScriptedMockProvider
from arena_runner.redact import is_secret_name, redact, secret_values_from_env
from arena_runner.runlog import RunLog
from arena_runner.runner import TournamentRunner, init_run
from arena_runner.workflow_inputs import expected_calls, validate

STRESS_TRIALS = 8
STRESS_AGENTS = 16
STRESS_CONCURRENCY = 8


def run_tournament(workspace, agents, seed, concurrency, budget=None, provider=None):
    state = init_run(workspace.path("run-%d" % seed), agents=agents, seed=seed,
                     task_file=workspace.write_task())
    runner = TournamentRunner(state, provider or ScriptedMockProvider(),
                              budget=budget or Budget(max_calls=2000),
                              concurrency=concurrency, backoff=0.0)
    result = runner.run_to_completion()
    return state["dir"], result


# ===========================================================================
# D1 - RunLog thread safety
# ===========================================================================

class D1RunLogThreadSafety(unittest.TestCase):
    """The original failure: RuntimeError: dictionary changed size during
    iteration, raised from redact_mapping while another thread mutated
    self.data['jobs']. Reproduced 6/6 at concurrency=8 with 16 agents."""

    def test_concurrent_save_and_mutate_never_raises(self):
        with TempWorkspace() as ws:
            log = RunLog(ws.path("run"))
            errors = []

            def worker(i):
                try:
                    for k in range(40):
                        jid = "job-%d-%d" % (i, k)
                        log.start_job(jid, "competitor", "competitor")
                        log.save()
                        log.note_attempt(jid, 1, None)
                        log.finish_job(jid, attempt=1)
                        log.save()
                except Exception as exc:                       # pragma: no cover
                    errors.append("%s: %s" % (type(exc).__name__, exc))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [],
                             "concurrent runlog access raised: %s" % errors[:3])

    def test_runlog_json_is_never_corrupt_under_concurrent_saves(self):
        with TempWorkspace() as ws:
            log = RunLog(ws.path("run"))
            bad = []

            def worker(i):
                for k in range(30):
                    jid = "j-%d-%d" % (i, k)
                    log.start_job(jid, "attacker", "attacker")
                    log.save()
                    log.finish_job(jid, attempt=1)
                    log.save()
                    # Read the file back the way a resuming process would.
                    try:
                        with open(log.path, encoding="utf-8") as fh:
                            json.load(fh)
                    except ValueError as exc:                  # pragma: no cover
                        bad.append(str(exc))

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(bad, [], "runlog.json was unreadable: %s" % bad[:2])
            with open(log.path, encoding="utf-8") as fh:
                final = json.load(fh)
            self.assertEqual(len(final["jobs"]), 180)

    def test_snapshot_is_isolated_from_later_mutation(self):
        with TempWorkspace() as ws:
            log = RunLog(ws.path("run"))
            log.start_job("a", "competitor", "competitor")
            snap = log.snapshot()
            log.start_job("b", "competitor", "competitor")
            self.assertNotIn("b", snap["jobs"],
                             "snapshot aliases the live record")
            self.assertIn("b", log.snapshot()["jobs"])

    def test_stress_tournament_at_concurrency_8_with_16_agents(self):
        """The exact configuration that failed 6/6 before the fix."""
        failures = []
        for trial in range(STRESS_TRIALS):
            with TempWorkspace(prefix="d1-stress-") as ws:
                run_dir, result = run_tournament(ws, STRESS_AGENTS,
                                                 500 + trial, STRESS_CONCURRENCY)
                with open(os.path.join(run_dir, "runlog.json"), encoding="utf-8") as fh:
                    log = json.load(fh)
                states = {}
                for entry in log["jobs"].values():
                    states[entry["status"]] = states.get(entry["status"], 0) + 1
                if states.get("failed") or states.get("running"):
                    failures.append("trial %d states=%s" % (trial, states))
                if result["champion"] is None:
                    failures.append("trial %d produced no champion" % trial)
        self.assertEqual(failures, [],
                         "%d/%d stress trials failed: %s"
                         % (len(failures), STRESS_TRIALS, failures[:3]))

    def test_no_job_is_left_marked_running_after_a_crash(self):
        """An unexpected exception must still leave a terminal record."""
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      concurrency=4, backoff=0.0)

            def explode(job):
                raise RuntimeError("simulated internal failure")

            runner._execute_job = explode
            job = write_prompts(state, "spawn")[0]
            outcome = runner.execute_job(job)
            self.assertEqual(outcome["status"], "failed")
            with open(ws.path("run", "runlog.json"), encoding="utf-8") as fh:
                log = json.load(fh)
            self.assertEqual(log["jobs"][outcome["job_id"]]["status"], "failed")
            self.assertNotIn("running",
                             [e["status"] for e in log["jobs"].values()])

    def test_orphaned_running_jobs_are_reconciled_on_reload(self):
        """A record left behind by a killed process must not claim work is live."""
        with TempWorkspace() as ws:
            run_dir = ws.path("run")
            os.makedirs(run_dir, exist_ok=True)
            # Write exactly what a process killed mid-job would have left.
            with open(os.path.join(run_dir, "runlog.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"version": 1, "run_dir": run_dir, "jobs": {
                    "ghost.job": {"job_id": "ghost.job", "status": "running",
                                  "attempts": 1, "artifacts": [],
                                  "history": []}}}, fh)

            reloaded = RunLog(run_dir)
            entry = reloaded.snapshot()["jobs"]["ghost.job"]
            self.assertEqual(entry["status"], "interrupted")
            self.assertEqual(entry["error_kind"], "interrupted")
            self.assertIn("previous process", entry["error"])
            self.assertEqual(entry["history"][-1]["status"], "interrupted")
            # The reconciliation is itself persisted.
            reloaded.save()
            with open(os.path.join(run_dir, "runlog.json"), encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk["jobs"]["ghost.job"]["status"], "interrupted")

    def test_final_state_is_consistent_after_a_stress_run(self):
        with TempWorkspace() as ws:
            run_dir, result = run_tournament(ws, STRESS_AGENTS, 777,
                                             STRESS_CONCURRENCY)
            with open(os.path.join(run_dir, "arena.json"), encoding="utf-8") as fh:
                state = json.load(fh)
            alive = [a for a, v in state["agents"].items() if v["alive"]]
            self.assertEqual(alive, [result["champion"]])
            self.assertEqual(len(alive), 1)
            for rnd in state["rounds"]:
                self.assertTrue(rnd["closed"])
                for match in rnd["matches"]:
                    self.assertIsNotNone(match["winner"])
                    self.assertIsNotNone(match["loser"])
                    self.assertEqual(match["source"], "judge")

    def test_runlog_lock_is_reentrant_and_cannot_self_deadlock(self):
        with TempWorkspace() as ws:
            log = RunLog(ws.path("run"))
            done = []

            def nested():
                with log._lock:
                    with log._lock:                 # would block on a plain Lock
                        log.start_job("x", "competitor", "competitor")
                        done.append(True)

            t = threading.Thread(target=nested)
            t.start()
            t.join(timeout=5)
            self.assertFalse(t.is_alive(), "nested acquisition deadlocked")
            self.assertEqual(done, [True])


# ===========================================================================
# D2 - Retry-After cap
# ===========================================================================

class _RetryAfterServer:
    def __init__(self, header_value, status=429):
        self.header_value = header_value
        self.status = status
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.hits += 1
                body = json.dumps({"error": {"message": "slow down"}}).encode()
                self.send_response(outer.status)
                if outer.header_value is not None:
                    self.send_header("Retry-After", str(outer.header_value))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class D2RetryAfterCap(unittest.TestCase):
    def test_huge_retry_after_is_capped_not_honoured(self):
        """The original failure: Retry-After 86400 produced 3 x 86400s = 72h."""
        server = _RetryAfterServer("86400")
        slept = []
        import arena_runner.http_provider as hp
        real_sleep = hp.time.sleep
        hp.time.sleep = slept.append
        try:
            provider = OpenAICompatibleProvider(
                model="m", base_url=server.url, api_key="k", max_attempts=4)
            with self.assertRaises(RateLimited):
                provider.generate(CompletionRequest("j", "competitor", "p"))
        finally:
            hp.time.sleep = real_sleep
            server.stop()
        self.assertEqual(len(slept), 3)
        self.assertTrue(all(s <= MAX_RETRY_DELAY for s in slept),
                        "a sleep exceeded the cap: %s" % slept)
        self.assertLessEqual(sum(slept), MAX_RETRY_DELAY * 3)
        self.assertEqual(server.hits, 4)

    def test_cap_value_is_documented_at_30_seconds(self):
        self.assertEqual(MAX_RETRY_DELAY, 30.0)

    def test_valid_retry_after_under_the_cap_is_honoured(self):
        self.assertEqual(bounded_delay("12", 1), 12.0)
        self.assertEqual(bounded_delay("0", 1), 0.0)
        self.assertEqual(bounded_delay("29.5", 1), 29.5)

    def test_retry_after_at_or_over_the_cap_is_clamped(self):
        self.assertEqual(bounded_delay("30", 1), 30.0)
        self.assertEqual(bounded_delay("31", 1), 30.0)
        self.assertEqual(bounded_delay("86400", 1), 30.0)
        self.assertEqual(bounded_delay("1e12", 1), 30.0)

    def test_missing_retry_after_falls_back_to_bounded_backoff(self):
        for attempt in (1, 2, 3, 10):
            self.assertLessEqual(bounded_delay(None, attempt), MAX_RETRY_DELAY)
        self.assertEqual(bounded_delay(None, 1), 1.0)
        self.assertEqual(bounded_delay(None, 2), 1.5)

    def test_malformed_retry_after_is_ignored(self):
        for raw in ("", "   ", "soon", "Wed, 21 Oct 2026 07:28:00 GMT",
                    "12abc", "inf", "-inf", "nan"):
            self.assertIsNone(parse_retry_after(raw), "parsed %r" % raw)
            self.assertLessEqual(bounded_delay(raw, 1), MAX_RETRY_DELAY)

    def test_negative_retry_after_is_rejected(self):
        self.assertIsNone(parse_retry_after("-30"))
        self.assertIsNone(parse_retry_after("-0.5"))
        self.assertLessEqual(bounded_delay("-30", 1), MAX_RETRY_DELAY)

    def test_whitespace_padded_value_is_accepted(self):
        self.assertEqual(parse_retry_after("  12  "), 12.0)

    def test_retry_count_is_unchanged_by_the_cap(self):
        server = _RetryAfterServer("86400")
        import arena_runner.http_provider as hp
        real_sleep = hp.time.sleep
        hp.time.sleep = lambda s: None
        try:
            provider = OpenAICompatibleProvider(
                model="m", base_url=server.url, api_key="k", max_attempts=3)
            with self.assertRaises(RateLimited):
                provider.generate(CompletionRequest("j", "competitor", "p"))
        finally:
            hp.time.sleep = real_sleep
            server.stop()
        self.assertEqual(server.hits, 3, "the cap changed the number of attempts")


# ===========================================================================
# D3 - redaction preserves provenance
# ===========================================================================

class D3ProvenanceNotRedacted(unittest.TestCase):
    WORKFLOW_ENV = {
        "ARENA_IN_PROVIDER": "anthropic",
        "ARENA_IN_MODEL": "claude-3-5-haiku-latest",
        "ARENA_IN_AGENTS": "2",
        "ARENA_IN_MAX_CALLS": "20",
        "ARENA_IN_TASK": "Write a 12-word pricing headline.",
        "ARENA_RUN_DIR": "/work/.arena/run",
        "ARENA_WORKSPACE": "/work/.arena-workspace",
    }
    REAL_SECRETS = {
        "ARENA_ANTHROPIC_API_KEY": "sk-ant-" + "0123456789abcdefghijklmn",
        "ARENA_OPENAI_API_KEY": "sk-" + "live0123456789abcdefghijkl",
        "ARENA_PROVIDER_API_KEY": "provider-key-value-0123456789",
        "GH_TOKEN": "ghp_" + "0123456789abcdefghijklmnopqrst",
        "OPENAI_API_KEY": "sk-" + "openai0123456789abcdefghij",
    }

    def test_configuration_variables_are_not_treated_as_secrets(self):
        for name in self.WORKFLOW_ENV:
            self.assertFalse(is_secret_name(name), "%s misclassified" % name)

    def test_credential_variables_are_still_treated_as_secrets(self):
        for name in self.REAL_SECRETS:
            self.assertTrue(is_secret_name(name), "%s not detected" % name)

    def test_provenance_survives_a_full_workflow_style_run(self):
        saved = dict(os.environ)
        os.environ.update(self.WORKFLOW_ENV)
        os.environ.update(self.REAL_SECRETS)
        try:
            with TempWorkspace() as ws:
                state = init_run(ws.path("run"), agents=2, seed=7,
                                 task_file=ws.write_task())
                TournamentRunner(state, ScriptedMockProvider(model="mock-model-1"),
                                 budget=Budget(max_calls=100),
                                 backoff=0.0).run_to_completion()
                with open(ws.path("run", "runlog.json"), encoding="utf-8") as fh:
                    log = json.load(fh)

                self.assertEqual(log["provider"]["provider"], "scripted-mock")
                self.assertEqual(log["provider"]["model"], "mock-model-1")
                for entry in log["jobs"].values():
                    self.assertEqual(entry.get("model"), "mock-model-1",
                                     "%s lost its model" % entry["job_id"])
                    self.assertEqual(entry.get("provider_name"), "scripted-mock")
                self.assertEqual(log["independence"]["models"], ["mock-model-1"])
                self.assertNotIn("[REDACTED]", json.dumps(log["provider"]))
                self.assertNotIn("[REDACTED]", json.dumps(log["independence"]))
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_real_secrets_never_reach_the_runlog_or_artifacts(self):
        saved = dict(os.environ)
        os.environ.update(self.WORKFLOW_ENV)
        os.environ.update(self.REAL_SECRETS)
        try:
            with TempWorkspace() as ws:
                state = init_run(ws.path("run"), agents=2, seed=7,
                                 task_file=ws.write_task())
                TournamentRunner(state, ScriptedMockProvider(),
                                 budget=Budget(max_calls=100),
                                 backoff=0.0).run_to_completion()
                for root, _dirs, files in os.walk(ws.path("run")):
                    for name in files:
                        with open(os.path.join(root, name), encoding="utf-8",
                                  errors="replace") as fh:
                            body = fh.read()
                        for secret in self.REAL_SECRETS.values():
                            self.assertNotIn(secret, body,
                                             "secret in %s" % name)
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_detection_is_not_weakened_for_oddly_named_secrets(self):
        """A well-formed key in a variable with an innocuous name is still
        caught, by pattern rather than by name."""
        key = "sk-" + "live0123456789abcdefghijkl"
        self.assertFalse(is_secret_name("SOME_RANDOM_CONFIG"))
        self.assertNotIn(key, redact("the value is %s" % key))
        self.assertIn("[REDACTED]", redact("the value is %s" % key))

    def test_bearer_and_key_value_forms_are_still_redacted(self):
        self.assertIn("[REDACTED]",
                      redact("Authorization: Bearer abcdef0123456789abcdef"))
        self.assertIn("[REDACTED]", redact("api_key=supersecretvalue123"))

    def test_secret_values_are_collected_from_credential_names_only(self):
        saved = dict(os.environ)
        os.environ.update(self.WORKFLOW_ENV)
        os.environ.update(self.REAL_SECRETS)
        try:
            values = secret_values_from_env()
            for secret in self.REAL_SECRETS.values():
                self.assertIn(secret, values)
            for value in self.WORKFLOW_ENV.values():
                if len(value) >= 8:
                    self.assertNotIn(value, values,
                                     "%r treated as a secret" % value)
        finally:
            os.environ.clear()
            os.environ.update(saved)


# ===========================================================================
# D4 - base_url is respected end to end
# ===========================================================================

class D4BaseUrlHonoured(unittest.TestCase):
    def test_requests_reach_the_configured_gateway(self):
        """Proves the override is wired through, not silently dropped."""
        seen = []

        class Gateway(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                seen.append({"path": self.path,
                             "body": json.loads(self.rfile.read(length))})
                payload = json.dumps({"choices": [{"message": {"content": "from gateway"}}],
                                      "usage": {}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = HTTPServer(("127.0.0.1", 0), Gateway)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = "http://127.0.0.1:%d/gateway/v1" % server.server_address[1]
            config = validate({"task": "t", "agents": "2",
                               "provider": "openai-compatible", "model": "gw-model",
                               "confirm": "I-understand-this-may-incur-API-charges",
                               "base_url": url},
                              secrets_present=["ARENA_OPENAI_API_KEY"])
            self.assertEqual(config["base_url"], url)

            from arena_runner.cli import build_registry
            provider = build_registry().create(
                config["provider"], model=config["model"],
                base_url=config["base_url"], api_key="test-key")
            result = provider.generate(CompletionRequest("j", "competitor", "hello"))
            self.assertEqual(result.text, "from gateway")
            self.assertEqual(seen[0]["path"], "/gateway/v1/chat/completions")
            self.assertEqual(seen[0]["body"]["model"], "gw-model")
        finally:
            server.shutdown()
            server.server_close()

    def test_base_url_survives_validation(self):
        config = validate({"task": "t", "agents": "2", "provider": "scripted-mock",
                           "model": "m", "base_url": "https://gw.example.com/v1/"})
        self.assertEqual(config["base_url"], "https://gw.example.com/v1",
                         "trailing slash should be normalised")

    def test_absent_base_url_is_none_not_an_error(self):
        config = validate({"task": "t", "agents": "2", "provider": "scripted-mock",
                           "model": "m"})
        self.assertIsNone(config["base_url"])

    def test_invalid_base_urls_are_rejected(self):
        base = {"task": "t", "agents": "2", "provider": "scripted-mock", "model": "m"}
        for bad in ("ftp://gw.example.com", "gw.example.com", "https:// bad",
                    "https://user:pw@gw.example.com", "https://", "x" * 501):
            with self.assertRaises(ArenaRunnerError, msg=bad):
                validate(dict(base, base_url=bad))

    def test_unknown_inputs_are_rejected_rather_than_ignored(self):
        """The root cause of D4 was a validator that dropped unknown keys."""
        with self.assertRaises(ArenaRunnerError) as ctx:
            validate({"task": "t", "agents": "2", "provider": "scripted-mock",
                      "model": "m", "base_urll": "https://typo.example.com"})
        self.assertIn("unknown input", str(ctx.exception))

    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_every_workflow_input_is_known_to_the_validator(self):
        from arena_runner.workflow_inputs import KNOWN_INPUTS
        workflows = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), ".github", "workflows")
        for name in ("arena-tournament.yml", "live-smoke.yml"):
            with open(os.path.join(workflows, name), encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            trig = doc.get("on", doc.get(True))
            inputs = (trig["workflow_dispatch"] or {}).get("inputs", {}) or {}
            for key in inputs:
                self.assertIn(key, KNOWN_INPUTS,
                              "%s exposes input %r the validator would reject"
                              % (name, key))


# ===========================================================================
# D5 - live smoke call count
# ===========================================================================

class D5LiveSmokeCallCount(unittest.TestCase):
    def test_derived_call_count_for_the_live_smoke_configuration(self):
        self.assertEqual(expected_calls(2, baseline=False), 7)

    def test_count_matches_the_engine_job_graph(self):
        from arena_runner.bracket import plan_totals
        self.assertEqual(expected_calls(2, baseline=False), plan_totals(2)["calls"])

    def test_count_matches_an_actual_offline_run(self):
        with TempWorkspace() as ws:
            state = init_run(ws.path("run"), agents=2, seed=7,
                             task_file=ws.write_task())
            provider = ScriptedMockProvider()
            TournamentRunner(state, provider, budget=Budget(max_calls=100),
                             backoff=0.0).run_to_completion()
            self.assertEqual(len(provider.calls),
                             expected_calls(2, baseline=False))

    def test_baseline_adds_exactly_one_call(self):
        self.assertEqual(expected_calls(2, baseline=True), 8)

    def test_documentation_states_the_correct_number(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, ".github", "workflows",
                               "live-smoke.yml"), encoding="utf-8") as fh:
            smoke = fh.read()
        with open(os.path.join(repo, "docs", "MIGRATION.md"), encoding="utf-8") as fh:
            docs = fh.read()
        self.assertIn("7 model calls", smoke)
        self.assertIn("7 model calls", docs)
        for stale in ("11 model calls", "about 11 calls"):
            self.assertNotIn(stale, smoke)
            self.assertNotIn(stale, docs)

    def test_live_smoke_configures_no_baseline_so_no_final_check(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo, ".github", "workflows",
                               "live-smoke.yml"), encoding="utf-8") as fh:
            smoke = fh.read()
        self.assertNotIn("ARENA_IN_BASELINE", smoke)


if __name__ == "__main__":
    unittest.main()
