"""Executor behaviour: completion, failure handling, retries, idempotence,
resume after interruption, concurrency, artifact persistence, and the rule that
the controller never picks a winner itself.

Every response comes from an offline test double. No credentials, no spend.
"""

import json
import os
import threading
import time
import unittest

from helper import TempWorkspace, make_state, read

from arena_runner.bracket import (advance, collect, current_round, load_state,
                                  phase_jobs, save_state, winner_report,
                                  write_prompts)
from arena_runner.budget import Budget
from arena_runner.errors import ArenaRunnerError, BudgetExceeded
from arena_runner.providers import CompletionResult, MockProvider, ScriptedMockProvider
from arena_runner.runlog import RunLog
from arena_runner.report import format_winner
from arena_runner.runner import TournamentRunner, independence_report


class _Args:
    def __init__(self, run_dir):
        self.dir = run_dir


def reload_state(run_dir):
    return load_state(_Args(run_dir))


class SmallTournamentCompletes(unittest.TestCase):
    def test_two_agents_reach_a_champion(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=100), backoff=0.0)
            result = runner.run_to_completion()
            self.assertIsNotNone(result["champion"])
            self.assertIn(result["champion"], state["agents"])

    def test_four_agents_reach_exactly_one_survivor(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=200), backoff=0.0)
            result = runner.run_to_completion()
            final = reload_state(ws.path("run"))
            alive = [a for a, v in final["agents"].items() if v["alive"]]
            self.assertEqual(alive, [result["champion"]])
            self.assertEqual(len(alive), 1)

    def test_final_check_runs_only_when_there_is_a_baseline(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7, baseline=True)
            result = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=100),
                                      backoff=0.0).run_to_completion()
            self.assertIsNotNone(result["final"])
            self.assertIn(result["final"]["result"]["better"],
                          ("champion", "baseline"))

        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7, baseline=False)
            result = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=100),
                                      backoff=0.0).run_to_completion()
            self.assertIsNone(result["final"])

    def test_winner_report_is_reproducible_from_persisted_state(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            TournamentRunner(state, ScriptedMockProvider(),
                             budget=Budget(max_calls=200),
                             backoff=0.0).run_to_completion()
            structured = winner_report(reload_state(ws.path("run")))
            self.assertEqual(structured["agents"], 4)
            self.assertEqual(structured["matches_won"], 2)
            rendered = format_winner(ws.path("run"))
            self.assertIn("CHAMPION", rendered)
            self.assertIn("rounds", rendered)
            self.assertIn("not proof of objective correctness", rendered)

    def test_champion_solution_file_exists_and_is_nonempty(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            result = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=200),
                                      backoff=0.0).run_to_completion()
            final = reload_state(ws.path("run"))
            path = final["agents"][result["champion"]]["solution"]
            self.assertTrue(os.path.exists(path))
            self.assertGreater(len(read(path).strip()), 0)


class FailureHandling(unittest.TestCase):
    def test_a_failing_job_is_recorded_as_failed_not_fabricated(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            provider = MockProvider(behaviour="http_400")
            runner = TournamentRunner(state, provider, backoff=0.0, max_attempts=1)
            summary = runner.run_phase("spawn")
            self.assertEqual(summary["done"], 0)
            self.assertEqual(summary["failed"], 2)
            for aid, agent in state["agents"].items():
                self.assertFalse(os.path.exists(agent["solution"]),
                                 "%s got an artifact despite failing" % aid)
            log = json.load(open(ws.path("run", "runlog.json")))
            for entry in log["jobs"].values():
                self.assertEqual(entry["status"], "failed")
                self.assertEqual(entry["error_kind"], "provider_error")
                self.assertEqual(entry["wrapper_kind"], "job_failed")

    def test_malformed_output_is_not_written_as_an_artifact(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            provider = MockProvider(behaviour="malformed")
            runner = TournamentRunner(state, provider, backoff=0.0, max_attempts=1)
            summary = runner.run_phase("attack") if phase_jobs(state, "attack") \
                else runner.run_phase("spawn")
            self.assertEqual(summary["failed"], summary["dispatched"])

    def test_a_transient_failure_recovers_on_retry(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            provider = ScriptedMockProvider(script={
                "a001.spawn": ["rate_limit", "ok"],
            })
            runner = TournamentRunner(state, provider, backoff=0.0, max_attempts=3)
            summary = runner.run_phase("spawn")
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["done"], 2)
            log = json.load(open(ws.path("run", "runlog.json")))
            self.assertEqual(log["jobs"]["a001.spawn"]["attempts"], 2)

    def test_retries_are_bounded(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            provider = MockProvider(behaviour="timeout")
            runner = TournamentRunner(state, provider, backoff=0.0, max_attempts=3)
            runner.run_phase("spawn")
            log = json.load(open(ws.path("run", "runlog.json")))
            for entry in log["jobs"].values():
                self.assertLessEqual(entry["attempts"], 3)
                self.assertEqual(entry["status"], "failed")

    def test_budget_exhaustion_stops_the_run_safely(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=2), backoff=0.0)
            summary = runner.run_phase("spawn")
            self.assertLessEqual(summary["done"], 2)
            self.assertGreaterEqual(summary["failed"], 2)
            with self.assertRaises(ArenaRunnerError):
                runner.run_to_completion()

    def test_missing_input_is_marked_not_invented(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
            runner.run_phase("spawn")
            # Delete one solution, then attack: the attacker must see it as missing.
            victim = sorted(state["agents"])[0]
            os.unlink(state["agents"][victim]["solution"])
            jobs = write_prompts(state, "attack")
            for job in jobs:
                _prompt, isolation = runner.build_prompt(job, read(job["prompt"]))
                if job["opponent"] == victim:
                    self.assertIn("target_solution", isolation["missing_inputs"])


class IdempotenceAndResume(unittest.TestCase):
    def test_completed_jobs_are_not_dispatched_twice(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            provider = ScriptedMockProvider()
            runner = TournamentRunner(state, provider, backoff=0.0)
            first = runner.run_phase("spawn")
            calls_after_first = len(provider.calls)
            second = runner.run_phase("spawn")
            self.assertEqual(first["done"], 4)
            self.assertEqual(second["dispatched"], 0)
            self.assertEqual(len(second["skipped_already_complete"]), 4)
            self.assertEqual(len(provider.calls), calls_after_first)

    def test_resume_after_an_interrupted_run_finishes_the_tournament(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            run_dir = ws.path("run")
            first = TournamentRunner(state, MockProvider(behaviour="http_400"),
                                     backoff=0.0, max_attempts=1)
            first.run_phase("spawn")            # every job fails
            self.assertFalse(any(os.path.exists(a["solution"])
                                 for a in state["agents"].values()))

            # A fresh process would reload from disk; do the same here.
            resumed_state = reload_state(run_dir)
            second = TournamentRunner(resumed_state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=200), backoff=0.0)
            result = second.run_to_completion()
            self.assertIsNotNone(result["champion"])
            final = reload_state(run_dir)
            self.assertEqual(len([a for a, v in final["agents"].items()
                                  if v["alive"]]), 1)

    def test_runlog_survives_a_reload_and_keeps_history(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
            runner.run_phase("spawn")
            reloaded = RunLog(ws.path("run"))
            self.assertEqual(len(list(reloaded.jobs())), 2)
            self.assertEqual(reloaded.counts()["done"], 2)

    def test_attempt_history_is_preserved_across_resumes(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            TournamentRunner(state, ScriptedMockProvider(script={
                "a001.spawn": ["timeout", "ok"]}), backoff=0.0,
                max_attempts=3).run_phase("spawn")
            log = json.load(open(ws.path("run", "runlog.json")))
            history = log["jobs"]["a001.spawn"]["history"]
            self.assertEqual(len(history), 2)
            self.assertEqual([h["status"] for h in history], ["retry", "done"])
            self.assertEqual([h["attempt"] for h in history], [1, 2])
            self.assertEqual(history[0]["error_kind"], "timeout")

    def test_a_corrupt_runlog_is_preserved_not_silently_replaced(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            TournamentRunner(state, ScriptedMockProvider(),
                             backoff=0.0).run_phase("spawn")
            with open(ws.path("run", "runlog.json"), "w") as fh:
                fh.write("{ this is not json")
            log = RunLog(ws.path("run"))
            self.assertIn("corrupt_record_moved_to", log.data)
            self.assertTrue(os.path.exists(log.data["corrupt_record_moved_to"]))


class Concurrency(unittest.TestCase):
    def test_jobs_run_concurrently_up_to_the_limit(self):
        observed = {"peak": 0, "active": 0}
        lock = threading.Lock()

        class CountingProvider(ScriptedMockProvider):
            def generate(self, request):
                with lock:
                    observed["active"] += 1
                    observed["peak"] = max(observed["peak"], observed["active"])
                try:
                    # Hold the worker briefly so real overlap is observable.
                    time.sleep(0.05)
                    return super().generate(request)
                finally:
                    with lock:
                        observed["active"] -= 1

        with TempWorkspace() as ws:
            state = make_state(ws, agents=8, seed=7)
            runner = TournamentRunner(state, CountingProvider(),
                                      concurrency=3, backoff=0.0)
            runner.run_phase("spawn")
            self.assertLessEqual(observed["peak"], 3)
            self.assertGreater(observed["peak"], 1)

    def test_concurrency_of_one_is_serial(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      concurrency=1, backoff=0.0)
            summary = runner.run_phase("spawn")
            self.assertEqual(summary["done"], 4)

    def test_one_job_failing_does_not_abort_the_phase(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            provider = ScriptedMockProvider(script={"a002.spawn": ["http_400"]})
            runner = TournamentRunner(state, provider, concurrency=4,
                                      backoff=0.0, max_attempts=1)
            summary = runner.run_phase("spawn")
            self.assertEqual(summary["done"], 3)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["failures"][0]["job_id"], "a002.spawn")


class ControllerNeverJudges(unittest.TestCase):
    def test_a_round_does_not_advance_without_a_verdict(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
            for phase in ("spawn", "attack", "defend"):
                runner.run_phase(phase)
            self.assertIsNone(current_round(state)["matches"][0]["winner"])
            results = collect(state)
            self.assertTrue(any(status == "missing" for _mid, status, _n in results))
            with self.assertRaises(Exception):
                advance(state)

    def test_an_unreadable_verdict_is_set_aside_and_the_run_stops(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            provider = ScriptedMockProvider(script={"r1-m01.judge": ["judge_bad_json"]})
            runner = TournamentRunner(state, provider, backoff=0.0, max_attempts=1)
            with self.assertRaises(ArenaRunnerError) as ctx:
                runner.run_to_completion()
            self.assertIn("failed", str(ctx.exception))
            final = reload_state(ws.path("run"))
            self.assertIsNone(final["champion"],
                              "a champion was crowned without a valid verdict")

    def test_the_recorded_winner_comes_from_the_judge_verdict_file(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
            for phase in ("spawn", "attack", "defend", "judge"):
                runner.run_phase(phase)
            runner._collect()
            final = reload_state(ws.path("run"))
            match = final["rounds"][0]["matches"][0]
            verdict = json.load(open(os.path.join(
                ws.path("run"), "r1", match["id"] + ".verdict.json")))
            self.assertEqual(match["winner"], verdict["winner"])
            self.assertEqual(match["source"], "judge")


class ArtifactPersistence(unittest.TestCase):
    def test_every_artifact_is_on_disk_and_listed_in_the_runlog(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=200), backoff=0.0)
            runner.run_to_completion()
            log = json.load(open(ws.path("run", "runlog.json")))
            for entry in log["jobs"].values():
                if entry["status"] != "done":
                    continue
                self.assertTrue(entry["artifacts"], "%s has no artifacts" % entry["job_id"])
                for rel in entry["artifacts"]:
                    self.assertTrue(os.path.exists(os.path.join(ws.path("run"), rel)),
                                    "%s is missing" % rel)

    def test_provenance_is_recorded_for_every_request(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            runner = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=200), backoff=0.0)
            runner.run_to_completion()
            log = json.load(open(ws.path("run", "runlog.json")))
            for entry in log["jobs"].values():
                if entry["status"] != "done":
                    continue
                self.assertIn("request_id", entry)
                self.assertIn("model", entry)
                self.assertIn("context_fingerprint", entry)
                self.assertIn("usage", entry["result"])
                self.assertNotIn("prompt", entry["request"])

    def test_engine_state_and_runlog_agree_on_completion(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            result = TournamentRunner(state, ScriptedMockProvider(),
                                      budget=Budget(max_calls=200),
                                      backoff=0.0).run_to_completion()
            final = reload_state(ws.path("run"))
            log = json.load(open(ws.path("run", "runlog.json")))
            self.assertIsNotNone(final["champion"])
            self.assertEqual(final["champion"], result["champion"])
            self.assertEqual(log["independence"]["level"], 1)
            self.assertEqual(log["engine"]["modified_by_migration"], False)

    def test_independence_report_reflects_the_actual_run(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=13)
            provider = ScriptedMockProvider()
            runner = TournamentRunner(state, provider, budget=Budget(max_calls=200),
                                      backoff=0.0)
            runner.run_to_completion()
            report = independence_report(RunLog(ws.path("run")), provider)
            self.assertEqual(report["competitor_jobs"], 4)
            self.assertEqual(report["distinct_competitor_contexts"], 4)
            self.assertFalse(report["live_provider"])
            self.assertTrue(any("NOT different underlying models" in c
                                for c in report["caveats"]))


class LivePathIsDistinguished(unittest.TestCase):
    def test_mock_provider_is_not_reported_as_live(self):
        report = {"live": ScriptedMockProvider().describe()["live"]}
        self.assertFalse(report["live"])

    def test_http_provider_is_reported_as_live(self):
        from arena_runner.http_provider import AnthropicProvider
        self.assertTrue(AnthropicProvider(model="m").describe()["live"])

    def test_runlog_records_whether_the_provider_was_live(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7)
            runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
            runner.run_to_completion()
            log = json.load(open(ws.path("run", "runlog.json")))
            self.assertFalse(log["provider"]["live"])
            self.assertEqual(log["provider"]["provider"], "scripted-mock")


if __name__ == "__main__":
    unittest.main()
