"""Budget enforcement and secret redaction."""

import os
import unittest

from helper import TempWorkspace, make_state  # noqa: F401

from arena_runner.budget import Budget
from arena_runner.errors import BudgetExceeded
from arena_runner.providers import CompletionResult, ScriptedMockProvider
from arena_runner.redact import (assert_clean, redact, redact_mapping,
                                 secret_values_from_env)
from arena_runner.runner import TournamentRunner

# Split so no contiguous credential-shaped literal is committed.
SECRET = "sk-" + "live-abcdef0123456789abcdef0123"
GH_TOKEN = "ghp_" + "0123456789abcdefghijklmnopqrstuv"


class CallAndTokenLimits(unittest.TestCase):
    def test_call_limit_is_enforced_before_the_call(self):
        budget = Budget(max_calls=2)
        for _ in range(2):
            budget.check_before_call(100)
            budget.record_call()
        with self.assertRaises(BudgetExceeded) as ctx:
            budget.check_before_call(100)
        self.assertEqual(ctx.exception.kind, "budget_exceeded")

    def test_prompt_size_limit_is_enforced(self):
        budget = Budget(max_input_chars=100)
        budget.check_before_call(99)
        with self.assertRaises(BudgetExceeded):
            budget.check_before_call(101)

    def test_per_call_output_token_limit_is_enforced(self):
        budget = Budget(max_output_tokens_per_call=512)
        budget.check_before_call(10, 512)
        with self.assertRaises(BudgetExceeded):
            budget.check_before_call(10, 1024)

    def test_total_token_limit_is_enforced(self):
        budget = Budget(max_total_tokens=100)
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"input_tokens": 60,
                                                   "output_tokens": 50}))
        with self.assertRaises(BudgetExceeded):
            budget.check_before_call(10)

    def test_time_limit_is_enforced(self):
        budget = Budget(max_seconds=0)
        with self.assertRaises(BudgetExceeded):
            budget.check_before_call(10)

    def test_retry_limit_is_enforced(self):
        budget = Budget(max_retries=1)
        budget.record_retry()
        with self.assertRaises(BudgetExceeded):
            budget.check_before_retry()

    def test_no_unlimited_retry_is_possible(self):
        budget = Budget()          # defaults
        self.assertIsNotNone(budget.max_retries)

    def test_stopped_run_stays_stopped(self):
        budget = Budget(max_calls=1)
        budget.check_before_call(10)
        budget.record_call()
        with self.assertRaises(BudgetExceeded):
            budget.check_before_call(10)
        with self.assertRaises(BudgetExceeded):
            budget.check_before_call(10)
        self.assertIn("call limit", budget.stopped_reason)

    def test_summary_reports_limits_and_usage(self):
        budget = Budget(max_calls=10, max_seconds=60)
        budget.check_before_call(50)
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"input_tokens": 5,
                                                   "output_tokens": 7}))
        summary = budget.summary()
        self.assertEqual(summary["calls"], 1)
        self.assertEqual(summary["total_tokens"], 12)
        self.assertEqual(summary["limits"]["max_calls"], 10)


class CostEstimates(unittest.TestCase):
    def test_no_estimate_without_pricing(self):
        budget = Budget()
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"input_tokens": 1000,
                                                   "output_tokens": 1000}))
        self.assertIsNone(budget.estimated_cost())
        self.assertFalse(budget.summary()["estimated_cost_is_estimate"])

    def test_estimate_is_computed_from_supplied_pricing(self):
        budget = Budget(price_per_million_input=3.0, price_per_million_output=15.0)
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"input_tokens": 1_000_000,
                                                   "output_tokens": 1_000_000}))
        self.assertAlmostEqual(budget.estimated_cost(), 18.0)

    def test_estimate_is_labelled_as_an_estimate(self):
        budget = Budget(price_per_million_input=3.0)
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"input_tokens": 10}))
        summary = budget.summary()
        self.assertTrue(summary["estimated_cost_is_estimate"])
        self.assertIsNotNone(summary["estimated_cost"])

    def test_estimate_limit_stops_the_run_and_says_it_is_not_a_hard_cap(self):
        budget = Budget(price_per_million_output=1000.0, max_estimated_usd=0.001)
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"input_tokens": 0,
                                                   "output_tokens": 10_000}))
        with self.assertRaises(BudgetExceeded) as ctx:
            budget.check_estimated_cost()
        self.assertIn("estimate", str(ctx.exception).lower())

    def test_no_hard_spend_cap_is_claimed_without_pricing(self):
        budget = Budget(max_estimated_usd=1.0)      # no prices configured
        budget.record_call(CompletionResult("x", "p", "m",
                                            usage={"output_tokens": 10_000_000}))
        budget.check_estimated_cost()               # must not raise
        self.assertIsNone(budget.estimated_cost())


class Redaction(unittest.TestCase):
    def test_openai_style_key_is_redacted(self):
        self.assertNotIn(SECRET, redact("key is %s" % SECRET))
        self.assertIn("[REDACTED]", redact("key is %s" % SECRET))

    def test_github_token_is_redacted(self):
        self.assertIn("[REDACTED]", redact("token %s here" % GH_TOKEN))

    def test_bearer_header_is_redacted(self):
        self.assertIn("[REDACTED]",
                      redact("Authorization: Bearer abcdef0123456789abcdef"))

    def test_key_value_assignment_is_redacted(self):
        self.assertIn("[REDACTED]", redact("api_key=supersecretvalue123"))

    def test_explicit_secret_values_are_redacted(self):
        text = "the value is hunter2hunter2hunter2"
        self.assertIn("[REDACTED]", redact(text, secrets=["hunter2hunter2hunter2"]))

    def test_plain_text_is_untouched(self):
        text = "Cut CI build times with a cache that works."
        self.assertEqual(redact(text), text)

    def test_nested_structures_are_redacted(self):
        data = {"a": ["safe", SECRET],
                "b": {"c": "bearer abcdef0123456789xyz"}}
        scrubbed = redact_mapping(data)
        self.assertNotIn(SECRET, str(scrubbed))
        self.assertEqual(scrubbed["a"][0], "safe")

    def test_assert_clean_raises_on_a_leak(self):
        from arena_runner.errors import SecretLeakAttempt
        with self.assertRaises(SecretLeakAttempt):
            assert_clean("leaked %s" % SECRET)

    def test_assert_clean_passes_on_clean_text(self):
        self.assertTrue(assert_clean("no secrets here"))

    def test_secret_shaped_env_vars_are_collected(self):
        os.environ["ARENA_TEST_SECRET_VALUE"] = "value-that-is-a-secret-123"
        try:
            values = secret_values_from_env()
            self.assertIn("value-that-is-a-secret-123", values)
        finally:
            del os.environ["ARENA_TEST_SECRET_VALUE"]

    def test_short_env_values_are_ignored(self):
        os.environ["ARENA_TEST_TINY"] = "abc"
        try:
            self.assertNotIn("abc", secret_values_from_env())
        finally:
            del os.environ["ARENA_TEST_TINY"]


class SecretsNeverReachArtifacts(unittest.TestCase):
    def test_runlog_contains_no_secret_after_a_full_run(self):
        os.environ["ARENA_ANTHROPIC_API_KEY"] = SECRET
        try:
            with TempWorkspace() as ws:
                state = make_state(ws, agents=2, seed=7)
                runner = TournamentRunner(state, ScriptedMockProvider(),
                                          budget=Budget(max_calls=100), backoff=0.0)
                runner.run_to_completion()
                raw = open(ws.path("run", "runlog.json"), encoding="utf-8").read()
                self.assertNotIn(SECRET, raw)
                for path, _dirs, files in os.walk(ws.path("run")):
                    for name in files:
                        body = open(os.path.join(path, name),
                                    encoding="utf-8", errors="replace").read()
                        self.assertNotIn(SECRET, body,
                                         "secret found in %s" % name)
        finally:
            del os.environ["ARENA_ANTHROPIC_API_KEY"]

    def test_prompt_does_not_contain_credentials(self):
        os.environ["ARENA_ANTHROPIC_API_KEY"] = SECRET
        try:
            with TempWorkspace() as ws:
                state = make_state(ws, agents=2, seed=7)
                runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
                from arena_runner.bracket import write_prompts
                job = write_prompts(state, "spawn")[0]
                prompt, _iso = runner.build_prompt(job, open(job["prompt"]).read())
                self.assertNotIn(SECRET, prompt)
        finally:
            del os.environ["ARENA_ANTHROPIC_API_KEY"]


if __name__ == "__main__":
    unittest.main()
