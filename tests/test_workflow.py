"""GitHub Actions input validation and workflow safety properties.

Two layers are tested:

  1. arena_runner/workflow_inputs.py, the validator the workflow actually calls.
     Fully unit tested.
  2. The workflow YAML files themselves. PyYAML is not installed in this
     environment, so these are structural assertions over the file text, not
     schema validation. That limitation is deliberate and stated rather than
     hidden: it checks the safety-critical properties (triggers, permissions,
     concurrency, timeout, secret handling) without pretending to parse YAML.
"""

import os
import re
import unittest

from helper import REPO  # noqa: F401

from arena_runner.errors import ArenaRunnerError
from arena_runner.workflow_inputs import (ALLOWED_PROVIDERS, CONFIRM_TOKEN, LIMITS,
                                          PROVIDER_SECRET_ENV, as_json,
                                          expected_calls, summary_line, validate)

WORKFLOWS = os.path.join(REPO, ".github", "workflows")


def base_inputs(**overrides):
    inputs = {
        "task": "Write a 12-word headline for a CI caching pricing page.",
        "agents": "2",
        "provider": "scripted-mock",
        "model": "mock-model-1",
    }
    inputs.update(overrides)
    return inputs


class InputValidation(unittest.TestCase):
    def test_minimal_offline_input_is_accepted(self):
        config = validate(base_inputs())
        self.assertEqual(config["agents"], 2)
        self.assertFalse(config["live"])
        self.assertEqual(config["provider"], "scripted-mock")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ArenaRunnerError) as ctx:
            validate(base_inputs(provider="geminis-maybe"))
        self.assertIn("provider must be one of", str(ctx.exception))
        self.assertIn(ALLOWED_PROVIDERS[0], str(ctx.exception))

    def test_missing_model_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(model=""))

    def test_missing_task_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(task="   "))

    def test_oversized_task_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(task="x" * 20001))

    def test_oversized_model_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(model="m" * 201))

    def test_non_numeric_agent_count_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(agents="lots"))

    def test_agent_count_is_bounded(self):
        lo, hi = LIMITS["agents"]
        self.assertEqual(validate(base_inputs(agents=str(lo)))["agents"], lo)
        big = base_inputs(agents=str(hi), max_calls=str(expected_calls(hi)))
        self.assertEqual(validate(big)["agents"], hi)
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(agents=str(hi + 1),
                                 max_calls=str(expected_calls(hi + 1))))
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(agents="0"))

    def test_the_expensive_default_is_not_offered(self):
        self.assertLess(LIMITS["agents"][1], 100)

    def test_every_numeric_input_is_bounded(self):
        for name, (lo, hi) in LIMITS.items():
            with self.assertRaises(ArenaRunnerError, msg=name):
                validate(base_inputs(**{name: str(hi + 1)}))

    def test_seed_is_optional_and_bounded(self):
        self.assertIsNone(validate(base_inputs())["seed"])
        self.assertEqual(validate(base_inputs(seed="42"))["seed"], 42)
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(seed="0"))
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(seed="1000000"))

    def test_negative_price_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(price_in="-1"))

    def test_non_mapping_input_is_rejected(self):
        with self.assertRaises(ArenaRunnerError):
            validate(["not", "a", "mapping"])


class LiveRunGates(unittest.TestCase):
    def test_live_provider_needs_the_confirmation_token(self):
        inputs = base_inputs(provider="anthropic", model="claude-x")
        with self.assertRaises(ArenaRunnerError) as ctx:
            validate(inputs, secrets_present=["ARENA_ANTHROPIC_API_KEY"])
        self.assertIn("confirmation", str(ctx.exception))

    def test_live_provider_needs_its_secret(self):
        inputs = base_inputs(provider="anthropic", model="claude-x",
                             confirm=CONFIRM_TOKEN)
        with self.assertRaises(ArenaRunnerError) as ctx:
            validate(inputs, secrets_present=[])
        self.assertIn("ARENA_ANTHROPIC_API_KEY", str(ctx.exception))

    def test_live_provider_with_token_and_secret_is_accepted(self):
        config = validate(
            base_inputs(provider="anthropic", model="claude-x", confirm=CONFIRM_TOKEN),
            secrets_present=["ARENA_ANTHROPIC_API_KEY"])
        self.assertTrue(config["live"])
        self.assertTrue(config["confirmed"])
        self.assertEqual(config["secret_env"], "ARENA_ANTHROPIC_API_KEY")

    def test_openai_provider_maps_to_its_own_secret(self):
        config = validate(
            base_inputs(provider="openai-compatible", model="gpt-x",
                        confirm=CONFIRM_TOKEN),
            secrets_present=["ARENA_OPENAI_API_KEY"])
        self.assertEqual(config["secret_env"], PROVIDER_SECRET_ENV["openai-compatible"])

    def test_the_wrong_secret_does_not_satisfy_the_provider(self):
        with self.assertRaises(ArenaRunnerError):
            validate(base_inputs(provider="anthropic", model="claude-x",
                                 confirm=CONFIRM_TOKEN),
                     secrets_present=["ARENA_OPENAI_API_KEY"])

    def test_offline_provider_needs_no_confirmation_or_secret(self):
        config = validate(base_inputs(provider="scripted-mock"))
        self.assertFalse(config["live"])
        self.assertIsNone(config["secret_env"])

    def test_every_live_provider_has_a_secret_mapping(self):
        from arena_runner.workflow_inputs import LIVE_PROVIDERS
        for provider in LIVE_PROVIDERS:
            self.assertIn(provider, PROVIDER_SECRET_ENV)

    def test_max_calls_must_cover_the_tournament(self):
        inputs = base_inputs(agents="4", max_calls="3")
        with self.assertRaises(ArenaRunnerError) as ctx:
            validate(inputs)
        self.assertIn("max_calls", str(ctx.exception))


class CallEstimates(unittest.TestCase):
    def test_expected_calls_matches_the_engine(self):
        from arena_runner.bracket import plan_totals
        for agents in (2, 3, 4, 7, 8, 16, 64):
            self.assertEqual(expected_calls(agents),
                             plan_totals(agents)["calls"],
                             "estimate disagrees with the engine at %d agents" % agents)

    def test_baseline_adds_exactly_one_call(self):
        self.assertEqual(expected_calls(4, baseline=True),
                         expected_calls(4, baseline=False) + 1)

    def test_two_agent_smoke_test_is_small(self):
        self.assertLessEqual(expected_calls(2, baseline=True), 12)

    def test_summary_line_says_live_or_offline(self):
        offline = summary_line(validate(base_inputs()))
        self.assertIn("OFFLINE", offline)
        self.assertIn("No spend", offline)
        live = summary_line(validate(
            base_inputs(provider="anthropic", model="claude-x", confirm=CONFIRM_TOKEN),
            secrets_present=["ARENA_ANTHROPIC_API_KEY"]))
        self.assertIn("LIVE", live)
        self.assertIn("paid", live)

    def test_serialised_form_omits_the_task_body(self):
        text = as_json(validate(base_inputs(task="SECRET TASK TEXT")))
        self.assertNotIn("SECRET TASK TEXT", text)
        self.assertIn("chars", text)


class WorkflowFiles(unittest.TestCase):
    """Structural assertions. PyYAML is unavailable here, so these read the file
    text directly rather than parsing a schema."""

    def read(self, name):
        with open(os.path.join(WORKFLOWS, name), encoding="utf-8") as fh:
            return fh.read()

    def test_the_three_expected_workflows_exist(self):
        for name in ("ci.yml", "arena-tournament.yml", "live-smoke.yml"):
            self.assertTrue(os.path.exists(os.path.join(WORKFLOWS, name)),
                            "%s is missing" % name)

    def test_tournament_workflow_is_manual_only(self):
        text = self.read("arena-tournament.yml")
        on_block = text.split("on:", 1)[1].split("permissions:", 1)[0]
        self.assertIn("workflow_dispatch", on_block)
        self.assertNotIn("push", on_block)
        self.assertNotIn("pull_request", on_block)
        self.assertNotIn("schedule", on_block)

    def test_live_smoke_is_manual_only(self):
        text = self.read("live-smoke.yml")
        on_block = text.split("on:", 1)[1].split("permissions:", 1)[0]
        self.assertIn("workflow_dispatch", on_block)
        self.assertNotIn("push", on_block)
        self.assertNotIn("pull_request", on_block)

    def test_ci_does_not_run_tournaments(self):
        text = self.read("ci.yml")
        self.assertIn("python3 -m unittest discover -s tests", text)
        self.assertNotIn("arena_runner workflow", text)

    def test_ci_needs_no_secrets(self):
        text = self.read("ci.yml")
        self.assertNotIn("secrets.", text)

    def test_permissions_are_least_privilege(self):
        for name in ("ci.yml", "arena-tournament.yml", "live-smoke.yml"):
            text = self.read(name)
            self.assertIn("permissions:", text, name)
            self.assertIn("contents: read", text, name)
            self.assertNotIn("contents: write", text, name)
            self.assertNotIn("actions: write", text, name)
            self.assertNotIn("id-token: write", text, name)

    def test_runs_are_time_bounded(self):
        for name in ("ci.yml", "arena-tournament.yml", "live-smoke.yml"):
            self.assertIn("timeout-minutes:", self.read(name), name)

    def test_parallel_tournaments_are_prevented(self):
        for name in ("arena-tournament.yml", "live-smoke.yml"):
            text = self.read(name)
            self.assertIn("concurrency:", text, name)
            self.assertIn("cancel-in-progress: false", text, name)

    def test_no_hardcoded_credentials(self):
        pattern = re.compile(
            r"(sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,}|"
            r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
        for name in os.listdir(WORKFLOWS):
            text = self.read(name)
            self.assertIsNone(pattern.search(text),
                              "a credential-shaped string appears in %s" % name)

    def test_credentials_come_from_secrets_only(self):
        text = self.read("arena-tournament.yml")
        self.assertIn("secrets.ARENA_OPENAI_API_KEY", text)
        self.assertIn("secrets.ARENA_ANTHROPIC_API_KEY", text)
        self.assertNotIn("echo $ARENA_OPENAI_API_KEY", text)
        self.assertNotIn("echo $ARENA_ANTHROPIC_API_KEY", text)

    def test_inputs_reach_the_job_through_the_environment(self):
        text = self.read("arena-tournament.yml")
        self.assertIn("ARENA_IN_TASK: ${{ inputs.task }}", text)
        self.assertIn("python3 -m arena_runner workflow", text)
        # Untrusted input must not be interpolated into a shell command line.
        self.assertNotIn('run: .*${{ inputs.', text)

    def test_confirm_input_is_documented_with_the_exact_token(self):
        self.assertIn(CONFIRM_TOKEN, self.read("arena-tournament.yml"))
        self.assertIn(CONFIRM_TOKEN, self.read("live-smoke.yml"))

    def test_artifacts_are_uploaded_even_on_failure(self):
        text = self.read("arena-tournament.yml")
        self.assertIn("if: always()", text)
        self.assertIn("actions/upload-artifact", text)
        self.assertIn("runlog.json", text)

    def test_live_smoke_is_small(self):
        text = self.read("live-smoke.yml")
        self.assertIn('ARENA_IN_AGENTS: "2"', text)
        self.assertIn('ARENA_IN_MAX_CALLS: "20"', text)

    def test_gitignore_covers_local_state_and_env_files(self):
        with open(os.path.join(REPO, ".gitignore"), encoding="utf-8") as fh:
            ignored = fh.read()
        self.assertIn(".arena/", ignored)
        self.assertIn(".env", ignored)


if __name__ == "__main__":
    unittest.main()
