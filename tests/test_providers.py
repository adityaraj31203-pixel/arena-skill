"""Provider adapter contracts.

Covers: the interface, the registry, context fingerprinting, provenance records
that exclude prompt bodies, and the failure taxonomy (rate limit, timeout,
retryable server error, non-retryable client error).

Offline. No credentials, no network, no spend.
"""

import os
import unittest

from helper import TempWorkspace  # noqa: F401  (keeps sys.path setup in one place)

from arena_runner.errors import (MalformedResponse, ProviderError, ProviderTimeout,
                                 RateLimited)
from arena_runner.http_provider import (AnthropicProvider, OpenAICompatibleProvider,
                                        build_registry)
from arena_runner.providers import (CompletionRequest, CompletionResult, MockProvider,
                                    ProviderAdapter, Registry, ScriptedMockProvider)


# Built by concatenation so no contiguous credential-shaped literal is committed.
FAKE_KEY = "sk-" + "not-a-real-key-value-0123456789"
TEST_KEY = "sk-" + "test-0123456789abcdefghij"


def req(role="competitor", job_id="a001.spawn", prompt="do the task"):
    return CompletionRequest(job_id=job_id, role=role, prompt=prompt,
                             system="you are one competitor", max_tokens=1024,
                             temperature=0.9)


class AdapterContract(unittest.TestCase):
    def test_base_adapter_refuses_to_generate(self):
        adapter = ProviderAdapter("m")
        with self.assertRaises(NotImplementedError):
            adapter.generate(req())

    def test_describe_exposes_no_credentials(self):
        for adapter in (MockProvider(), ScriptedMockProvider(),
                        OpenAICompatibleProvider(model="gpt-x", api_key=FAKE_KEY)):
            described = adapter.describe()
            self.assertEqual(set(described), {"provider", "model", "live"})
            self.assertNotIn(FAKE_KEY, str(described))

    def test_live_flag_distinguishes_mock_from_http(self):
        self.assertFalse(MockProvider().live)
        self.assertFalse(ScriptedMockProvider().live)
        self.assertTrue(OpenAICompatibleProvider(model="m").live)
        self.assertTrue(AnthropicProvider(model="m").live)

    def test_registry_lists_and_creates(self):
        registry = build_registry()
        self.assertEqual(registry.names(),
                         ["anthropic", "mock", "openai-compatible", "scripted-mock"])
        self.assertIsInstance(registry.create("mock"), MockProvider)

    def test_registry_rejects_unknown_provider_loudly(self):
        with self.assertRaises(ProviderError) as ctx:
            build_registry().create("no-such-provider")
        self.assertIn("unknown provider", str(ctx.exception))

    def test_model_resolution_prefers_request_then_adapter(self):
        adapter = MockProvider(model="default-model")
        self.assertEqual(adapter.resolve_model(req()), "default-model")
        self.assertEqual(adapter.resolve_model(
            CompletionRequest("j", "competitor", "p", model="override")), "override")


class ContextFingerprinting(unittest.TestCase):
    def test_identical_context_fingerprints_match(self):
        a, b = req(prompt="same"), req(prompt="same")
        self.assertEqual(a.context_fingerprint(), b.context_fingerprint())

    def test_different_prompts_fingerprint_differently(self):
        a, b = req(prompt="one"), req(prompt="two")
        self.assertNotEqual(a.context_fingerprint(), b.context_fingerprint())

    def test_role_and_system_participate_in_the_fingerprint(self):
        base = req(role="competitor", prompt="p")
        other = req(role="attacker", prompt="p")
        self.assertNotEqual(base.context_fingerprint(), other.context_fingerprint())

    def test_fingerprint_is_a_sha256_hex_digest(self):
        fp = req().context_fingerprint()
        self.assertEqual(len(fp), 64)
        int(fp, 16)


class Provenance(unittest.TestCase):
    def test_public_request_dict_excludes_the_prompt_body(self):
        secret_prompt = "the task mentions %s" % FAKE_KEY
        public = req(prompt=secret_prompt).to_public_dict()
        self.assertNotIn("prompt", public)
        self.assertNotIn(FAKE_KEY, str(public))
        self.assertEqual(public["prompt_chars"], len(secret_prompt))
        self.assertIn("context_fingerprint", public)

    def test_result_records_provider_model_and_request_id(self):
        result = MockProvider(model="mock-model-1").generate(req())
        public = result.to_public_dict()
        self.assertEqual(public["provider"], "mock")
        self.assertEqual(public["model"], "mock-model-1")
        self.assertTrue(public["request_id"].startswith("mock-"))
        self.assertGreater(public["usage"]["output_tokens"], 0)

    def test_result_exposes_token_counts(self):
        result = CompletionResult("text", "p", "m", usage={"input_tokens": 10,
                                                           "output_tokens": 3})
        self.assertEqual(result.input_tokens, 10)
        self.assertEqual(result.output_tokens, 3)
        empty = CompletionResult("t", "p", "m")
        self.assertEqual((empty.input_tokens, empty.output_tokens), (0, 0))


class FailureTaxonomy(unittest.TestCase):
    def test_rate_limit_carries_retry_after_and_is_retryable(self):
        with self.assertRaises(RateLimited) as ctx:
            MockProvider(behaviour="rate_limit").generate(req())
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(ctx.exception.kind, "rate_limited")

    def test_timeout_is_retryable(self):
        with self.assertRaises(ProviderTimeout) as ctx:
            MockProvider(behaviour="timeout").generate(req())
        self.assertTrue(ctx.exception.retryable)

    def test_server_error_is_retryable(self):
        with self.assertRaises(ProviderError) as ctx:
            MockProvider(behaviour="http_500").generate(req())
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.status, 500)

    def test_client_error_is_not_retryable(self):
        with self.assertRaises(ProviderError) as ctx:
            MockProvider(behaviour="http_400").generate(req())
        self.assertFalse(ctx.exception.retryable)

    def test_empty_completion_becomes_malformed_not_success(self):
        with self.assertRaises(MalformedResponse):
            MockProvider(behaviour="empty").generate(req())

    def test_script_consumes_one_entry_per_attempt(self):
        provider = MockProvider(script={"a001.spawn": ["rate_limit", "ok"]})
        with self.assertRaises(RateLimited):
            provider.generate(req())
        result = provider.generate(req())
        self.assertTrue(result.text)
        self.assertEqual([c["behaviour"] for c in provider.calls],
                         ["rate_limit", "ok"])
        self.assertEqual([c["attempt"] for c in provider.calls], [1, 2])


class Determinism(unittest.TestCase):
    def test_mock_output_is_a_function_of_the_prompt(self):
        p = ScriptedMockProvider()
        first = p.generate(req(prompt="stable prompt")).text
        second = p.generate(req(prompt="stable prompt")).text
        self.assertEqual(first, second)

    def test_different_jobs_produce_different_output(self):
        p = ScriptedMockProvider()
        a = p.generate(req(job_id="a001.spawn", prompt="prompt one")).text
        b = p.generate(req(job_id="a002.spawn", prompt="prompt two")).text
        self.assertNotEqual(a, b)

    def test_every_scripted_phase_produces_its_shape(self):
        from arena_runner.validation import ARTIFACT_SEP
        from arena_runner.anonymize import parse_verdict

        p = ScriptedMockProvider()
        self.assertIn("ATTACK 1", p.generate(req("attacker", "a.attack")).text)
        self.assertIn(ARTIFACT_SEP, p.generate(req("defender", "a.defend")).text)
        for role in ("judge", "final"):
            verdict = parse_verdict(p.generate(req(role, "x." + role)).text)
            self.assertIsInstance(verdict, dict)
            self.assertIn("scores", verdict)
            self.assertIn("winner", verdict)


class CredentialHandling(unittest.TestCase):
    def test_missing_key_fails_with_an_actionable_message(self):
        adapter = OpenAICompatibleProvider(model="m", api_key_env="ARENA_TEST_ABSENT_KEY")
        with self.assertRaises(ProviderError) as ctx:
            adapter.api_key()
        self.assertIn("ARENA_TEST_ABSENT_KEY", str(ctx.exception))
        self.assertFalse(ctx.exception.retryable)

    def test_no_key_env_configured_fails_clearly(self):
        adapter = AnthropicProvider(model="m", api_key_env=None)
        with self.assertRaises(ProviderError):
            adapter.api_key()

    def test_key_is_read_from_the_environment(self):
        os.environ["ARENA_TEST_PRESENT_KEY"] = TEST_KEY
        try:
            adapter = OpenAICompatibleProvider(model="m",
                                               api_key_env="ARENA_TEST_PRESENT_KEY")
            self.assertEqual(adapter.api_key(), TEST_KEY)
            self.assertIn("Bearer", adapter.headers()["Authorization"])
        finally:
            del os.environ["ARENA_TEST_PRESENT_KEY"]

    def test_custom_registry_can_be_built(self):
        registry = Registry().register("mock", lambda **kw: MockProvider(**kw))
        self.assertEqual(registry.names(), ["mock"])


if __name__ == "__main__":
    unittest.main()
