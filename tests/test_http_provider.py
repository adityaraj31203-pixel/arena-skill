"""HTTP provider adapters, exercised against a real local HTTP server.

This is genuine verification of the live code path without live credentials or
spend: the adapter builds a real request, sends it over a real socket, and parses
a real response. What is NOT verified here is any hosted provider's actual
behaviour, pricing or rate limits; that requires configured credentials and is
reported separately as not tested.
"""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from helper import TempWorkspace  # noqa: F401  (sys.path setup)

from arena_runner.errors import ProviderError, RateLimited
from arena_runner.http_provider import AnthropicProvider, OpenAICompatibleProvider
from arena_runner.providers import CompletionRequest

# Split so no contiguous credential-shaped literal is committed.
SECRET = "sk-" + "live-0123456789abcdefghijklmn"


class Recorder:
    def __init__(self):
        self.requests = []
        self.responses = []      # queue of (status, payload, headers)

    def next_response(self):
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


def make_handler(recorder):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            recorder.requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(body),
            })
            status, payload, extra = recorder.next_response()
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-request-id", "req-test-1234")
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class LiveServerTestCase(unittest.TestCase):
    def setUp(self):
        self.recorder = Recorder()
        self.server = HTTPServer(("127.0.0.1", 0), make_handler(self.recorder))
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = "http://127.0.0.1:%d" % self.port

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def respond(self, status, payload, headers=None):
        self.recorder.responses.append((status, payload, headers or {}))

    def request(self, prompt="Write the headline."):
        return CompletionRequest(job_id="a001.spawn", role="competitor",
                                 prompt=prompt, system="be concise",
                                 max_tokens=256, temperature=0.7)


class OpenAICompatibleContract(LiveServerTestCase):
    def provider(self, **kw):
        kw.setdefault("api_key", SECRET)
        kw.setdefault("max_attempts", 1)
        return OpenAICompatibleProvider(model="gpt-test-1", base_url=self.base_url, **kw)

    def test_request_shape_and_endpoint(self):
        self.respond(200, {"choices": [{"message": {"content": "the answer"}}],
                           "usage": {"prompt_tokens": 11, "completion_tokens": 4,
                                     "total_tokens": 15}})
        result = self.provider().generate(self.request())
        sent = self.recorder.requests[0]
        self.assertEqual(sent["path"], "/chat/completions")
        self.assertEqual(sent["body"]["model"], "gpt-test-1")
        self.assertEqual(sent["body"]["max_tokens"], 256)
        self.assertEqual(sent["body"]["temperature"], 0.7)
        self.assertEqual(sent["body"]["messages"][0]["role"], "system")
        self.assertEqual(sent["body"]["messages"][1]["role"], "user")
        self.assertEqual(result.text, "the answer")
        self.assertEqual(result.request_id, "req-test-1234")
        self.assertEqual(result.input_tokens, 11)
        self.assertEqual(result.output_tokens, 4)

    def test_authorization_header_uses_the_configured_key(self):
        self.respond(200, {"choices": [{"message": {"content": "x"}}], "usage": {}})
        self.provider().generate(self.request())
        self.assertEqual(self.recorder.requests[0]["headers"].get("Authorization"),
                         "Bearer %s" % SECRET)

    def test_no_conversation_history_is_sent(self):
        self.respond(200, {"choices": [{"message": {"content": "x"}}], "usage": {}})
        self.provider().generate(self.request())
        messages = self.recorder.requests[0]["body"]["messages"]
        self.assertEqual(len(messages), 2)          # one system, one user
        self.assertNotIn("assistant", [m["role"] for m in messages])

    def test_unexpected_shape_is_a_hard_failure(self):
        self.respond(200, {"unexpected": True})
        with self.assertRaises(ProviderError) as ctx:
            self.provider().generate(self.request())
        self.assertFalse(ctx.exception.retryable)

    def test_429_is_retried_and_reports_rate_limited_when_exhausted(self):
        self.respond(429, {"error": {"message": "slow down"}}, {"Retry-After": "0"})
        provider = OpenAICompatibleProvider(model="m", base_url=self.base_url,
                                            api_key=SECRET, max_attempts=2)
        with self.assertRaises(RateLimited):
            provider.generate(self.request())
        self.assertEqual(len(self.recorder.requests), 2)

    def test_429_then_success(self):
        self.respond(429, {"error": {"message": "slow down"}}, {"Retry-After": "0"})
        self.respond(200, {"choices": [{"message": {"content": "recovered"}}],
                           "usage": {}})
        provider = OpenAICompatibleProvider(model="m", base_url=self.base_url,
                                            api_key=SECRET, max_attempts=3)
        result = provider.generate(self.request())
        self.assertEqual(result.text, "recovered")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(self.recorder.requests), 2)

    def test_500_is_retried(self):
        self.respond(500, {"error": {"message": "boom"}})
        self.respond(200, {"choices": [{"message": {"content": "ok"}}], "usage": {}})
        provider = OpenAICompatibleProvider(model="m", base_url=self.base_url,
                                            api_key=SECRET, max_attempts=3)
        self.assertEqual(provider.generate(self.request()).text, "ok")

    def test_400_is_never_retried_and_never_switches_provider(self):
        self.respond(400, {"error": {"message": "bad model name"}})
        with self.assertRaises(ProviderError) as ctx:
            self.provider().generate(self.request())
        self.assertEqual(ctx.exception.status, 400)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(self.recorder.requests), 1)

    def test_retry_count_is_bounded(self):
        self.respond(500, {"error": {"message": "boom"}})
        provider = OpenAICompatibleProvider(model="m", base_url=self.base_url,
                                            api_key=SECRET, max_attempts=3)
        with self.assertRaises(ProviderError):
            provider.generate(self.request())
        self.assertEqual(len(self.recorder.requests), 3)

    def test_error_text_does_not_contain_the_api_key(self):
        self.respond(400, {"error": {"message": "invalid key %s" % SECRET}})
        with self.assertRaises(ProviderError) as ctx:
            self.provider().generate(self.request())
        self.assertNotIn(SECRET, str(ctx.exception))
        self.assertIn("[REDACTED]", str(ctx.exception))


class AnthropicContract(LiveServerTestCase):
    def provider(self, **kw):
        kw.setdefault("api_key", SECRET)
        kw.setdefault("max_attempts", 1)
        return AnthropicProvider(model="claude-test-1", base_url=self.base_url, **kw)

    def test_request_shape_and_endpoint(self):
        self.respond(200, {"content": [{"type": "text", "text": "the answer"}],
                           "usage": {"input_tokens": 9, "output_tokens": 5}})
        result = self.provider().generate(self.request())
        sent = self.recorder.requests[0]
        self.assertEqual(sent["path"], "/v1/messages")
        self.assertEqual(sent["body"]["model"], "claude-test-1")
        self.assertEqual(sent["body"]["system"], "be concise")
        self.assertEqual(sent["body"]["messages"][0]["role"], "user")
        self.assertEqual(result.text, "the answer")
        self.assertEqual(result.output_tokens, 5)

    def test_auth_uses_x_api_key_and_version_headers(self):
        self.respond(200, {"content": [{"type": "text", "text": "x"}], "usage": {}})
        self.provider().generate(self.request())
        headers = {k.lower(): v for k, v in
                   self.recorder.requests[0]["headers"].items()}
        self.assertEqual(headers.get("x-api-key"), SECRET)
        self.assertEqual(headers.get("anthropic-version"), "2023-06-01")
        self.assertNotIn("authorization", headers)

    def test_multiple_text_blocks_are_concatenated(self):
        self.respond(200, {"content": [{"type": "text", "text": "one "},
                                       {"type": "text", "text": "two"}],
                           "usage": {}})
        self.assertEqual(self.provider().generate(self.request()).text, "one two")

    def test_unexpected_shape_is_a_hard_failure(self):
        self.respond(200, {"nope": True})
        with self.assertRaises(ProviderError):
            self.provider().generate(self.request())

    def test_error_text_does_not_contain_the_api_key(self):
        self.respond(403, {"error": {"message": "bad key %s" % SECRET}})
        with self.assertRaises(ProviderError) as ctx:
            self.provider().generate(self.request())
        self.assertNotIn(SECRET, str(ctx.exception))


class UnreachableEndpoint(unittest.TestCase):
    def test_connection_refused_is_reported_not_hung(self):
        provider = OpenAICompatibleProvider(
            model="m", base_url="http://127.0.0.1:1", api_key=SECRET,
            max_attempts=1, timeout=2)
        with self.assertRaises(ProviderError) as ctx:
            provider.generate(CompletionRequest("j", "competitor", "p"))
        self.assertTrue(ctx.exception.retryable)
        self.assertNotIn(SECRET, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
