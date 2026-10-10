"""HTTP provider adapters, standard library only.

Two documented API shapes are supported:

  openai-compatible  POST {base_url}/chat/completions   (OpenAI, and any
                     endpoint that implements the same contract)
  anthropic          POST {base_url}/v1/messages

Authentication is header based and read from the environment. The key is never
placed in prompt text, never written to an artifact, and never included in an
error message.

Retries here are transport-level only (network errors, 429, 5xx) and always
bounded. A 4xx that is not 429 is a hard failure: the runner does not switch
provider or model behind the operator's back.
"""

import json
import os
import time
import urllib.error
import urllib.request

from .errors import ProviderError, ProviderTimeout, RateLimited
from .providers import CompletionResult, ProviderAdapter
from .redact import redact

DEFAULT_TIMEOUT = 120
MAX_TRANSPORT_ATTEMPTS = 4


class HTTPProvider(ProviderAdapter):
    """Base HTTP adapter. Subclasses build the request body and parse the reply."""

    live = True
    endpoint_suffix = ""
    api_version = None

    def __init__(self, model, api_key_env=None, api_key=None, base_url=None,
                 timeout=DEFAULT_TIMEOUT, max_attempts=MAX_TRANSPORT_ATTEMPTS,
                 max_tokens=4096, temperature=0.9, opener=None):
        super().__init__(model, default_max_tokens=max_tokens,
                         default_temperature=temperature)
        self.api_key_env = api_key_env
        self._api_key = api_key
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.opener = opener or urllib.request.urlopen
        self._secret_cache = None

    default_base_url = ""

    # -- credentials ---------------------------------------------------------
    def api_key(self):
        if self._api_key is not None:
            return self._api_key
        if not self.api_key_env:
            raise ProviderError(
                "%s: no api_key and no api_key_env configured" % self.name,
                retryable=False)
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ProviderError(
                "%s: environment variable %s is not set. Set it in your GitHub "
                "Actions secrets or shell environment; it is never stored in "
                "the repository." % (self.name, self.api_key_env),
                retryable=False)
        return value

    def _secrets(self):
        if self._secret_cache is None:
            key = self._api_key or (os.environ.get(self.api_key_env) if self.api_key_env else None)
            self._secret_cache = [key] if key else []
        return self._secret_cache

    def _safe(self, text):
        return redact(text, self._secrets())

    # -- request building ----------------------------------------------------
    def url(self):
        return self.base_url + self.endpoint_suffix

    def headers(self):
        raise NotImplementedError

    def body(self, request, model, max_tokens):
        raise NotImplementedError

    def parse(self, payload, status):
        raise NotImplementedError

    def extract_request_id(self, headers):
        for name in ("x-request-id", "request-id", "x-goog-request-id",
                     "anthropic-request-id", "openai-request-id"):
            if headers.get(name):
                return headers.get(name)
        return None

    # -- execution -----------------------------------------------------------
    def generate(self, request):
        model = self.resolve_model(request)
        max_tokens = request.max_tokens or self.default_max_tokens
        payload = json.dumps(self.body(request, model, max_tokens)).encode("utf-8")
        headers = self.headers()
        headers["Content-Type"] = "application/json"

        last_error = None
        started = time.monotonic()
        for attempt in range(1, self.max_attempts + 1):
            try:
                req = urllib.request.Request(self.url(), data=payload,
                                             headers=headers, method="POST")
                with self.opener(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    status = getattr(resp, "status", None) or resp.getcode()
                    request_id = self.extract_request_id(resp.headers or {})
                data = json.loads(raw)
                text = self.parse(data, status)
                usage = self.extract_usage(data)
                return CompletionResult(
                    text=text, provider=self.name, model=model,
                    request_id=request_id, usage=usage,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    status=status, attempts=attempt)
            except RateLimited as exc:
                last_error = exc
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                except Exception:
                    pass
                detail = self._safe(detail)
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        retry_after = float(retry_after) if retry_after else None
                    except ValueError:
                        retry_after = None
                    last_error = RateLimited(
                        "%s: 429 %s" % (self.name, detail), retry_after=retry_after)
                elif 500 <= exc.code < 600:
                    last_error = ProviderError(
                        "%s: HTTP %d %s" % (self.name, exc.code, detail),
                        status=exc.code, retryable=True)
                else:
                    # 4xx other than 429: configuration or request problem.
                    # Never retried, never silently switched to another provider.
                    raise ProviderError(
                        "%s: HTTP %d %s" % (self.name, exc.code, detail),
                        status=exc.code, retryable=False)
            except urllib.error.URLError as exc:
                last_error = ProviderTimeout(
                    "%s: %s" % (self.name, self._safe(str(getattr(exc, "reason", exc)))))
            except (json.JSONDecodeError, TimeoutError) as exc:
                last_error = ProviderError(
                    "%s: %s" % (self.name, self._safe(str(exc))), retryable=True)

            if attempt < self.max_attempts:
                delay = getattr(last_error, "retry_after", None)
                if delay is None:
                    delay = min(30.0, 1.5 ** (attempt - 1))
                time.sleep(delay)

        raise last_error if last_error else ProviderError(
            "%s: exhausted %d attempts" % (self.name, self.max_attempts))

    def extract_usage(self, data):
        return {}


class OpenAICompatibleProvider(HTTPProvider):
    """POST /chat/completions. Works with OpenAI and compatible gateways.

    Override `base_url` to point at any endpoint implementing this contract.
    """

    name = "openai-compatible"
    default_base_url = "https://api.openai.com/v1"
    endpoint_suffix = "/chat/completions"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("api_key_env", "ARENA_OPENAI_API_KEY")
        super().__init__(*args, **kwargs)

    def headers(self):
        return {"Authorization": "Bearer %s" % self.api_key()}

    def body(self, request, model, max_tokens):
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": request.temperature if request.temperature is not None
            else self.default_temperature,
            "user": "arena-%s" % request.job_id,
        }

    def parse(self, payload, status):
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "openai-compatible: unexpected response shape (%s)" % exc,
                status=status, retryable=False)

    def extract_usage(self, data):
        usage = data.get("usage") or {}
        return {"input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)}


class AnthropicProvider(HTTPProvider):
    """POST /v1/messages."""

    name = "anthropic"
    default_base_url = "https://api.anthropic.com"
    endpoint_suffix = "/v1/messages"
    api_version = "2023-06-01"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("api_key_env", "ARENA_ANTHROPIC_API_KEY")
        super().__init__(*args, **kwargs)

    def headers(self):
        return {"x-api-key": self.api_key(),
                "anthropic-version": self.api_version}

    def body(self, request, model, max_tokens):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": max_tokens,
            "temperature": request.temperature if request.temperature is not None
            else self.default_temperature,
        }
        if request.system:
            body["system"] = request.system
        return body

    def parse(self, payload, status):
        try:
            blocks = payload["content"]
        except (KeyError, TypeError):
            raise ProviderError("anthropic: unexpected response shape",
                                status=status, retryable=False)
        parts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "".join(parts)

    def extract_usage(self, data):
        usage = data.get("usage") or {}
        return {"input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0)}


def build_registry():
    """Registry with the offline mock plus the two HTTP adapters."""
    from .providers import MockProvider, Registry, ScriptedMockProvider

    return (Registry()
            .register("mock", lambda **kw: MockProvider(**kw))
            .register("scripted-mock", lambda **kw: ScriptedMockProvider(**kw))
            .register("openai-compatible", lambda **kw: OpenAICompatibleProvider(**kw))
            .register("anthropic", lambda **kw: AnthropicProvider(**kw)))
