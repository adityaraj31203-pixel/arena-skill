"""Provider adapter interface and a deterministic mock.

The contract every provider must satisfy:

    generate(request) -> CompletionResult

Providers own their authentication, request shape, retries at the transport
level, rate-limit handling and response parsing. The runner never reaches into
provider internals, and never silently substitutes a different provider or model
when a request fails: a failure is raised, recorded, and surfaced.

No credentials are ever accepted as prompt text. Adapters read them from the
environment or from an explicit config dict supplied by the operator.
"""

import hashlib
import time

from .errors import MalformedResponse, ProviderError, ProviderTimeout, RateLimited


class CompletionRequest:
    """One stateless request. Carries no conversation history, ever."""

    __slots__ = ("job_id", "role", "system", "prompt", "max_tokens",
                 "temperature", "model", "metadata")

    def __init__(self, job_id, role, prompt, system=None, max_tokens=4096,
                 temperature=0.9, model=None, metadata=None):
        self.job_id = job_id
        self.role = role                      # competitor|attacker|defender|judge|final
        self.system = system
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.metadata = dict(metadata or {})

    def context_fingerprint(self):
        """Stable digest of exactly what this request was sent.

        Two requests with the same fingerprint saw the same context. This is how
        the runner proves competitors did not share a context.
        """
        blob = "\x00".join([
            self.role,
            self.system or "",
            self.prompt,
            str(self.max_tokens),
            str(self.temperature),
            self.model or "",
        ]).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def to_public_dict(self):
        """Provenance record. Deliberately excludes the prompt body."""
        return {
            "job_id": self.job_id,
            "role": self.role,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "context_fingerprint": self.context_fingerprint(),
            "prompt_chars": len(self.prompt),
            "metadata": dict(self.metadata),
        }


class CompletionResult:
    __slots__ = ("text", "provider", "model", "request_id", "usage",
                 "latency_ms", "status", "attempts")

    def __init__(self, text, provider, model, request_id=None, usage=None,
                 latency_ms=0, status=200, attempts=1):
        self.text = text
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.usage = dict(usage or {})
        self.latency_ms = latency_ms
        self.status = status
        self.attempts = attempts

    @property
    def input_tokens(self):
        return int(self.usage.get("input_tokens", 0) or 0)

    @property
    def output_tokens(self):
        return int(self.usage.get("output_tokens", 0) or 0)

    def to_public_dict(self):
        return {
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "usage": dict(self.usage),
            "latency_ms": self.latency_ms,
            "status": self.status,
            "attempts": self.attempts,
        }


class ProviderAdapter:
    """Base class. Subclasses set `name` and implement `generate`."""

    name = "abstract"
    #: True when this adapter talks to a real paid API.
    live = False

    def __init__(self, model, default_max_tokens=4096, default_temperature=0.9):
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

    def generate(self, request):
        raise NotImplementedError

    def resolve_model(self, request):
        return request.model or self.model

    def describe(self):
        return {"provider": self.name, "model": self.model, "live": self.live}


# ---------------------------------------------------------------------------
# Mock provider: deterministic, offline, free. Used for all CI tests.
# ---------------------------------------------------------------------------

class MockProvider(ProviderAdapter):
    """Deterministic offline provider.

    It is explicit about what it is: a test double. Outputs are a pure function
    of (job_id, role, prompt digest), so tests are reproducible and two
    competitors with different prompts always get different text.

    `script` lets a test force specific behaviour per job_id:
        {"a001.spawn": ["rate_limit", "timeout", "ok"]}
    consumes one entry per attempt, so retry behaviour can be tested exactly.
    """

    name = "mock"
    live = False

    def __init__(self, model="mock-model-1", behaviour=None, script=None,
                 default_temperature=0.9, response_builder=None):
        super().__init__(model, default_temperature=default_temperature)
        self.behaviour = behaviour or "ok"     # ok|malformed|empty|timeout|rate_limit
        self.script = dict(script or {})
        self.response_builder = response_builder
        self.calls = []                        # full provenance of every call

    def _next_behaviour(self, request):
        queue = self.script.get(request.job_id)
        if queue:
            return queue.pop(0) if len(queue) > 1 else queue[0]
        return self.behaviour

    def _body(self, request):
        if self.response_builder:
            return self.response_builder(request)
        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:8]
        role = request.role
        if role == "competitor":
            return ("Solution for %s (mock, prompt %s).\n\nThis is a deterministic "
                    "test response. It addresses the task as stated.\n"
                    % (request.job_id, digest))
        if role == "attacker":
            return ("ATTACK 1 [MINOR] mock attack from %s (prompt %s)\n"
                    "Where: the solution\n"
                    "Problem: placeholder flaw recorded by the test double.\n"
                    % (request.job_id, digest))
        if role == "defender":
            return "DEFENSE:mock"
        if role == "judge":
            return "VERDICT:mock"
        return "mock response %s %s" % (request.job_id, digest)

    def generate(self, request):
        started = time.monotonic()
        behaviour = self._next_behaviour(request)
        attempt = len([c for c in self.calls if c["job_id"] == request.job_id]) + 1
        record = {"job_id": request.job_id, "role": request.role,
                  "attempt": attempt, "behaviour": behaviour,
                  "context_fingerprint": request.context_fingerprint()}
        self.calls.append(record)

        if behaviour == "rate_limit":
            raise RateLimited("mock 429", retry_after=0)
        if behaviour == "timeout":
            raise ProviderTimeout("mock timeout")
        if behaviour == "http_500":
            raise ProviderError("mock server error", status=500, retryable=True)
        if behaviour == "http_400":
            raise ProviderError("mock bad request", status=400, retryable=False)

        text = self._body(request)
        if behaviour == "malformed":
            text = "this is not the required output shape at all"
        elif behaviour == "empty":
            text = ""
        elif behaviour == "judge_bad_json":
            text = "{ not json"

        result = CompletionResult(
            text=text, provider=self.name, model=self.resolve_model(request),
            request_id="mock-%s-%d" % (hashlib.sha256(request.job_id.encode()).hexdigest()[:8], attempt),
            usage={"input_tokens": max(1, len(request.prompt) // 4),
                   "output_tokens": max(1, len(text) // 4)},
            latency_ms=int((time.monotonic() - started) * 1000),
            status=200, attempts=attempt,
        )
        record.update({"model": result.model, "request_id": result.request_id,
                       "output_tokens": result.output_tokens,
                       "input_tokens": result.input_tokens})
        if not text:
            raise MalformedResponse("provider returned an empty completion",
                                    detail={"job_id": request.job_id})
        return result


class Registry:
    """Named provider lookup. Fails loudly when nothing usable is configured."""

    def __init__(self):
        self._factories = {}

    def register(self, name, factory):
        self._factories[name] = factory
        return self

    def names(self):
        return sorted(self._factories)

    def create(self, name, **kwargs):
        if name not in self._factories:
            raise ProviderError(
                "unknown provider %r (available: %s)" % (name, ", ".join(self.names()) or "none"))
        return self._factories[name](**kwargs)


class ScriptedMockProvider(MockProvider):
    """A mock that produces schema-valid artifacts for every phase.

    This is the "local mocked-test mode": it lets the whole tournament, including
    blind judging and the final check, run offline and free, so the orchestration
    can be exercised end to end without credentials or spend. It is a test double
    and reports itself as one (`live = False`).

    Judge verdicts are derived from a hash of the prompt so the outcome is
    deterministic but not constant, and the two sides are never scored
    identically, which exercises the real tie-break paths.
    """

    name = "scripted-mock"

    def _scores(self, seed_text, base):
        h = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        return {
            "correctness": base + (h % 3),
            "completeness": base + ((h >> 2) % 3),
            "specificity": base + ((h >> 4) % 3),
            "robustness": base + ((h >> 6) % 3),
            "clarity": base + ((h >> 8) % 3),
            "fatal": False,
        }

    def _body(self, request):
        digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()[:8]
        role = request.role

        if role == "competitor":
            return ("# Solution (%s)\n\n"
                    "This deterministic test response addresses the task as stated. "
                    "It is long enough to satisfy the minimum solution length and "
                    "stands on its own without referring to any other competitor.\n"
                    % request.job_id)
        if role == "attacker":
            return ("ATTACK 1 [MAJOR] concrete flaw found by %s\n"
                    "Where: the first claim\n"
                    "Problem: the response asserts a requirement the task states is "
                    "not met. Counterexample: the stated edge case fails.\n\n"
                    "ATTACK 2 [MINOR] vague step from %s\n"
                    "Where: the closing section\n"
                    "Problem: the reader would have to guess the value.\n"
                    % (request.job_id, digest))
        if role == "defender":
            from .validation import ARTIFACT_SEP
            return ("ATTACK 1: CONCEDE. The flaw is real and the revised solution "
                    "fixes it explicitly.\n"
                    "ATTACK 2: REBUT. The task states this value, so the step is "
                    "concrete as written.\n"
                    "%s\n"
                    "# Revised solution (%s)\n\n"
                    "The complete revised solution, standalone, with the conceded "
                    "points fixed. It does not refer to a previous version.\n"
                    % (ARTIFACT_SEP, request.job_id))
        if role == "judge":
            import json as _json
            return _json.dumps({
                "match": "blinded",
                "scores": {"solA": self._scores(request.prompt + "A", 6),
                           "solB": self._scores(request.prompt + "B", 5)},
                "winner": "solA",
                "reason": "deterministic mock verdict: solA met more stated requirements",
                "survived": ["revised the attacked claim"],
                "standing": {"solA": [], "solB": ["one minor gap"]},
            }, indent=2)
        if role == "final":
            import json as _json
            return _json.dumps({
                "scores": {"X": self._scores(request.prompt + "X", 4),
                           "Y": self._scores(request.prompt + "Y", 7)},
                "winner": "Y",
                "reason": "deterministic mock final check",
                "fixed": ["names the concrete benefit", "drops the banned wording"],
            }, indent=2)
        return "mock response %s" % request.job_id
