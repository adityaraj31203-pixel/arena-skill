"""Budget and cost controls.

What this can actually enforce, and what it cannot:

  HARD (enforced before the request is made)
    - maximum model calls
    - maximum input characters per request
    - maximum output tokens requested per call
    - maximum elapsed wall-clock seconds
    - maximum retries per job
    - maximum total input+output tokens, as reported by the provider

  SOFT (an estimate, always labelled as one)
    - monetary spend. Provider pricing is not knowable in advance, billing is
      asynchronous, and reported token counts arrive only after the call. A
      spend figure here is derived from operator-supplied per-million-token
      prices and a token estimate. It is a guide, not a guarantee.

Nothing in this module promises a hard spending cap, because an implementation
that counts tokens after the fact cannot deliver one.
"""

import time

from .errors import BudgetExceeded


class Budget:
    def __init__(self, max_calls=None, max_input_chars=None, max_output_tokens_per_call=None,
                 max_total_tokens=None, max_seconds=None, max_retries=2,
                 price_per_million_input=None, price_per_million_output=None,
                 max_estimated_usd=None, currency="USD"):
        self.max_calls = max_calls
        self.max_input_chars = max_input_chars
        self.max_output_tokens_per_call = max_output_tokens_per_call
        self.max_total_tokens = max_total_tokens
        self.max_seconds = max_seconds
        self.max_retries = max_retries
        self.price_per_million_input = price_per_million_input
        self.price_per_million_output = price_per_million_output
        self.max_estimated_usd = max_estimated_usd
        self.currency = currency

        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.retries = 0
        self.started = time.monotonic()
        self.stopped_reason = None

    # -- pre-flight checks ---------------------------------------------------
    def elapsed(self):
        return time.monotonic() - self.started

    def check_before_call(self, prompt_chars, requested_max_tokens=None):
        """Raise BudgetExceeded before spending. Called for every request."""
        if self.stopped_reason:
            raise BudgetExceeded("run already stopped: %s" % self.stopped_reason)
        if self.max_calls is not None and self.calls >= self.max_calls:
            self.stopped_reason = "call limit reached (%d)" % self.max_calls
            raise BudgetExceeded(self.stopped_reason, self.max_calls, self.calls)
        if self.max_input_chars is not None and prompt_chars > self.max_input_chars:
            raise BudgetExceeded(
                "prompt is %d chars, over the %d char limit"
                % (prompt_chars, self.max_input_chars),
                self.max_input_chars, prompt_chars)
        if self.max_output_tokens_per_call is not None and requested_max_tokens \
                and requested_max_tokens > self.max_output_tokens_per_call:
            raise BudgetExceeded(
                "requested %d output tokens, over the %d limit"
                % (requested_max_tokens, self.max_output_tokens_per_call),
                self.max_output_tokens_per_call, requested_max_tokens)
        if self.max_seconds is not None and self.elapsed() > self.max_seconds:
            self.stopped_reason = "time limit reached (%.0fs of %.0fs)" % (
                self.elapsed(), self.max_seconds)
            raise BudgetExceeded(self.stopped_reason, self.max_seconds, self.elapsed())
        if self.max_total_tokens is not None and \
                self.total_tokens() >= self.max_total_tokens:
            self.stopped_reason = "token limit reached (%d)" % self.max_total_tokens
            raise BudgetExceeded(self.stopped_reason, self.max_total_tokens,
                                 self.total_tokens())

    def check_before_retry(self):
        if self.max_retries is not None and self.retries >= self.max_retries:
            raise BudgetExceeded("retry limit reached (%d)" % self.max_retries,
                                 self.max_retries, self.retries)

    # -- accounting ----------------------------------------------------------
    def record_call(self, result=None, input_tokens=None, output_tokens=None):
        self.calls += 1
        if result is not None:
            input_tokens = result.input_tokens
            output_tokens = result.output_tokens
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)

    def record_retry(self):
        self.retries += 1

    def total_tokens(self):
        return self.input_tokens + self.output_tokens

    def estimated_cost(self):
        """An ESTIMATE, or None when no pricing was supplied.

        Never presented as a guarantee. Provider billing is asynchronous and the
        real figure can differ from the provider's own reported token counts.
        """
        if self.price_per_million_input is None and self.price_per_million_output is None:
            return None
        cost = 0.0
        if self.price_per_million_input:
            cost += self.input_tokens / 1_000_000.0 * self.price_per_million_input
        if self.price_per_million_output:
            cost += self.output_tokens / 1_000_000.0 * self.price_per_million_output
        return round(cost, 6)

    def check_estimated_cost(self):
        if self.max_estimated_usd is None:
            return
        estimate = self.estimated_cost()
        if estimate is None:
            return
        if estimate >= self.max_estimated_usd:
            self.stopped_reason = (
                "estimated spend %s %s reached the %s %s estimate limit (estimate only, "
                "not a hard cap)" % (estimate, self.currency,
                                     self.max_estimated_usd, self.currency))
            raise BudgetExceeded(self.stopped_reason, self.max_estimated_usd, estimate)

    def summary(self):
        estimate = self.estimated_cost()
        return {
            "calls": self.calls,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens(),
            "elapsed_seconds": round(self.elapsed(), 2),
            "estimated_cost": estimate,
            "estimated_cost_is_estimate": estimate is not None,
            "currency": self.currency if estimate is not None else None,
            "limits": {
                "max_calls": self.max_calls,
                "max_input_chars": self.max_input_chars,
                "max_output_tokens_per_call": self.max_output_tokens_per_call,
                "max_total_tokens": self.max_total_tokens,
                "max_seconds": self.max_seconds,
                "max_retries": self.max_retries,
                "max_estimated_usd": self.max_estimated_usd,
            },
            "stopped_reason": self.stopped_reason,
        }
