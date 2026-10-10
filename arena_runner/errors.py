"""Error types for the runner. Each carries a machine-readable kind so failures
can be recorded and reported without leaking secrets."""


class ArenaRunnerError(Exception):
    """Base class."""

    kind = "error"


class ProviderError(ArenaRunnerError):
    """The provider could not produce a completion."""

    kind = "provider_error"

    def __init__(self, message, status=None, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class RateLimited(ProviderError):
    """The provider asked us to slow down. retry_after is seconds, if given."""

    kind = "rate_limited"

    def __init__(self, message="rate limited", retry_after=None, status=429):
        super().__init__(message, status=status, retryable=True)
        self.retry_after = retry_after


class ProviderTimeout(ProviderError):
    kind = "timeout"

    def __init__(self, message="request timed out", status=None):
        super().__init__(message, status=status, retryable=True)


class MalformedResponse(ArenaRunnerError):
    """The provider answered, but the answer is not what the phase requires."""

    kind = "malformed_response"

    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail


class JobFailed(ArenaRunnerError):
    """A job exhausted its attempts. Never turned into a fake success."""

    kind = "job_failed"

    def __init__(self, job_id, attempts, cause):
        super().__init__("%s failed after %d attempt(s): %s" % (job_id, attempts, cause))
        self.job_id = job_id
        self.attempts = attempts
        self.cause = cause


class BudgetExceeded(ArenaRunnerError):
    """A configured limit was reached. The run stops safely and says so."""

    kind = "budget_exceeded"

    def __init__(self, message, limit=None, observed=None):
        super().__init__(message)
        self.limit = limit
        self.observed = observed


class SecretLeakAttempt(ArenaRunnerError):
    """Raised in tests and by redaction checks when a secret reaches output."""

    kind = "secret_leak"
