"""Secret redaction.

Secrets must never reach logs, artifacts, prompts, exception text or reports.
Redaction is defence in depth: credentials are read from the environment only,
never passed through prompt text, and anything that does look like a secret is
scrubbed on the way out.
"""

import os
import re

# Well-known credential shapes. These are patterns, not real secrets.
PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),                  # OpenAI-style
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),              # Anthropic-style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),              # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),            # GitHub fine-grained
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),           # Slack-style
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),                 # Google-style
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-]{16,}\b"),     # Authorization headers
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd)"
               r"(\s*[:=]\s*)['\"]?[A-Za-z0-9._\-]{8,}['\"]?"),
]

REDACTED = "[REDACTED]"

# Environment variables whose *values* are treated as secrets wherever they
# appear.
#
# The match is on the shape of the variable NAME, not on a vendor prefix. The
# earlier version of this list treated every "ARENA_*" variable as a secret,
# which was wrong in a way that mattered: the GitHub Actions workflow passes
# every input as ARENA_IN_*, so the provider name and model name were scrubbed
# out of the run log. That destroyed the very provenance the independence report
# is supposed to present, while adding no security at all, because a variable
# called ARENA_IN_MODEL does not hold a credential.
#
# Detection is not weakened to compensate. A name is a secret when it carries a
# credential-shaped word anywhere in it, or ends in a credential suffix, and the
# regex patterns below still catch well-formed keys regardless of the name they
# arrived in.
SECRET_NAME_TOKENS = (
    "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL",
    "APIKEY", "API_KEY", "PRIVATE_KEY", "ACCESS_KEY", "AUTH",
)
SECRET_ENV_SUFFIXES = (
    "_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIAL", "_CREDENTIALS",
)
# Names that are credentials but do not follow either convention above.
SECRET_ENV_EXACT = frozenset({"GH_TOKEN", "GITHUB_TOKEN"})

# Configuration namespaces that carry no credentials. Listed so the intent is
# explicit and testable; the rules above already exclude them.
NON_SECRET_ENV_PREFIXES = ("ARENA_IN_", "ARENA_RUN_DIR", "ARENA_WORKSPACE")


def is_secret_name(name):
    """True when an environment variable name says it holds a credential."""
    upper = str(name).upper()
    if upper in SECRET_ENV_EXACT:
        return True
    if any(upper.endswith(suffix) for suffix in SECRET_ENV_SUFFIXES):
        return True
    return any(token in upper for token in SECRET_NAME_TOKENS)


def secret_values_from_env(environ=None, extra=()):
    """Concrete secret strings currently held in the environment.

    Values are returned so callers can scrub them from free text. They are never
    logged by this function.
    """
    environ = os.environ if environ is None else environ
    found = set()
    for name, value in environ.items():
        if not value or len(value) < 8:
            continue
        if is_secret_name(name):
            found.add(value)
    for value in extra:
        if value and len(value) >= 8:
            found.add(value)
    return found


def redact(text, secrets=()):  # noqa: C901
    """Scrub known credential shapes and any explicitly supplied secret values."""
    if text is None:
        return None
    text = str(text)
    for pattern in PATTERNS:
        text = pattern.sub(REDACTED, text)
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    return text


def redact_mapping(mapping, secrets=()):
    """Redact the string values of a dict, recursing into nested dicts and lists."""
    if isinstance(mapping, dict):
        return {k: redact_mapping(v, secrets) for k, v in mapping.items()}
    if isinstance(mapping, (list, tuple)):
        return [redact_mapping(v, secrets) for v in mapping]
    if isinstance(mapping, str):
        return redact(mapping, secrets)
    return mapping


def assert_clean(text, secrets=(), context="output"):
    """Raise if a secret is present. Used by tests and before writing artifacts."""
    from .errors import SecretLeakAttempt

    scrubbed = redact(text, secrets)
    if scrubbed != str(text):
        raise SecretLeakAttempt("a secret-shaped string reached %s" % context)
    return True
