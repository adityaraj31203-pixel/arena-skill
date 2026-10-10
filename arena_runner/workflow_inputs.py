"""Validation for GitHub Actions workflow inputs.

Every workflow_dispatch input is untrusted text typed into a browser form, so it
is parsed and bounded here before anything is executed. The workflow calls this
module in a dedicated step and fails the job on any invalid value, which keeps
the validation unit-testable rather than buried in shell.
"""

import json

from .errors import ArenaRunnerError

ALLOWED_PROVIDERS = ("scripted-mock", "mock", "openai-compatible", "anthropic")
LIVE_PROVIDERS = ("openai-compatible", "anthropic")
CONFIRM_TOKEN = "I-understand-this-may-incur-API-charges"

#: Provider -> environment variable that must hold its key.
PROVIDER_SECRET_ENV = {
    "openai-compatible": "ARENA_OPENAI_API_KEY",
    "anthropic": "ARENA_ANTHROPIC_API_KEY",
}

LIMITS = {
    "agents": (1, 64),            # the expensive 100-agent default is not offered
    "concurrency": (1, 8),
    "max_calls": (1, 5000),
    "max_retries": (0, 5),
    "max_minutes": (1, 360),
    "max_input_chars": (100, 400000),
    "max_output_tokens": (64, 32000),
}


def _int(name, raw, lo, hi, default=None, optional=False):
    if raw is None or str(raw).strip() == "":
        if optional or default is not None:
            return default
        raise ArenaRunnerError("%s is required" % name)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ArenaRunnerError("%s must be an integer, got %r" % (name, raw))
    if not lo <= value <= hi:
        raise ArenaRunnerError("%s must be between %d and %d, got %d"
                               % (name, lo, hi, value))
    return value


def _float(name, raw):
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        raise ArenaRunnerError("%s must be a number, got %r" % (name, raw))
    if value < 0:
        raise ArenaRunnerError("%s must not be negative, got %s" % (name, value))
    return value


def validate(inputs, secrets_present=(), require_live_confirmation=True):
    """Validate a workflow_dispatch input mapping. Returns a normalised dict.

    `secrets_present` is the set of secret names that are actually configured.
    A live provider without its key fails here, before any call is attempted.
    """
    if not isinstance(inputs, dict):
        raise ArenaRunnerError("inputs must be a mapping")

    provider = str(inputs.get("provider") or "").strip()
    if provider not in ALLOWED_PROVIDERS:
        raise ArenaRunnerError("provider must be one of %s, got %r"
                               % (", ".join(ALLOWED_PROVIDERS), provider))

    model = str(inputs.get("model") or "").strip()
    if not model:
        raise ArenaRunnerError("model is required")
    if len(model) > 200:
        raise ArenaRunnerError("model looks too long to be a model identifier")

    agents = _int("agents", inputs.get("agents"), *LIMITS["agents"], default=2)
    live = provider in LIVE_PROVIDERS

    confirm = str(inputs.get("confirm") or "").strip()
    if live and require_live_confirmation and confirm != CONFIRM_TOKEN:
        raise ArenaRunnerError(
            "live runs are billed by the provider and need explicit confirmation. "
            "Set the confirm input to exactly %r" % CONFIRM_TOKEN)

    if live:
        env = PROVIDER_SECRET_ENV[provider]
        if env not in set(secrets_present):
            raise ArenaRunnerError(
                "provider %s needs the repository secret %s. Add it under "
                "Settings > Secrets and variables > Actions; never commit it."
                % (provider, env))

    seed_raw = inputs.get("seed")
    seed = None
    if seed_raw is not None and str(seed_raw).strip() != "":
        seed = _int("seed", seed_raw, 1, 999999)

    task = str(inputs.get("task") or "").strip()
    if not task:
        raise ArenaRunnerError("task is required")
    if len(task) > 20000:
        raise ArenaRunnerError("task is over 20000 characters; put it in a file instead")

    normalised = {
        "provider": provider,
        "model": model,
        "agents": agents,
        "seed": seed,
        "task": task,
        "baseline": str(inputs.get("baseline") or "").strip() or None,
        "live": live,
        "confirmed": confirm == CONFIRM_TOKEN,
        "secret_env": PROVIDER_SECRET_ENV.get(provider),
        "concurrency": _int("concurrency", inputs.get("concurrency"),
                            *LIMITS["concurrency"], default=2),
        "max_calls": _int("max_calls", inputs.get("max_calls"),
                          *LIMITS["max_calls"], default=60),
        "max_retries": _int("max_retries", inputs.get("max_retries"),
                            *LIMITS["max_retries"], default=2),
        "max_minutes": _int("max_minutes", inputs.get("max_minutes"),
                            *LIMITS["max_minutes"], default=30),
        "max_input_chars": _int("max_input_chars", inputs.get("max_input_chars"),
                                *LIMITS["max_input_chars"], optional=True),
        "max_output_tokens": _int("max_output_tokens", inputs.get("max_output_tokens"),
                                  *LIMITS["max_output_tokens"], optional=True),
        "price_in": _float("price_in", inputs.get("price_in")),
        "price_out": _float("price_out", inputs.get("price_out")),
        "max_estimated_usd": _float("max_estimated_usd",
                                    inputs.get("max_estimated_usd")),
    }

    expected = expected_calls(agents, baseline=bool(normalised["baseline"]))
    if normalised["max_calls"] < expected:
        raise ArenaRunnerError(
            "max_calls=%d is below the %d calls this tournament needs; raise it or "
            "lower agents" % (normalised["max_calls"], expected))
    normalised["expected_calls"] = expected
    return normalised


def expected_calls(agents, baseline=False):
    """Model calls for a full tournament: 1 spawn per agent, 5 per match per round."""
    total = agents
    alive = agents
    while alive > 1:
        matches = alive // 2
        total += 5 * matches
        alive = matches + (alive % 2)
    return total + (1 if baseline else 0)


def summary_line(normalised):
    """The one line an operator should see before the run starts."""
    if normalised["live"]:
        cost = ""
        if normalised["price_in"] or normalised["price_out"]:
            cost = " (a cost estimate will be reported; it is an estimate, not a cap)"
        return ("LIVE RUN: %d competitors, about %d paid model calls on %s/%s%s"
                % (normalised["agents"], normalised["expected_calls"],
                   normalised["provider"], normalised["model"], cost))
    return ("OFFLINE RUN: %d competitors, about %d calls against the %s test double. "
            "No spend." % (normalised["agents"], normalised["expected_calls"],
                           normalised["provider"]))


def as_json(normalised):
    """Serialisable form for passing between workflow steps. Excludes the task body."""
    out = dict(normalised)
    out["task"] = "<%d chars>" % len(normalised["task"])
    out["baseline"] = ("<%d chars>" % len(normalised["baseline"])
                       if normalised["baseline"] else None)
    return json.dumps(out, indent=2, sort_keys=True)
