"""Output validation.

Every phase has a required shape. A response that does not match is a failure,
recorded as a failure. It is never coerced into a plausible-looking artifact,
because a fabricated success is worse than a visible gap: the engine already
defines what a missing attack, solution or verdict means, and the operator is
told which jobs failed.

Multi-output jobs (the defender writes a defense file and a revised solution)
are separated by ARTIFACT_SEP. This is the one intentional deviation from the
original brief, which assumed the sub-agent could write two files itself. It is
documented in docs/MIGRATION.md.
"""

import re

from .anonymize import de_anonymize_verdict, parse_verdict
from .errors import MalformedResponse

ARTIFACT_SEP = "===ARTIFACT==="
NO_OUTPUT = "NO OUTPUT"
MIN_SOLUTION_CHARS = 40
SEVERITIES = ("FATAL", "MAJOR", "MINOR")
SCORE_KEYS = ("correctness", "completeness", "specificity", "robustness", "clarity")


def multi_output_directive(outputs):
    """Instruction appended when a job must return more than one artifact."""
    if len(outputs) < 2:
        return ""
    names = ", ".join("%d) %s" % (i + 1, o) for i, o in enumerate(outputs))
    return ("\n\nReturn ALL of these artifacts in this exact order, separated by a "
            "line containing only %s:\n%s\n" % (ARTIFACT_SEP, names))


def split_artifacts(text, expected):
    parts = [p.strip("\n") for p in str(text).split(ARTIFACT_SEP)]
    if len(parts) != expected:
        raise MalformedResponse(
            "expected %d artifact(s) separated by %s, got %d"
            % (expected, ARTIFACT_SEP, len(parts)),
            detail={"parts": len(parts)})
    return parts


def _require(condition, message, **detail):
    if not condition:
        raise MalformedResponse(message, detail=detail or None)


def validate_scores(scores, who):
    """Scores must be five numbers in 0..10 plus an optional boolean fatal."""
    _require(isinstance(scores, dict), "%s: scores is not an object" % who, who=who)
    for key in SCORE_KEYS:
        _require(key in scores, "%s: missing score %r" % (who, key), who=who, missing=key)
        value = scores[key]
        _require(not isinstance(value, bool),
                 "%s: %s must be a number, not a boolean" % (who, key), who=who)
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise MalformedResponse("%s: %s is not a number (%r)" % (who, key, value))
        _require(0.0 <= number <= 10.0,
                 "%s: %s is %s, outside 0..10" % (who, key, number), who=who, value=number)
    if "fatal" in scores:
        _require(isinstance(scores["fatal"], bool) or
                 str(scores["fatal"]).strip().lower() in ("true", "false", "yes", "no", "0", "1"),
                 "%s: fatal must be boolean" % who, who=who)
    return True


def validate(job, text, first=None, second=None):
    """Return {output_path: content} for a validated response.

    Raises MalformedResponse for anything that does not match the phase contract.
    """
    outputs = job["outputs"]
    kind = job["kind"]
    text = (text or "").strip()

    if kind == "competitor":
        _require(len(text) >= MIN_SOLUTION_CHARS,
                 "competitor solution is %d chars, under %d"
                 % (len(text), MIN_SOLUTION_CHARS), chars=len(text))
        _require(len(outputs) == 1, "competitor job must have one output")
        return {outputs[0]: text + "\n"}

    if kind == "attacker":
        _require(len(outputs) == 1, "attacker job must have one output")
        if text.upper() == NO_OUTPUT:
            return {outputs[0]: NO_OUTPUT + "\n"}
        _require("ATTACK" in text.upper(),
                 "attack response contains no ATTACK line", snippet=text[:200])
        _require(any(s in text.upper() for s in SEVERITIES),
                 "attack response has no FATAL/MAJOR/MINOR label", snippet=text[:200])
        count = len(re.findall(r"^\s*ATTACK\s+\d+", text, re.M | re.I))
        _require(count >= 1, "attack response has no numbered attacks", snippet=text[:200])
        return {outputs[0]: text + "\n"}

    if kind == "defender":
        _require(len(outputs) == 2, "defender job must have two outputs")
        defense, solution = split_artifacts(text, 2)
        _require(defense.upper() == NO_OUTPUT
                 or "CONCEDE" in defense.upper() or "REBUT" in defense.upper(),
                 "defense names neither CONCEDE nor REBUT", snippet=defense[:200])
        _require("NO ATTACKS RECEIVED" in defense.upper()
                 or re.search(r"ATTACK\s*\d+\s*:", defense, re.I) is not None,
                 "defense is not point-by-point", snippet=defense[:200])
        _require(len(solution) >= MIN_SOLUTION_CHARS,
                 "revised solution is %d chars, under %d" % (len(solution), MIN_SOLUTION_CHARS),
                 chars=len(solution))
        return {outputs[0]: defense + "\n", outputs[1]: solution + "\n"}

    if kind == "judge":
        _require(len(outputs) == 1, "judge job must have one output")
        _require(first is not None and second is not None,
                 "judge validation needs both competitor IDs")
        verdict = parse_verdict(text)
        _require(verdict is not None, "judge response is not JSON", snippet=text[:200])
        _require(isinstance(verdict, dict), "judge response is not a JSON object")
        scores = verdict.get("scores")
        _require(isinstance(scores, dict) and len(scores) == 2,
                 "judge verdict must score exactly two competitors")
        for who in sorted(scores):
            validate_scores(scores[who], "verdict[%s]" % who)
        _require(bool(verdict.get("reason")), "judge verdict has no reason")
        # Map the blinded labels back to real IDs; raises if unmappable.
        mapped = de_anonymize_verdict(verdict, first, second)
        return {outputs[0]: _dump(mapped)}

    if kind == "final":
        _require(len(outputs) == 1, "final job must have one output")
        verdict = parse_verdict(text)
        _require(verdict is not None, "final response is not JSON", snippet=text[:200])
        scores = verdict.get("scores")
        _require(isinstance(scores, dict) and set(scores) == {"X", "Y"},
                 "final verdict must score exactly X and Y")
        for who in ("X", "Y"):
            validate_scores(scores[who], "final[%s]" % who)
        winner = str(verdict.get("winner") or "").strip().upper()
        _require(winner in ("X", "Y"), "final winner must be X or Y, got %r"
                 % verdict.get("winner"))
        verdict["winner"] = winner
        return {outputs[0]: _dump(verdict)}

    raise MalformedResponse("unknown job kind %r" % kind)


def _dump(obj):
    import json
    return json.dumps(obj, indent=2) + "\n"
