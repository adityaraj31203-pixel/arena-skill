"""Blinded judging.

The judge must not learn which competitor produced which solution, which
provider or model produced it, or which strategy card it was playing. The
original judge template already omits the strategy cards; this module removes
the remaining identifiers by substituting neutral labels.

Labels are deliberately not single letters: bracket.decide lowercases the score
keys and searches for the winner with a word-boundary match, so a bare "a" would
collide with ordinary English. "solA"/"solB" survive that match intact.

The judge's raw reply is kept verbatim as an audit artifact. The verdict written
for bracket.collect is de-anonymised back to real competitor IDs, so the engine
records a genuine judge verdict and the controller never picks a winner.
"""

import copy
import json
import re

LABELS = ("solA", "solB")


def anonymize(text, first, second):
    """Replace the two competitor IDs with neutral labels."""
    mapping = {first: LABELS[0], second: LABELS[1]}
    for aid, label in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        text = re.sub(r"\b%s\b" % re.escape(aid), label, text)
    return text, mapping


def label_map(first, second):
    return {LABELS[0]: first, LABELS[1]: second}


def de_anonymize_verdict(verdict, first, second):
    """Rewrite a label-keyed verdict into real competitor IDs.

    Returns a deep copy. Raises ValueError when the judge produced something that
    cannot be mapped, so a malformed verdict is never silently accepted.
    """
    back = label_map(first, second)
    out = copy.deepcopy(verdict)

    scores = out.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("verdict has no 'scores' object")
    remapped = {}
    for key, value in scores.items():
        real = back.get(str(key).strip()) or back.get(str(key).strip().lower())
        if real is None:
            # Tolerate a judge that echoed a real ID despite the blinding.
            if str(key).strip().lower() in (first.lower(), second.lower()):
                real = str(key).strip().lower()
            else:
                raise ValueError("verdict scores contain unknown competitor %r" % key)
        remapped[real] = value
    if set(remapped) != {first, second}:
        raise ValueError("verdict must score exactly %s and %s, got %s"
                         % (first, second, sorted(remapped)))
    out["scores"] = remapped

    winner_raw = str(out.get("winner") or "").strip()
    real_winner = None
    for label, real in back.items():
        if re.search(r"\b%s\b" % re.escape(label), winner_raw, re.I):
            if real_winner and real_winner != real:
                raise ValueError("verdict winner names both competitors: %r" % winner_raw)
            real_winner = real
    if real_winner is None:
        lowered = winner_raw.lower()
        if lowered in (first.lower(), second.lower()):
            real_winner = lowered
        else:
            raise ValueError("verdict winner %r is neither %s nor %s"
                             % (winner_raw, first, second))
    out["winner"] = real_winner

    standing = out.get("standing")
    if isinstance(standing, dict):
        fixed = {}
        for key, value in standing.items():
            real = back.get(str(key).strip()) or back.get(str(key).strip().lower())
            fixed[real or str(key)] = value
        out["standing"] = fixed

    out["_blinded"] = True
    return out


def scrub(text, agent_ids, provider_names=(), model_names=()):
    """Strip competitor IDs, provider names and model names from judge inputs."""
    for aid in sorted(agent_ids, key=len, reverse=True):
        text = re.sub(r"\b%s\b" % re.escape(aid), "[competitor]", text)
    for name in list(provider_names) + list(model_names):
        if name:
            text = re.sub(re.escape(str(name)), "[model]", text)
    return text


def parse_verdict(text):
    """Extract the JSON object from a judge reply. None if there is not one."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except ValueError:
            return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except ValueError:
            return None
    return None
