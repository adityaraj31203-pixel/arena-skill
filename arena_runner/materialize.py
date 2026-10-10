"""Turn a file-oriented sub-agent brief into a self-contained API prompt.

The original briefs (rendered by bracket.render_job) tell a Claude Code
sub-agent to *read files from disk*. An API request has no filesystem, so every
input file is inlined into the prompt and every output path is replaced with a
return directive.

This module is also the isolation boundary. `input_paths` declares exactly which
files a job may see, and `materialize` refuses to inline anything else. A
competitor's spawn prompt therefore cannot contain another competitor's draft,
because that path is not in its declared input set.
"""

import os
import re

from .errors import ArenaRunnerError

INLINE_TAG = "arena-inline"
NO_OUTPUT_SENTINEL = "NO OUTPUT"

RETURN_NOTE = (
    "You are running as a stateless API request. You have NO filesystem, NO "
    "tools and NO shell: you cannot read, write or create any file, and you "
    "cannot see this conversation or any other competitor's context. Every "
    "input you need is inlined below between <%s> tags. Where the brief names "
    "an output path, return that artifact as your reply text instead of writing "
    "it. Reply with exactly the artifact the brief asks for." % INLINE_TAG
)


def _join(run_dir, *parts):
    return os.path.join(run_dir, *parts)


def input_paths(state, job):
    """The complete, ordered set of files this job is allowed to see.

    Mirrors bracket.render_job exactly. Anything not listed here is not inlined,
    which is what makes context isolation enforceable rather than aspirational.
    """
    d = state["dir"]
    agents = state["agents"]
    kind = job["kind"]
    paths = []

    if kind == "competitor":
        if state.get("has_baseline"):
            paths.append(("baseline", _join(d, "baseline.md")))
    elif kind == "attacker":
        me, opp = job["agent"], job["opponent"]
        paths.append(("own_solution", agents[me]["solution"]))
        paths.append(("target_solution", agents[opp]["solution"]))
    elif kind == "defender":
        me, opp = job["agent"], job["opponent"]
        from .bracket import attack_out
        paths.append(("own_solution", agents[me]["solution"]))
        paths.append(("attacks", attack_out(d, job["round"], job["match"], opp)))
    elif kind == "judge":
        from .bracket import attack_out, defense_out, revised_out
        n, mid, a, b = job["round"], job["match"], job["a"], job["b"]
        paths += [
            ("rubric", _join(d, "rubric.md")),
            ("first_solution", revised_out(d, n, mid, a)),
            ("first_attacks", attack_out(d, n, mid, b)),
            ("first_defense", defense_out(d, n, mid, a)),
            ("second_solution", revised_out(d, n, mid, b)),
            ("second_attacks", attack_out(d, n, mid, a)),
            ("second_defense", defense_out(d, n, mid, b)),
        ]
    elif kind == "final":
        paths += [("rubric", _join(d, "rubric.md")),
                  ("x_solution", _join(d, "final", "X.md")),
                  ("y_solution", _join(d, "final", "Y.md"))]
    else:
        raise ArenaRunnerError("unknown job kind %r" % kind)
    return paths


def _read_or_missing(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    if not text.strip():
        return None
    return text


def materialize(state, job, brief_text):
    """Return (prompt, inlined) with every declared input inlined.

    `inlined` maps label -> path actually embedded, so a test can assert the
    isolation property directly.
    """
    text = brief_text
    inlined = {}
    missing = []

    for label, path in input_paths(state, job):
        content = _read_or_missing(path)
        if content is None:
            missing.append(label)
            block = ("<%s name=\"%s\" missing=\"true\">\n%s\n</%s>"
                     % (INLINE_TAG, label, NO_OUTPUT_SENTINEL, INLINE_TAG))
        else:
            block = ("<%s name=\"%s\">\n%s\n</%s>"
                     % (INLINE_TAG, label, content.rstrip("\n"), INLINE_TAG))
            inlined[label] = path
        # Replace the literal path token wherever the brief mentions it.
        text = text.replace(path, "\n" + block + "\n")

    for out in job.get("outputs", []):
        text = text.replace(out, "[return this artifact as your reply]")

    prompt = RETURN_NOTE + "\n\n" + text.strip() + "\n"
    return prompt, {"inlined": inlined, "missing_inputs": missing}


def scrub_agent_ids(text, agent_ids, replacement="[competitor]"):
    """Remove competitor identifiers from text destined for a blinded reader."""
    for aid in sorted(agent_ids, key=len, reverse=True):
        text = re.sub(r"\b%s\b" % re.escape(aid), replacement, text)
    return text
