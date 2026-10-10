"""Mobile-friendly reporting.

The operator may be on a phone and may have closed the app while the run was in
progress. Reporting therefore reads only persisted state: the engine's arena.json
plus the run log. A run is never reported as complete unless the persisted state
actually says so.
"""

import json
import os

from .bracket import alive_ids, rounds_played, winner_report

STATUS_GLYPH = {"done": "[ok]", "failed": "[FAIL]", "running": "[..]",
                "pending": "[ ]"}


def _load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def run_status(run_dir):
    """A compact status dict derived from persisted state only."""
    state = _load_json(os.path.join(run_dir, "arena.json"))
    if state is None:
        return {"complete": False, "error": "no arena.json in %s" % run_dir}
    log = _load_json(os.path.join(run_dir, "runlog.json")) or {}

    champion = state.get("champion")
    final = state.get("final")
    final_done = bool(final and final.get("result"))
    needs_final = bool(final)
    complete = champion is not None and (final_done or not needs_final)

    counts = {}
    for entry in (log.get("jobs") or {}).values():
        counts[entry.get("status", "unknown")] = counts.get(entry.get("status", "unknown"), 0) + 1

    return {
        "run_dir": run_dir,
        "seed": state.get("seed"),
        "agents": state.get("agents_n"),
        "rounds_played": rounds_played(state),
        "alive": alive_ids(state),
        "champion": champion,
        "complete": complete,
        "awaiting_final_check": bool(needs_final and not final_done),
        "final_result": (final or {}).get("result"),
        "jobs": counts,
        "budget": log.get("budget"),
        "independence": log.get("independence"),
        "provider": log.get("provider"),
    }


def format_status(status):
    """Plain text a phone browser can read without a table renderer."""
    if status.get("error"):
        return "RUN NOT FOUND\n  %s\n" % status["error"]
    lines = [
        "ARENA RUN",
        "  agents      %s" % status["agents"],
        "  seed        %s" % status["seed"],
        "  rounds done %s" % status["rounds_played"],
        "  alive       %s" % (", ".join(status["alive"]) or "none"),
        "  champion    %s" % (status["champion"] or "not yet"),
    ]
    if status["jobs"]:
        lines.append("  jobs        " + ", ".join(
            "%s=%d" % kv for kv in sorted(status["jobs"].items())))
    budget = status.get("budget") or {}
    if budget:
        lines.append("  model calls %s (limit %s)"
                     % (budget.get("calls"), (budget.get("limits") or {}).get("max_calls")))
        if budget.get("estimated_cost") is not None:
            lines.append("  est. cost   %s %s (ESTIMATE, not a cap)"
                         % (budget["estimated_cost"], budget.get("currency")))
    independence = status.get("independence") or {}
    if independence:
        lines.append("  independence Level %s: %s"
                     % (independence.get("level"), independence.get("label")))
    provider = status.get("provider") or {}
    if provider:
        lines.append("  provider    %s / %s (live=%s)"
                     % (provider.get("provider"), provider.get("model"),
                        provider.get("live")))
    lines.append("  COMPLETE    %s" % ("yes" if status["complete"] else "NO"))
    if status.get("awaiting_final_check"):
        lines.append("              the final check against the rejected answer "
                     "has not run yet")
    return "\n".join(lines) + "\n"


def format_failures(run_dir):
    log = _load_json(os.path.join(run_dir, "runlog.json")) or {}
    failed = [e for e in (log.get("jobs") or {}).values() if e.get("status") == "failed"]
    if not failed:
        return "no failed jobs recorded\n"
    lines = ["FAILED JOBS (%d)" % len(failed)]
    for entry in sorted(failed, key=lambda e: e["job_id"]):
        lines.append("  %s  %s  %s" % (entry["job_id"],
                                       entry.get("error_kind"),
                                       (entry.get("error") or "")[:120]))
        lines.append("      attempts: %s" % entry.get("attempts"))
    return "\n".join(lines) + "\n"


def format_winner(run_dir):
    """Render the engine's winner_report dict as plain text.

    The engine returns a structured record; this only formats it. Nothing here
    chooses or overrides a winner.
    """
    state = _load_json(os.path.join(run_dir, "arena.json"))
    if state is None:
        return "no arena.json in %s\n" % run_dir
    report = winner_report(state)
    if report is None:
        return "no champion yet: the run is not complete\n"

    out = ["CHAMPION  %s" % report["champion"],
           "card      %s" % report["card_line"],
           "solution  %s" % report["solution"],
           "rounds    %s played, %s agents in, %s match(es) won"
           % (report["rounds"], report["agents"], report["matches_won"])]
    for step in report["path"]:
        if step.get("bye"):
            out.append("  round %s  bye" % step["round"])
            continue
        out.append("  round %s  beat %s, %s to %s"
                   % (step["round"], step["beat"], step["score"],
                      step["opponent_score"]))
        for line in step.get("survived") or []:
            out.append("           survived: %s" % line)
    final = report.get("final")
    if final:
        out.append("vs the answer you rejected: %s is better, %s to %s"
                   % (final.get("better"), final.get("champion_total"),
                      final.get("baseline_total")))
        if final.get("reason"):
            out.append("  %s" % final["reason"])

    log = _load_json(os.path.join(run_dir, "runlog.json")) or {}
    independence = log.get("independence") or {}
    if independence:
        out.append("")
        out.append("INDEPENDENCE ACHIEVED")
        out.append("  Level %s - %s" % (independence.get("level"),
                                         independence.get("label")))
        for line in independence.get("evidence", []):
            out.append("  + %s" % line)
        for line in independence.get("caveats", []):
            out.append("  - %s" % line)
    out.append("")
    out.append("NOTE: model-based judging is an evaluation against a written "
               "rubric, not proof of objective correctness.")
    return "\n".join(out) + "\n"


def format_plan(rows, totals, baseline=False):
    lines = ["EXPECTED MODEL CALLS"]
    lines.append("  %-8s %6s %8s %6s %8s" %
                 ("round", "alive", "matches", "bye", "calls"))
    for row in rows:
        lines.append("  %-8s %6d %8d %6s %8d" % (
            row["round"], row["alive"], row["matches"],
            "yes" if row["bye"] else "-", row["calls"]))
    extra = 1 if baseline else 0
    lines.append("  %-8s %6s %8s %6s %8d" %
                 ("total", "", "", "", totals["calls"] + extra))
    if baseline:
        lines.append("  plus 1 call for the final check against the rejected answer")
    lines.append("\n  Every call is one paid model request unless the provider is "
                 "an offline test double.")
    return "\n".join(lines) + "\n"


def markdown_report(run_dir):
    """A Markdown report suitable for a workflow artifact."""
    status = run_status(run_dir)
    parts = ["# Arena run report\n", "```", format_status(status).rstrip(), "```\n"]
    parts.append("## Failures\n\n```")
    parts.append(format_failures(run_dir).rstrip())
    parts.append("```\n")
    if status.get("complete"):
        parts.append("## Winner\n\n```")
        parts.append(format_winner(run_dir).rstrip())
        parts.append("```\n")
        champion = status["champion"]
        state = _load_json(os.path.join(run_dir, "arena.json"))
        if state and champion:
            path = state["agents"][champion]["solution"]
            if os.path.exists(path):
                parts.append("## Winning solution\n")
                with open(path, encoding="utf-8") as fh:
                    parts.append(fh.read())
    independence = status.get("independence") or {}
    parts.append("\n## How to read this\n")
    parts.append("- Judging is performed by a model against "
                 "`skills/arena/rubric.md`. It is an evaluation, not proof.")
    if independence.get("level") is not None:
        parts.append("- Independence level %s. See the caveats above before "
                     "describing this run as multi-agent." % independence.get("level"))
    return "\n".join(parts) + "\n"
