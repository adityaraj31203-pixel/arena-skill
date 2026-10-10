"""Command line entry point.

    python3 -m arena_runner plan    --agents 4 [--baseline-file f]
    python3 -m arena_runner run     --agents 4 --task-file t.md [options]
    python3 -m arena_runner report  --run-dir .arena/run-...
    python3 -m arena_runner doctor

Safety defaults: the provider is the offline test double, so `run` costs nothing
unless an operator explicitly names a live provider AND confirms with
--confirm-live (or ARENA_CONFIRM_LIVE=yes). Tournaments are never started
implicitly.
"""

import argparse
import os
import sys

from .bracket import bracket_sizes, combo_count, load_strategies, plan_rows, plan_totals
from .budget import Budget
from .errors import ArenaRunnerError
from .http_provider import build_registry
from .report import (format_failures, format_plan, format_status, format_winner,
                     markdown_report, run_status)
from .runner import TournamentRunner, init_run

CONFIRM_TOKEN = "I-understand-this-may-incur-API-charges"


def _provider(args):
    """Build the configured provider. Offline doubles need no credentials."""
    registry = build_registry()
    model = args.model or ("mock-model-1"
                           if args.provider in ("mock", "scripted-mock") else "")
    return registry.create(args.provider, model=model)


def _require_live_confirmation(args):
    if args.confirm_live or os.environ.get("ARENA_CONFIRM_LIVE") == "yes":
        return
    raise ArenaRunnerError(
        "refusing to start a live run without explicit confirmation.\n"
        "  Live calls are billed by the provider and this run is expected to make "
        "%d model request(s).\n"
        "  Re-run with --confirm-live %s\n"
        "  or set ARENA_CONFIRM_LIVE=yes." % (args.max_calls or 0, CONFIRM_TOKEN))


def _budget(args):
    return Budget(
        max_calls=args.max_calls,
        max_input_chars=args.max_input_chars,
        max_output_tokens_per_call=args.max_output_tokens,
        max_seconds=(args.max_minutes * 60) if args.max_minutes else None,
        max_retries=args.max_retries,
        price_per_million_input=args.price_in,
        price_per_million_output=args.price_out,
        max_estimated_usd=args.max_estimated_usd,
    )


def cmd_plan(args):
    data = load_strategies()
    if args.agents < 1 or args.agents > combo_count(data):
        raise ArenaRunnerError("--agents must be between 1 and %d" % combo_count(data))
    rows = plan_rows(args.agents, wave=args.wave)
    totals = plan_totals(args.agents, wave=args.wave)
    print(format_plan(rows, totals, baseline=bool(args.baseline_file)))
    print("  %d rounds: %s" % (totals["rounds"],
                               " -> ".join(str(s) for s in bracket_sizes(args.agents))))
    return 0


def cmd_run(args):
    provider = _provider(args)
    if getattr(provider, "live", False):
        _require_live_confirmation(args)
        if not args.api_key_env or not os.environ.get(args.api_key_env):
            raise ArenaRunnerError(
                "provider %s needs an API key in the environment variable %s.\n"
                "  In GitHub Actions add it as a repository secret; never commit it."
                % (provider.name, args.api_key_env or "<unset>"))

    if args.resume:
        from .bracket import load_state
        state = load_state(type("A", (), {"dir": os.path.abspath(args.resume)})())
    else:
        if not args.task_file:
            raise ArenaRunnerError("--task-file is required unless --resume is used")
        if args.agents == 100:
            print("WARNING: 100 competitors is the expensive default. "
                  "Consider --agents 2 for a smoke test.", file=sys.stderr)
        state = init_run(args.run_dir or ".arena/run", args.agents, seed=args.seed,
                         task_file=args.task_file, baseline_file=args.baseline_file)
        print("run directory: %s" % state["dir"])

    runner = TournamentRunner(state, provider, budget=_budget(args),
                              concurrency=args.concurrency,
                              max_attempts=args.max_retries + 1,
                              on_event=_printer)
    try:
        result = runner.run_to_completion()
    except ArenaRunnerError as exc:
        print("\nRUN STOPPED: %s" % exc, file=sys.stderr)
        print(format_status(run_status(state["dir"])))
        print(format_failures(state["dir"]))
        return 2

    print()
    print(format_status(run_status(state["dir"])))
    if result.get("champion"):
        print(format_winner(state["dir"]))
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as fh:
            fh.write(markdown_report(state["dir"]))
        print("report written to %s" % args.report_out)
    return 0


def _printer(event):
    kind = event.get("type")
    if kind == "phase_done":
        print("  phase %-8s dispatched=%d done=%d failed=%d"
              % (event["phase"], event["dispatched"], event["done"], event["failed"]))
    elif kind == "round_advanced":
        print("  round advanced: %d alive" % event["alive"])
    elif kind == "job_failed":
        print("  job %s FAILED: %s" % (event["job_id"], event.get("error")))


def cmd_report(args):
    run_dir = os.path.abspath(args.run_dir)
    print(format_status(run_status(run_dir)))
    print(format_failures(run_dir))
    status = run_status(run_dir)
    if status.get("complete"):
        print(format_winner(run_dir))
    else:
        print("not complete: nothing is reported as finished until the persisted "
              "tournament state confirms it.\n")
    if args.markdown_out:
        with open(args.markdown_out, "w", encoding="utf-8") as fh:
            fh.write(markdown_report(run_dir))
        print("markdown report written to %s" % args.markdown_out)
    return 0


ENV_PREFIX = "ARENA_IN_"


def _inputs_from_env(environ=None):
    """Read workflow inputs from environment variables.

    Values arrive through the environment rather than the command line, so a
    hostile input string cannot break out into a shell command.
    """
    environ = os.environ if environ is None else environ
    return {key[len(ENV_PREFIX):].lower(): value
            for key, value in environ.items() if key.startswith(ENV_PREFIX)}


def cmd_workflow(args):
    """Entry point used by the GitHub Actions workflow.

    Inputs come from ARENA_IN_* environment variables. Secrets are read from
    their own variables and are never printed.
    """
    from .workflow_inputs import as_json, summary_line, validate

    inputs = _inputs_from_env()
    secrets_present = {name for name in
                       ("ARENA_OPENAI_API_KEY", "ARENA_ANTHROPIC_API_KEY")
                       if os.environ.get(name)}
    config = validate(inputs, secrets_present=secrets_present)
    print(summary_line(config))
    print(as_json(config))

    if config["live"]:
        env = config["secret_env"]
        if not os.environ.get(env):
            raise ArenaRunnerError(
                "%s is set as a secret but is empty in this job" % env)

    run_dir = os.environ.get("ARENA_RUN_DIR", ".arena/run")
    workspace = os.environ.get("ARENA_WORKSPACE", ".arena-workspace")
    os.makedirs(workspace, exist_ok=True)
    task_file = os.path.join(workspace, "task.md")
    with open(task_file, "w", encoding="utf-8") as fh:
        fh.write(config["task"] + "\n")
    baseline_file = None
    if config["baseline"]:
        baseline_file = os.path.join(workspace, "baseline.md")
        with open(baseline_file, "w", encoding="utf-8") as fh:
            fh.write(config["baseline"] + "\n")

    provider = build_registry().create(
        config["provider"], model=config["model"],
        **({"api_key_env": config["secret_env"]} if config["secret_env"] else {}))

    state = init_run(run_dir, config["agents"], seed=config["seed"],
                     task_file=task_file, baseline_file=baseline_file)
    print("run directory: %s" % state["dir"])
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write("run_dir=%s\n" % state["dir"])

    runner = TournamentRunner(
        state, provider,
        budget=Budget(max_calls=config["max_calls"],
                      max_input_chars=config["max_input_chars"],
                      max_output_tokens_per_call=config["max_output_tokens"],
                      max_seconds=config["max_minutes"] * 60,
                      max_retries=config["max_retries"],
                      price_per_million_input=config["price_in"],
                      price_per_million_output=config["price_out"],
                      max_estimated_usd=config["max_estimated_usd"]),
        concurrency=config["concurrency"],
        max_attempts=config["max_retries"] + 1,
        on_event=_printer)
    try:
        runner.run_to_completion()
    except ArenaRunnerError as exc:
        print("\nRUN STOPPED: %s" % exc, file=sys.stderr)
        print(format_status(run_status(state["dir"])))
        print(format_failures(state["dir"]))
        return 2

    print()
    print(format_status(run_status(state["dir"])))
    status = run_status(state["dir"])
    if status.get("complete"):
        print(format_winner(state["dir"]))
    else:
        print("not reported as complete: the persisted state does not say it is.")
    report_path = os.path.join(workspace, "report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(markdown_report(state["dir"]))
    print("report written to %s" % report_path)
    return 0


def cmd_doctor(args):
    print("ARENA CONFIGURATION CHECK")
    print("  python        %s" % sys.version.split()[0])
    from .bracket import ENGINE_PATH
    print("  engine        %s (%s)" % (ENGINE_PATH,
                                       "present" if os.path.exists(ENGINE_PATH) else "MISSING"))
    registry = build_registry()
    print("  providers     %s" % ", ".join(registry.names()))
    for name, env in (("openai-compatible", "ARENA_OPENAI_API_KEY"),
                      ("anthropic", "ARENA_ANTHROPIC_API_KEY")):
        configured = bool(os.environ.get(env))
        print("  %-18s %s (%s)" % (name,
                                   "key present" if configured else "no key configured",
                                   env))
    print("  default       offline test double; no spend without --confirm-live")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="arena_runner",
                                     description="Independent multi-agent Arena runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="show the expected model calls before spending")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--wave", type=int, default=10)
    p.add_argument("--baseline-file")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="run a tournament")
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--seed", type=int)
    p.add_argument("--task-file")
    p.add_argument("--baseline-file")
    p.add_argument("--run-dir", default=".arena/run")
    p.add_argument("--resume", help="path to an existing run directory")
    p.add_argument("--provider", default="scripted-mock",
                   choices=["mock", "scripted-mock", "openai-compatible", "anthropic"])
    p.add_argument("--model")
    p.add_argument("--api-key-env", default="ARENA_PROVIDER_API_KEY")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-calls", type=int, default=200)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-minutes", type=int)
    p.add_argument("--max-input-chars", type=int)
    p.add_argument("--max-output-tokens", type=int)
    p.add_argument("--price-in", type=float, help="USD per million input tokens")
    p.add_argument("--price-out", type=float, help="USD per million output tokens")
    p.add_argument("--max-estimated-usd", type=float)
    p.add_argument("--confirm-live", action="store_true",
                   help="required before any run that can incur API charges")
    p.add_argument("--report-out")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="report a run from its persisted state")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--markdown-out")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("workflow",
                       help="run using ARENA_IN_* environment variables (used by CI)")
    p.set_defaults(func=cmd_workflow)

    p = sub.add_parser("doctor", help="check the local configuration")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ArenaRunnerError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
