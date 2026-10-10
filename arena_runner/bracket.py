"""Loader for the ORIGINAL tournament engine.

skills/arena/bracket.py is imported as-is and never modified. This module
resolves it by path relative to the repository root so the runner package stays
independent of how it is installed or where the checkout lives.

Everything the runner needs from the engine is re-exported here, which keeps the
rest of the package free of importlib plumbing and makes the reuse explicit and
auditable.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
ENGINE_PATH = os.path.join(REPO_ROOT, "skills", "arena", "bracket.py")
_MODULE_NAME = "arena_original_bracket"


def load_engine(path=ENGINE_PATH):
    """Import the untouched engine from disk and cache it in sys.modules."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    if not os.path.exists(path):
        raise ImportError("cannot find the tournament engine at %s" % path)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


engine = load_engine()

# --- state management -------------------------------------------------------
new_state = engine.new_state
load_state = engine.load_state
save_state = engine.save_state
current_round = engine.current_round
open_round = engine.open_round
pair_round = engine.pair_round
next_action = engine.next_action

# --- competitor generation --------------------------------------------------
load_strategies = engine.load_strategies
deal = engine.deal
combo_count = engine.combo_count
agent_ids = engine.agent_ids
bracket_sizes = engine.bracket_sizes

# --- job construction and briefs -------------------------------------------
phase_jobs = engine.phase_jobs
missing_jobs = engine.missing_jobs
render_job = engine.render_job
write_prompts = engine.write_prompts
load_templates = engine.load_templates
read_task = engine.read_task
prompt_path = engine.prompt_path

# --- output artifact paths --------------------------------------------------
spawn_out = engine.spawn_out
attack_out = engine.attack_out
defense_out = engine.defense_out
revised_out = engine.revised_out
verdict_out = engine.verdict_out
final_out = engine.final_out

# --- judging, elimination, reporting ----------------------------------------
weighted_total = engine.weighted_total
decide = engine.decide
collect = engine.collect
record = engine.record
advance = engine.advance
rounds_played = engine.rounds_played
alive_ids = engine.alive_ids
winner_report = engine.winner_report
status_rows = engine.status_rows
plan_totals = engine.plan_totals
plan_rows = engine.plan_rows
norm_agent = engine.norm_agent
find_match = engine.find_match

# --- constants and engine paths --------------------------------------------
WEIGHTS = engine.WEIGHTS
PHASES = engine.PHASES
RUBRIC_PATH = engine.RUBRIC_PATH
SKILL_PATH = engine.SKILL_PATH
STRATEGIES_PATH = engine.STRATEGIES_PATH
DEFAULT_WAVE = engine.DEFAULT_WAVE
ArenaError = engine.ArenaError
ROOT = engine.ROOT
LATEST = engine.LATEST
STATE_FILE = engine.STATE_FILE
DEFAULT_AGENTS = engine.DEFAULT_AGENTS
QUICK_AGENTS = engine.QUICK_AGENTS
