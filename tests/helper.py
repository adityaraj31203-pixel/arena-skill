"""Shared fixtures for the migration tests. Standard library only."""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TASK = """# Task
Write the pricing-page headline for a CI caching product.

Requirements stated by the user:
- Audience: backend engineers evaluating the product.
- At most 12 words. No exclamation marks.
- Must name the concrete benefit: cutting CI build times.
- Must not use the words "revolutionary", "game-changing" or "AI-powered".
- Plain, direct tone.

Done looks like: one headline plus one sentence of supporting subcopy.
"""

BASELINE = ('The rejected answer:\n'
            '"Unleash the power of revolutionary AI-powered caching to transform builds!"\n')


class TempWorkspace:
    """A scratch directory with a task file, cleaned up on exit."""

    def __init__(self, prefix="arena-test-"):
        self.prefix = prefix
        self.dir = None

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix=self.prefix)
        return self

    def __exit__(self, *exc):
        if self.dir and os.path.isdir(self.dir):
            shutil.rmtree(self.dir, ignore_errors=True)
        return False

    def path(self, *parts):
        return os.path.join(self.dir, *parts)

    def write_task(self, text=TASK):
        p = self.path("task.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return p

    def write_baseline(self, text=BASELINE):
        p = self.path("baseline.md")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return p


def make_state(workspace, agents=2, seed=7, baseline=True):
    """Initialise a run with the migration's own init and return its state."""
    from arena_runner.runner import init_run

    task = workspace.write_task()
    base = workspace.write_baseline() if baseline else None
    return init_run(workspace.path("run"), agents=agents, seed=seed,
                    task_file=task, baseline_file=base)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()
