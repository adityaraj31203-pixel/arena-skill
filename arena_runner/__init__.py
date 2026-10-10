"""Independent, provider-agnostic execution layer for the Arena skill.

The original tournament engine (skills/arena/bracket.py) is imported, never
modified. This package replaces the one piece that required Claude Code: the
Agent tool that spawned sub-agents. Instead of one assistant imitating many
roles, every competitor, attacker, defender and judge becomes a separate,
stateless model API request with its own isolated context.

Independence levels this package can establish:

  Level 0  one executor imitates multiple roles          (never produced here)
  Level 1  multiple stateless API requests, separate contexts
  Level 2  separate jobs, isolated state, validated outputs, bounded
           concurrency, reliable recovery
  Level 3  Level 2 plus explicitly configured provider/model diversity

Nothing here claims a level it has not verified. See docs/MIGRATION.md.

Original work Copyright (c) 2025 Jake Schincariol, MIT License.
Modifications in this directory are offered under the same MIT License.
"""

__version__ = "1.0.0"

INDEPENDENCE_LEVELS = {
    0: "Simulated: one executor imitates multiple roles",
    1: "Isolated requests: multiple stateless API requests with separate contexts",
    2: "Independently orchestrated: separate jobs, isolated state, validated "
       "outputs, controlled concurrency, reliable recovery",
    3: "Multi-provider evaluation: Level 2 plus configured provider diversity",
}
