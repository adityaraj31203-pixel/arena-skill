# Arena migration: independent multi-agent execution

This document covers the migration from the Claude Code sub-agent path to a
provider-agnostic runner that can be operated from a phone through GitHub
Actions.

The original skill is unchanged. `skills/arena/bracket.py` and its 27 tests are
untouched, and the original Claude Code path still works exactly as before.

Original work Copyright (c) 2025 Jake Schincariol, MIT License. The additions
described here are offered under the same MIT License.

---

## 1. What was built and what was left alone

| Path | Status |
| --- | --- |
| `skills/arena/bracket.py` | Unchanged. Still the tournament engine. |
| `skills/arena/SKILL.md`, `rubric.md`, `strategies.json` | Unchanged. |
| `tests/test_bracket.py` | Unchanged. Still 27 passing tests. |
| `.claude-plugin/*` | Unchanged. The Claude Code plugin still installs. |
| `arena_runner/` | New. Provider adapters, executor, budget, reporting. |
| `.github/workflows/` | New. CI plus two manual tournament workflows. |
| `docs/MIGRATION.md` | This file. |
| `README.md` | Original text preserved; a new section appended at the end. |

The runner **imports** the engine rather than reimplementing it. The engine
still owns card dealing, seeded pairing, byes, the rubric arithmetic, the
fatal-flaw rule, tie-breaking, elimination and the winner report. See
`arena_runner/bracket.py`, which re-exports the engine functions the runner uses
so the reuse is explicit and auditable.

## 2. How competitors are isolated

The one thing the migration had to replace was the Agent tool. In its place,
every job becomes one stateless HTTP request:

1. `bracket.phase_jobs` enumerates the jobs the engine needs.
2. `bracket.write_prompts` renders the same brief text a Claude Code sub-agent
   would have received.
3. `materialize.input_paths` declares, per job kind, exactly which files that
   job is allowed to see. This is an allowlist, not a best effort.
4. `materialize.materialize` inlines only those files. A competitor's spawn
   prompt therefore *cannot* contain another competitor's draft, because that
   path is not in its allowlist. `tests/test_isolation.py` asserts this against
   planted private drafts.
5. The request carries no conversation history. `CompletionRequest` has no
   message list, only a system string and one user prompt.
6. `CompletionRequest.context_fingerprint()` digests exactly what was sent. Two
   requests with the same fingerprint saw the same context, which is how the run
   record proves competitors did not share one.

Permitted inputs by phase:

| Phase | May see |
| --- | --- |
| spawn | the task, its own card, the baseline if there is one |
| attack | the task, its own card, its own solution, the opponent's solution |
| defend | the task, its own card, its own solution, the attacks against it |
| judge | the task, the rubric, both revised solutions, both attack sets, both defences |
| final | the task, the rubric, `X.md`, `Y.md` |

Judges are additionally blinded: `anonymize.anonymize` replaces both competitor
IDs with `solA`/`solB`, and `anonymize.scrub` removes any other competitor ID,
the provider name and the model name. The judge's raw reply is kept verbatim;
`de_anonymize_verdict` maps it back to real IDs before `bracket.collect` records
it. A verdict that cannot be mapped raises rather than being guessed.

**The controller never picks a winner.** `bracket.collect` reads the verdict file
the judge wrote. If no valid verdict exists, the round does not advance and the
run stops with an error. `tests/test_runner.py::ControllerNeverJudges` covers
this, including the case where a judge returns unparseable JSON.

## 3. Independence levels, and what this actually achieves

| Level | Meaning |
| --- | --- |
| 0 | Simulated: one executor imitates multiple roles |
| 1 | Isolated requests: multiple stateless API requests with separate contexts |
| 2 | Independently orchestrated: separate jobs, isolated state, validated outputs, controlled concurrency, reliable recovery |
| 3 | Multi-provider evaluation: Level 2 plus configured provider diversity |

The level is **computed from the run record**, never asserted. See
`runner.independence_report`, which reads `runlog.json` and counts distinct
context fingerprints. Its output always includes caveats.

Two things to be precise about:

- Multiple requests to the same model are separate contexts, **not** different
  underlying models. The report says so whenever only one model was used.
- Level 2 additionally requires separately orchestrated worker processes
  verified under real provider failures. The GitHub Actions path runs one
  process with a bounded thread pool, which gives concurrency control and
  recovery but not process isolation. It is reported as Level 1 unless that
  changes.

## 4. Intentional differences from the Claude Code path

1. **Receipt lines are suppressed.** The original briefs end with "reply with
   this one line and nothing else: DONE a001 ...". A sub-agent wrote files and
   then sent a receipt. An API request cannot write files, so the runner appends
   an instruction to return the artifact itself instead.
2. **Multi-output jobs are separated by a delimiter.** The defender writes a
   defence file and a revised solution. Since one request must return both, the
   runner appends an instruction to separate them with `===ARTIFACT===`.
3. **File reads become inlined content.** Paths in the brief are replaced with
   the file contents in `<arena-inline>` tags.
4. **`--agents` is capped at 64 in the workflow.** The 100-agent default is not
   offered through the browser form. It is still available on the command line.

Everything else, including all scoring and elimination logic, is the engine's.

## 5. Using it

### Locally, free

```bash
python3 -m arena_runner doctor                 # show the configuration
python3 -m arena_runner plan --agents 4        # show expected calls first
python3 -m arena_runner run --agents 2 --task-file task.md --provider scripted-mock
```

`scripted-mock` is an offline test double. It costs nothing and reports itself
as `live=False`. It is not a substitute for a real run: its outputs are
deterministic fixtures.

### Locally, live

```bash
export ARENA_ANTHROPIC_API_KEY=...
python3 -m arena_runner run --agents 2 --task-file task.md \
    --provider anthropic --model claude-3-5-haiku-latest \
    --api-key-env ARENA_ANTHROPIC_API_KEY \
    --max-calls 20 --max-minutes 20 --confirm-live
```

`--confirm-live` (or `ARENA_CONFIRM_LIVE=yes`) is mandatory for any live
provider. Without it the runner refuses and prints how many calls the run would
make.

### From a phone, through GitHub Actions

1. Open the repository in a mobile browser.
2. Actions, then **Arena tournament (manual)**, then **Run workflow**.
3. Fill in `task`. Set `agents` to `2` for a first run.
4. Leave `provider` at `scripted-mock` for a free dry run.
5. To go live: add the provider key as a repository secret (Settings, Secrets
   and variables, Actions), pick the provider, set `model`, and set `confirm` to
   `I-understand-this-may-incur-API-charges`.
6. Run it. You can close the browser. The run uploads an artifact named
   `arena-run` containing `arena.json`, `runlog.json`, every solution, attack
   and defence, and `report.md`.

Nothing needs to stay open on your phone and no local server is involved.

### Resume after a failure

```bash
python3 -m arena_runner run --resume .arena/run-20260101-120000-s7 \
    --provider anthropic --model ... --api-key-env ... --confirm-live
```

Jobs whose outputs already exist are skipped, so a resumed run does not redo or
re-charge completed work. `bracket.py next` also still works on the same
directory.

### Read the result

```bash
python3 -m arena_runner report --run-dir .arena/run-... --markdown-out report.md
```

A run is never reported as complete unless `arena.json` says a champion exists
and, when there was a baseline, the final check has a recorded result.

## 6. Provider configuration

| Provider | Secret / env var | Notes |
| --- | --- | --- |
| `scripted-mock` | none | Offline test double. No spend. |
| `mock` | none | Offline double that produces non-conforming output, for failure tests. |
| `openai-compatible` | `ARENA_OPENAI_API_KEY` | `POST {base_url}/chat/completions`. Point `base_url` at any compatible gateway. |
| `anthropic` | `ARENA_ANTHROPIC_API_KEY` | `POST {base_url}/v1/messages`. |

Both live adapters use documented, header-based authentication. Adding a
provider means subclassing `HTTPProvider` with a `body()` and `parse()`; nothing
else in the system changes. The runner never switches provider or model when a
request fails: a 4xx other than 429 is a hard failure and is recorded as one.

## 7. Cost and safety controls

Enforced before the request is made:

- maximum model calls (`--max-calls`)
- maximum prompt characters per request
- maximum output tokens requested per call
- maximum total tokens reported
- maximum elapsed wall-clock minutes
- maximum retries per job (default 2; never unlimited)

An estimated spend figure is available if you supply per-million-token prices.
**It is an estimate, not a cap.** Billing is asynchronous and token counts
arrive after the call, so no implementation counting tokens after the fact can
promise a hard spending ceiling. The runner says this in its own output rather
than implying a guarantee it cannot keep.

Other controls:

- Tournaments never start implicitly. The default provider is the test double
  and live runs need explicit confirmation.
- The tournament workflows are `workflow_dispatch` only. They do not run on push
  or pull request.
- `concurrency` with `cancel-in-progress: false` prevents two tournaments
  running at once.
- Every job has `timeout-minutes`.
- Workflow permissions are `contents: read`. No write access is requested.
- Inputs reach the job through environment variables, not the command line, so a
  hostile input string cannot break out into a shell command.
- Model output is never executed. Artifacts are written as text; no generated
  code or shell is run.

## 8. Secrets

- Credentials come only from the environment or GitHub Actions secrets.
- They are never placed in prompt text, artifacts, or the run log.
- `redact.py` scrubs known credential shapes and any secret found in the
  environment before anything is written or raised.
- `tests/test_budget_redact.py` runs a full tournament with a fake key in the
  environment and asserts the key appears in no file in the run directory.
- `.gitignore` covers `.env`, `.env.*`, `.arena/` and `.arena-workspace/`.
- CI fails if a credential-shaped string is committed.

## 9. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `refusing to start a live run without explicit confirmation` | Add `--confirm-live` locally, or set the `confirm` input in the workflow. |
| `provider X needs an API key in the environment variable Y` | The secret is not set, or the workflow input `provider` does not match the secret you added. |
| `max_calls=N is below the M calls this tournament needs` | Raise `max_calls` or lower `agents`. `plan` prints the number you need. |
| `phase judge had 1 failed job(s)` | The judge returned something that did not validate. The run stops rather than picking a winner. Re-run with `--resume`. |
| `collect could not record N verdict(s)` | A verdict file was unreadable and was moved aside as `.unreadable`. Re-run with `--resume` to re-judge that match. |
| Run stopped at the call limit | The budget did its job. Raise `max_calls` and `--resume`. |
| `no arena.json in ...` | Wrong `--run-dir`. Each run gets its own timestamped directory. |
| Actions job fails at input validation | The validator prints which input was invalid. All limits are in `workflow_inputs.LIMITS`. |

## 10. Testing

```bash
python3 -m unittest discover -s tests -v
```

195 tests, standard library only, no credentials, no network beyond a loopback
HTTP server used to exercise the adapters. CI runs this on every push and pull
request and cannot incur model charges.

The live path is exercised only by the manual **Live smoke test** workflow: two
competitors, **7 model calls** (2 spawn, then 5 for the single match: 2 attacks,
2 defences, 1 judge), opt-in, requires a configured secret and the confirmation
token. That workflow configures no baseline, so there is no final check and no
eighth call. `expected_calls(2, baseline=False)` in `arena_runner/workflow_inputs.py`
derives this number, and it agrees with the engine's own `plan_totals(2)`.

## 11. What this does not do

- It does not prove the winner is correct. Judging is a model scoring against a
  written rubric. It is an evaluation.
- It does not make one model into many models. Strategy cards change the prompt,
  not the weights.
- It does not fix a vague task.
- It does not run tournaments automatically, ever.

## 12. Known follow-up items

These were found in the pre-merge audit of the migration and are deliberately
left open. None of them blocks a run, and each is recorded here rather than
quietly dropped.

| ID | Severity | Summary |
| --- | --- | --- |
| D6 | Low-medium | `Budget.max_retries` is a single global counter, while `budget.py` and the `max_retries` workflow input describe it as per job. With `max_retries=2` the third retry anywhere in a run is refused for every job. Per-job retry accounting would match the documentation. |
| D7 | Low | `Budget.check_before_call` and `record_call` are not atomic together, so the call cap has a check-then-act window. In the real runner path the measured overshoot was zero, because the intervening run-log write happens to serialise the workers. That is an accident, not a guarantee; making the gate atomic would remove the reliance on it. |
| D8 | Low | There is no whole-run lock. `runlog.lock()` covers the collect and advance transitions only, not `run_phase` or `init_run`. Two concurrent processes pointed at one run directory could dispatch the same jobs twice. GitHub Actions prevents this with its `concurrency` group; running locally does not. |
| D9 | Low | The CI secret guard excludes `tests/*`, so a credential committed under `tests/` would not be caught by that step. The repository's test fixtures are built by concatenation so no contiguous credential shape exists today. |
| D10 | Low | `test_jobs_run_concurrently_up_to_the_limit` runs at concurrency 3 with 8 jobs. It passes consistently, but it exercises the same code path as the D1 race and could become flaky on a slower shared runner. |

D1 through D5 from that audit are fixed, and each fix has a regression test in
`tests/test_regression_defects.py` that fails against the pre-fix code.

### D11, found while verifying the fixes above

`test_the_recorded_winner_comes_from_the_judge_verdict_file` failed about one
run in six with `'a001' != 'a002'`. It was a wrong expectation in the test, not
a race, and it predates these fixes: the same test fails 2 times in 30 runs
against the pre-fix code at the pull request head.

The judge prompt embeds the absolute scratch path of the run directory, and the
offline test double derives its verdict scores from a hash of that prompt, so a
different temporary workspace yields different scores. Roughly one workspace in
six produces a verdict whose stated pick disagrees with its own weighted totals,
and the original engine documents that the scores win in that case.

The test now asserts the real invariant: re-running `bracket.decide` on the
persisted verdict file must reproduce the recorded winner and both totals. Two
tests were added beside it, one that repeats the tournament in twelve fresh
workspaces and one that pins the scores-beat-the-pick rule with fixed inputs.
Both fail against a runner that substitutes its own winner, and the second fails
against a `decide` that prefers the judge's stated pick.

The remaining cosmetic oddity is left alone on purpose: because the prompt
carries an absolute path, the offline double's verdicts are not reproducible
across run directories. That affects only the test double. A real provider does
not score by hashing the scratch path.
