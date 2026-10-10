"""Context isolation, common-task propagation, card uniqueness and blind judging.

These are the tests that separate a genuine multi-request tournament from one
assistant playing every part. They assert, from the artifacts actually produced:

  - every competitor receives the task text byte for byte identical
  - every competitor receives a distinct strategy card
  - a competitor's spawn prompt contains no other competitor's draft
  - an attacker sees only its own and its opponent's solution
  - a defender sees only its own solution and the attacks against it
  - a judge sees no competitor ID, no provider name and no model name
  - a judge verdict is mapped back to real IDs, never invented by the controller

Offline: every response comes from the scripted test double.
"""

import json
import unittest

from helper import TempWorkspace, make_state, read

from arena_runner.anonymize import (LABELS, de_anonymize_verdict, label_map,
                                    parse_verdict)
from arena_runner.bracket import phase_jobs, read_task, write_prompts
from arena_runner.materialize import input_paths, materialize
from arena_runner.providers import ScriptedMockProvider
from arena_runner.runner import TournamentRunner
from arena_runner.validation import validate


def build_all_prompts(state):
    """Render and materialise every spawn prompt for a fresh run."""
    jobs = write_prompts(state, "spawn")
    runner = TournamentRunner(state, ScriptedMockProvider())
    out = {}
    for job in jobs:
        prompt, isolation = runner.build_prompt(job, read(job["prompt"]))
        out[job["agent"]] = (prompt, isolation)
    return out


class CommonTaskPropagation(unittest.TestCase):
    def test_every_competitor_gets_the_task_byte_for_byte(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=11)
            task = read_task(state)
            prompts = build_all_prompts(state)
            self.assertEqual(len(prompts), 4)
            for agent, (prompt, _iso) in prompts.items():
                self.assertIn(task, prompt,
                              "%s did not receive the exact task text" % agent)

    def test_task_text_is_not_paraphrased_between_competitors(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=11)
            prompts = build_all_prompts(state)
            occurrences = [p.count(read_task(state)) for p, _ in prompts.values()]
            self.assertEqual(occurrences, [1, 1, 1, 1])

    def test_a_task_containing_braces_survives_unchanged(self):
        with TempWorkspace() as ws:
            weird = "Task: render {{placeholder}} and {literal} exactly.\n"
            state = make_state(ws, agents=2, seed=3)
            with open(ws.path("run", "task.md"), "w", encoding="utf-8") as fh:
                fh.write(weird)
            prompts = build_all_prompts(state)
            for prompt, _ in prompts.values():
                self.assertIn("{{placeholder}}", prompt)
                self.assertIn("{literal}", prompt)


class CardUniqueness(unittest.TestCase):
    def test_cards_are_distinct_across_competitors(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=8, seed=5)
            cards = [json.dumps(a["card"], sort_keys=True)
                     for a in state["agents"].values()]
            self.assertEqual(len(set(cards)), len(cards))

    def test_each_card_appears_in_exactly_one_prompt(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=5)
            prompts = build_all_prompts(state)
            for agent, (prompt, _iso) in prompts.items():
                card = state["agents"][agent]["card"]
                self.assertIn(card["reasoning"]["name"], prompt)
                self.assertIn(card["workflow"]["name"], prompt)
                self.assertIn(card["strategy"]["name"], prompt)
                for other, other_prompt in prompts.items():
                    if other == agent:
                        continue
                    self.assertNotIn(card["reasoning"]["how"], other_prompt,
                                     "%s's card leaked into %s" % (agent, other))

    def test_cards_are_stable_for_a_seed(self):
        with TempWorkspace() as ws:
            a = make_state(ws, agents=4, seed=99)
            first = [x["card"] for x in a["agents"].values()]
            ws2dir = ws.path("run2")
            from arena_runner.runner import init_run
            b = init_run(ws2dir, agents=4, seed=99, task_file=ws.path("task.md"))
            second = [x["card"] for x in b["agents"].values()]
            self.assertEqual(first, second)


class SpawnIsolation(unittest.TestCase):
    """A competitor's first response must not be influenced by any other draft."""

    def test_spawn_prompt_contains_no_other_competitor_solution(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            # Rendering the briefs creates the output directories; plant a private
            # draft in each, then re-render and check nothing crossed over.
            write_prompts(state, "spawn")
            for aid, agent in state["agents"].items():
                with open(agent["solution"], "w", encoding="utf-8") as fh:
                    fh.write("PRIVATE DRAFT OF %s - must never leak\n" % aid)
            prompts = build_all_prompts(state)
            for agent, (prompt, _iso) in prompts.items():
                for other in state["agents"]:
                    if other == agent:
                        continue
                    self.assertNotIn("PRIVATE DRAFT OF %s" % other, prompt,
                                     "%s saw %s's private draft" % (agent, other))

    def test_spawn_input_allowlist_is_only_the_baseline(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=3, seed=7, baseline=True)
            jobs = write_prompts(state, "spawn")
            for job in jobs:
                labels = [label for label, _ in input_paths(state, job)]
                self.assertEqual(labels, ["baseline"])

    def test_spawn_without_baseline_inlines_nothing(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=2, seed=7, baseline=False)
            jobs = write_prompts(state, "spawn")
            for job in jobs:
                self.assertEqual(input_paths(state, job), [])
                prompt, isolation = materialize(state, job, read(job["prompt"]))
                self.assertEqual(isolation["inlined"], {})

    def test_baseline_is_shared_identically(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=3, seed=7, baseline=True)
            prompts = build_all_prompts(state)
            baseline = read(ws.path("run", "baseline.md")).strip()
            for prompt, _ in prompts.values():
                self.assertIn(baseline, prompt)

    def test_contexts_are_pairwise_distinct(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            prompts = build_all_prompts(state)
            fingerprints = set()
            for agent, (prompt, _iso) in prompts.items():
                from arena_runner.providers import CompletionRequest
                fingerprints.add(CompletionRequest(agent, "competitor", prompt)
                                 .context_fingerprint())
            self.assertEqual(len(fingerprints), 4)


class PhaseInputIsolation(unittest.TestCase):
    def _played_round(self, ws, agents=4):
        """Run spawn + attack so attack/defend jobs have real inputs."""
        state = make_state(ws, agents=agents, seed=7)
        runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
        runner.run_phase("spawn")
        runner.run_phase("attack")
        return state, runner

    def test_attacker_sees_only_its_own_and_its_opponents_solution(self):
        with TempWorkspace() as ws:
            state, runner = self._played_round(ws)
            jobs = write_prompts(state, "attack")
            for job in jobs:
                labels = {label for label, _ in input_paths(state, job)}
                self.assertEqual(labels, {"own_solution", "target_solution"})
                prompt, isolation = runner.build_prompt(job, read(job["prompt"]))
                self.assertEqual(set(isolation["inlined"]), labels)
                bystanders = [a for a in state["agents"]
                              if a not in (job["agent"], job["opponent"])]
                for other in bystanders:
                    self.assertNotIn(state["agents"][other]["solution"],
                                     isolation["inlined"].values(),
                                     "%s saw uninvolved %s" % (job["id"], other))

    def test_defender_sees_only_its_own_solution_and_the_attacks_against_it(self):
        with TempWorkspace() as ws:
            state, runner = self._played_round(ws)
            jobs = write_prompts(state, "defend")
            for job in jobs:
                labels = {label for label, _ in input_paths(state, job)}
                self.assertEqual(labels, {"own_solution", "attacks"})
                _prompt, isolation = runner.build_prompt(job, read(job["prompt"]))
                self.assertEqual(set(isolation["inlined"]), labels)

    def test_defender_does_not_see_the_opponents_solution(self):
        with TempWorkspace() as ws:
            state, runner = self._played_round(ws)
            for job in write_prompts(state, "defend"):
                prompt, _iso = runner.build_prompt(job, read(job["prompt"]))
                opponent_solution = state["agents"][job["opponent"]]["solution"]
                self.assertNotIn(read(opponent_solution).strip(), prompt)

    def test_attacks_are_attributed_to_the_right_defender(self):
        with TempWorkspace() as ws:
            state, runner = self._played_round(ws)
            for job in write_prompts(state, "defend"):
                from arena_runner.bracket import attack_out
                expected = attack_out(state["dir"], job["round"], job["match"],
                                      job["opponent"])
                labels = dict(input_paths(state, job))
                self.assertEqual(labels["attacks"], expected)


class BlindJudging(unittest.TestCase):
    def _to_judge_phase(self, ws, agents=4):
        state = make_state(ws, agents=agents, seed=7)
        runner = TournamentRunner(state, ScriptedMockProvider(), backoff=0.0)
        for phase in ("spawn", "attack", "defend"):
            runner.run_phase(phase)
        return state, runner

    def test_judge_prompt_hides_competitor_ids(self):
        with TempWorkspace() as ws:
            state, runner = self._to_judge_phase(ws)
            for job in write_prompts(state, "judge"):
                prompt, _iso = runner.build_prompt(job, read(job["prompt"]))
                for agent in state["agents"]:
                    self.assertNotIn(agent, prompt,
                                     "%s leaked into the judge prompt" % agent)

    def test_judge_prompt_uses_neutral_labels(self):
        with TempWorkspace() as ws:
            state, runner = self._to_judge_phase(ws)
            for job in write_prompts(state, "judge"):
                prompt, _iso = runner.build_prompt(job, read(job["prompt"]))
                for label in LABELS:
                    self.assertIn(label, prompt)

    def test_judge_prompt_hides_provider_and_model(self):
        with TempWorkspace() as ws:
            state = make_state(ws, agents=4, seed=7)
            provider = ScriptedMockProvider(model="secret-model-name-9")
            runner = TournamentRunner(state, provider, backoff=0.0)
            for phase in ("spawn", "attack", "defend"):
                runner.run_phase(phase)
            for job in write_prompts(state, "judge"):
                prompt, _iso = runner.build_prompt(job, read(job["prompt"]))
                self.assertNotIn("secret-model-name-9", prompt)
                self.assertNotIn(provider.name, prompt)

    def test_judge_sees_the_rubric(self):
        with TempWorkspace() as ws:
            state, runner = self._to_judge_phase(ws)
            rubric = read(ws.path("run", "rubric.md"))
            for job in write_prompts(state, "judge"):
                prompt, _iso = runner.build_prompt(job, read(job["prompt"]))
                self.assertIn("Correctness", prompt)
                self.assertIn("fatal", rubric)

    def test_verdict_is_mapped_back_to_real_ids(self):
        mapping = label_map("a001", "a004")
        self.assertEqual(mapping, {"solA": "a001", "solB": "a004"})
        raw = {"scores": {"solA": {"correctness": 9, "completeness": 9,
                                   "specificity": 9, "robustness": 9,
                                   "clarity": 9, "fatal": False},
                          "solB": {"correctness": 5, "completeness": 5,
                                   "specificity": 5, "robustness": 5,
                                   "clarity": 5, "fatal": False}},
               "winner": "solB", "reason": "test",
               "standing": {"solA": [], "solB": ["x"]}}
        mapped = de_anonymize_verdict(raw, "a001", "a004")
        self.assertEqual(set(mapped["scores"]), {"a001", "a004"})
        self.assertEqual(mapped["winner"], "a004")
        self.assertEqual(set(mapped["standing"]), {"a001", "a004"})
        self.assertTrue(mapped["_blinded"])

    def test_unmappable_verdict_is_rejected_not_guessed(self):
        raw = {"scores": {"solA": {}, "ghost": {}}, "winner": "solA", "reason": "r"}
        with self.assertRaises(ValueError):
            de_anonymize_verdict(raw, "a001", "a004")

    def test_ambiguous_winner_is_rejected(self):
        raw = {"scores": {"solA": {}, "solB": {}}, "winner": "solA or solB",
               "reason": "r"}
        with self.assertRaises(ValueError):
            de_anonymize_verdict(raw, "a001", "a004")

    def test_unknown_winner_is_rejected(self):
        raw = {"scores": {"solA": {}, "solB": {}}, "winner": "nobody", "reason": "r"}
        with self.assertRaises(ValueError):
            de_anonymize_verdict(raw, "a001", "a004")

    def test_judge_output_is_validated_before_it_is_written(self):
        with TempWorkspace() as ws:
            state, runner = self._to_judge_phase(ws)
            job = write_prompts(state, "judge")[0]
            good = json.dumps({
                "scores": {"solA": {"correctness": 8, "completeness": 8,
                                    "specificity": 8, "robustness": 8,
                                    "clarity": 8, "fatal": False},
                           "solB": {"correctness": 6, "completeness": 6,
                                    "specificity": 6, "robustness": 6,
                                    "clarity": 6, "fatal": False}},
                "winner": "solA", "reason": "decisive difference",
                "standing": {"solA": [], "solB": []}})
            artifacts = validate(job, good, first=job["a"], second=job["b"])
            written = json.loads(list(artifacts.values())[0])
            self.assertEqual(set(written["scores"]), {job["a"], job["b"]})
            self.assertIn(written["winner"], (job["a"], job["b"]))

    def test_out_of_range_score_is_rejected(self):
        from arena_runner.errors import MalformedResponse
        with TempWorkspace() as ws:
            state, runner = self._to_judge_phase(ws)
            job = write_prompts(state, "judge")[0]
            bad = json.dumps({
                "scores": {"solA": {"correctness": 99, "completeness": 8,
                                    "specificity": 8, "robustness": 8,
                                    "clarity": 8, "fatal": False},
                           "solB": {"correctness": 6, "completeness": 6,
                                    "specificity": 6, "robustness": 6,
                                    "clarity": 6, "fatal": False}},
                "winner": "solA", "reason": "r"})
            with self.assertRaises(MalformedResponse):
                validate(job, bad, first=job["a"], second=job["b"])

    def test_missing_score_key_is_rejected(self):
        from arena_runner.errors import MalformedResponse
        with TempWorkspace() as ws:
            state, runner = self._to_judge_phase(ws)
            job = write_prompts(state, "judge")[0]
            bad = json.dumps({"scores": {"solA": {"correctness": 8},
                                         "solB": {"correctness": 6}},
                              "winner": "solA", "reason": "r"})
            with self.assertRaises(MalformedResponse):
                validate(job, bad, first=job["a"], second=job["b"])


class FatalAndTieRulesSurviveBlinding(unittest.TestCase):
    """The engine's own fatal-flaw and tie logic must still decide the match."""

    def _verdict(self, a_scores, b_scores, winner, standing=None):
        return json.dumps({
            "scores": {"solA": a_scores, "solB": b_scores},
            "winner": winner, "reason": "test reason",
            "standing": standing or {"solA": [], "solB": []}})

    def _scores(self, base, fatal=False):
        return {"correctness": base, "completeness": base, "specificity": base,
                "robustness": base, "clarity": base, "fatal": fatal}

    def _decide(self, ws, text, first="a001", second="a004"):
        state = make_state(ws, agents=4, seed=7)
        job = {"kind": "judge", "outputs": [ws.path("verdict.json")],
               "a": first, "b": second, "match": "r1-m01", "round": 1,
               "prompt": ws.path("brief.md"), "id": "r1-m01.judge"}
        with open(job["prompt"], "w") as fh:
            fh.write("brief")
        artifacts = validate(job, text, first=first, second=second)
        from arena_runner.bracket import decide
        return decide(json.loads(list(artifacts.values())[0]), first, second)

    def test_fatal_solution_loses_despite_a_higher_total(self):
        with TempWorkspace() as ws:
            text = self._verdict(self._scores(9, fatal=True), self._scores(5),
                                 winner="solA")
            winner, totals, _note = self._decide(ws, text)
            self.assertEqual(winner, "a004")
            self.assertGreater(totals["a001"], totals["a004"])

    def test_higher_weighted_total_wins_over_the_judges_pick(self):
        with TempWorkspace() as ws:
            text = self._verdict(self._scores(5), self._scores(9), winner="solA")
            winner, totals, note = self._decide(ws, text)
            self.assertEqual(winner, "a004")
            self.assertGreater(totals["a004"], totals["a001"])
            self.assertIn("scores favour", note)

    def test_exact_tie_goes_to_fewer_standing_attacks(self):
        with TempWorkspace() as ws:
            text = self._verdict(self._scores(7), self._scores(7), "solB",
                                 standing={"solA": ["x", "y"], "solB": ["x"]})
            winner, _totals, _note = self._decide(ws, text)
            self.assertEqual(winner, "a004")


if __name__ == "__main__":
    unittest.main()
