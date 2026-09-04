import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  encodeFeedback,
  feedbackCodeWords,
  filterCandidates,
  prepareModel,
  solveBoard,
} from "../../web/solver-core.js";

function feedbackStates(code) {
  return Array.from({ length: 5 }, () => {
    const state = code % 3;
    code = Math.floor(code / 3);
    return state;
  });
}

function fixtureModel() {
  return prepareModel({
    schemaVersion: 1,
    actionWords: ["cigar", "rebut", "soare"],
    answerActionIndices: [0, 1],
    primaryPrior: [0.7, 0.3],
    auxiliaryPrior: [0.6, 0.4],
    answerLexicalZ: [0.2, -0.2],
    classicSolutionCount: 2,
    allGreenCode: 242,
    trainedThrough: "2026-01-01",
    policy: {
      starter: "soare",
      direct_hit_weight: 3,
      late_hit_weight: 0,
      exploit_threshold: 0.5,
      exact_endgame_limit: 3,
      tail_wordfreq_weight: 0,
      tail_wordfreq_gap: 0.1,
      tail_wordfreq_start_turn: 3,
      expanded_direct_hit_factor: 1.5,
      expanded_language_override: false,
      expanded_language_min_probability: 0.2,
      expanded_language_editorial_min: 0.15,
      expanded_language_editorial_max: 0.2,
      expanded_language_min_candidates: 3,
      expanded_language_max_candidates: 20,
      expanded_language_turn: 3,
    },
  });
}

test("feedback encoding matches duplicate-letter Wordle semantics", () => {
  assert.equal(feedbackCodeWords("allee", "apple"), 167);
  assert.equal(encodeFeedback([2, 1, 0, 0, 2]), 167);
});

test("board filtering retains only answers matching every tile", () => {
  const model = fixtureModel();
  const feedback = feedbackStates(feedbackCodeWords("soare", "cigar"));
  assert.deepEqual(filterCandidates(model, [{ word: "soare", feedback }]), [0]);
});

test("the browser policy uses the selected starter and resolves a singleton", async () => {
  const model = fixtureModel();
  const opening = await solveBoard(model, []);
  assert.equal(opening.word, "soare");
  assert.equal(opening.reason, "starter");
  assert.ok(Number.isFinite(opening.entropy));
  assert.ok(Number.isFinite(opening.utility));

  const feedback = feedbackStates(feedbackCodeWords("soare", "cigar"));
  const next = await solveBoard(model, [{ word: "soare", feedback }]);
  assert.equal(next.word, "cigar");
  assert.equal(next.reason, "only-candidate");
});

test("the published browser snapshot matches the Python policy parity case", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../../web/model.json", import.meta.url), "utf8"),
  );
  const model = prepareModel(snapshot);
  const feedback = feedbackStates(feedbackCodeWords("soare", "click"));
  const result = await solveBoard(model, [{ word: "soare", feedback }]);

  assert.equal(result.candidateCount, 421);
  assert.equal(result.word, "milty");
  assert.equal(result.reason, "information");
  assert.equal(result.hitWeight, 3);
  assert.ok(Number.isFinite(result.entropy));
  assert.ok(Number.isFinite(result.utility));
  assert.ok(result.utility >= result.entropy);
});

test("an information-only recommendation is identified as a zero-hit probe", async () => {
  const snapshot = JSON.parse(
    await readFile(new URL("../../web/model.json", import.meta.url), "utf8"),
  );
  const model = prepareModel(snapshot);
  const result = await solveBoard(model, [
    { word: "soare", feedback: [0, 0, 2, 0, 2] },
  ]);

  assert.equal(result.candidateCount, 50);
  assert.equal(result.word, "clipt");
  assert.equal(result.reason, "information");
  assert.equal(result.answerProbability, 0);
  assert.equal(result.effectiveHitProbability, 0);
  assert.equal(result.utility, result.entropy);
  assert.equal(result.topCandidates[0].word, "place");
});
