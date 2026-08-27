const FEEDBACK_STATES = ["absent", "present", "correct"];

export function encodeFeedback(states) {
  if (!Array.isArray(states) || states.length !== 5) {
    throw new Error("Feedback must contain five tile states.");
  }
  let code = 0;
  let multiplier = 1;
  for (const state of states) {
    const value = typeof state === "number" ? state : FEEDBACK_STATES.indexOf(state);
    if (value < 0 || value > 2) {
      throw new Error(`Unknown feedback state: ${state}`);
    }
    code += value * multiplier;
    multiplier *= 3;
  }
  return code;
}

export function feedbackCodeWords(guess, answer) {
  if (!/^[a-z]{5}$/.test(guess) || !/^[a-z]{5}$/.test(answer)) {
    throw new Error("Guess and answer must be lowercase five-letter words.");
  }
  const result = [0, 0, 0, 0, 0];
  const counts = new Uint8Array(26);
  for (let position = 0; position < 5; position += 1) {
    if (guess[position] === answer[position]) {
      result[position] = 2;
    } else {
      counts[answer.charCodeAt(position) - 97] += 1;
    }
  }
  for (let position = 0; position < 5; position += 1) {
    if (result[position] !== 0) continue;
    const letter = guess.charCodeAt(position) - 97;
    if (counts[letter] > 0) {
      result[position] = 1;
      counts[letter] -= 1;
    }
  }
  return encodeFeedback(result);
}

export function prepareModel(raw) {
  const requiredArrays = [
    "actionWords",
    "answerActionIndices",
    "primaryPrior",
    "auxiliaryPrior",
    "answerLexicalZ",
  ];
  if (raw.schemaVersion !== 1 || requiredArrays.some((field) => !Array.isArray(raw[field]))) {
    throw new Error("Unsupported or incomplete Quintropy model snapshot.");
  }
  const answerCount = raw.answerActionIndices.length;
  if (
    raw.primaryPrior.length !== answerCount ||
    raw.auxiliaryPrior.length !== answerCount ||
    raw.answerLexicalZ.length !== answerCount
  ) {
    throw new Error("Model answer arrays are not aligned.");
  }

  const encodedActions = new Uint8Array(raw.actionWords.length * 5);
  const actionIndex = new Map();
  raw.actionWords.forEach((word, action) => {
    if (!/^[a-z]{5}$/.test(word) || actionIndex.has(word)) {
      throw new Error("Model action vocabulary is invalid.");
    }
    actionIndex.set(word, action);
    for (let position = 0; position < 5; position += 1) {
      encodedActions[action * 5 + position] = word.charCodeAt(position) - 97;
    }
  });

  const primaryPrior = Float64Array.from(raw.primaryPrior);
  const auxiliaryPrior = Float64Array.from(raw.auxiliaryPrior);
  const primaryTotal = primaryPrior.reduce((sum, value) => sum + value, 0);
  const auxiliaryTotal = auxiliaryPrior.reduce((sum, value) => sum + value, 0);
  if (
    !Number.isFinite(primaryTotal) ||
    !Number.isFinite(auxiliaryTotal) ||
    primaryTotal <= 0 ||
    auxiliaryTotal <= 0 ||
    primaryPrior.some((value) => !Number.isFinite(value) || value < 0) ||
    auxiliaryPrior.some((value) => !Number.isFinite(value) || value < 0)
  ) {
    throw new Error("Model priors are invalid.");
  }

  return {
    ...raw,
    actionIndex,
    encodedActions,
    answerActionIndices: Int32Array.from(raw.answerActionIndices),
    primaryPrior,
    auxiliaryPrior,
    answerLexicalZ: Float64Array.from(raw.answerLexicalZ),
  };
}

function feedbackCodeAt(model, guessAction, answerAction) {
  const guessOffset = guessAction * 5;
  const answerOffset = answerAction * 5;
  const result = [0, 0, 0, 0, 0];
  const counts = new Uint8Array(26);
  for (let position = 0; position < 5; position += 1) {
    const guessLetter = model.encodedActions[guessOffset + position];
    const answerLetter = model.encodedActions[answerOffset + position];
    if (guessLetter === answerLetter) {
      result[position] = 2;
    } else {
      counts[answerLetter] += 1;
    }
  }
  for (let position = 0; position < 5; position += 1) {
    if (result[position] !== 0) continue;
    const letter = model.encodedActions[guessOffset + position];
    if (counts[letter] > 0) {
      result[position] = 1;
      counts[letter] -= 1;
    }
  }
  return result[0] + result[1] * 3 + result[2] * 9 + result[3] * 27 + result[4] * 81;
}

export function filterCandidates(model, rows) {
  let candidates = Array.from({ length: model.answerActionIndices.length }, (_, index) => index);
  for (const row of rows) {
    const word = String(row.word || "").toLowerCase();
    const guessAction = model.actionIndex.get(word);
    if (guessAction === undefined) {
      throw new Error(`${word.toUpperCase()} is not in the playable vocabulary.`);
    }
    const expected = encodeFeedback(row.feedback);
    candidates = candidates.filter(
      (answerPosition) =>
        feedbackCodeAt(model, guessAction, model.answerActionIndices[answerPosition]) === expected,
    );
  }
  return candidates;
}

function normalizeCandidateWeights(source, candidates) {
  const weights = candidates.map((position) => source[position]);
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) throw new Error("Candidate posterior has no probability mass.");
  return weights.map((value) => value / total);
}

function entropyForAction(model, action, candidates, posterior) {
  const mass = new Float64Array(243);
  for (let index = 0; index < candidates.length; index += 1) {
    const answerAction = model.answerActionIndices[candidates[index]];
    mass[feedbackCodeAt(model, action, answerAction)] += posterior[index];
  }
  let entropy = 0;
  for (const value of mass) {
    if (value > 0) entropy -= value * Math.log2(value);
  }
  return entropy;
}

function topCandidateDetails(model, candidates, posterior, limit = 5) {
  return candidates
    .map((position, index) => ({
      word: model.actionWords[model.answerActionIndices[position]],
      probability: posterior[index],
    }))
    .sort((left, right) => right.probability - left.probability || left.word.localeCompare(right.word))
    .slice(0, limit);
}

function expectedCost(model, action, state, prior, value) {
  const total = state.reduce((sum, position) => sum + prior[position], 0);
  const groups = new Map();
  for (const position of state) {
    const code = feedbackCodeAt(model, action, model.answerActionIndices[position]);
    const group = groups.get(code) || [];
    group.push(position);
    groups.set(code, group);
  }
  let cost = 1;
  for (const [code, group] of groups) {
    if (code === model.allGreenCode) continue;
    if (group.length === state.length) return Number.POSITIVE_INFINITY;
    const mass = group.reduce((sum, position) => sum + prior[position], 0) / total;
    cost += mass * value(group);
  }
  return cost;
}

function exactEndgameAction(model, candidates, prior) {
  const memo = new Map();
  const value = (state) => {
    if (state.length === 1) return 1;
    const key = [...state].sort((a, b) => a - b).join(",");
    if (memo.has(key)) return memo.get(key);
    let best = Number.POSITIVE_INFINITY;
    for (let action = 0; action < model.actionWords.length; action += 1) {
      best = Math.min(best, expectedCost(model, action, state, prior, value));
    }
    memo.set(key, best);
    return best;
  };

  let bestAction = 0;
  let bestCost = Number.POSITIVE_INFINITY;
  for (let action = 0; action < model.actionWords.length; action += 1) {
    const cost = expectedCost(model, action, candidates, prior, value);
    if (cost < bestCost) {
      bestCost = cost;
      bestAction = action;
    }
  }
  return { action: bestAction, expectedGuesses: bestCost };
}

function resultForAction(model, action, reason, candidates, posterior, entropy = null) {
  const actionProbability = candidates.reduce(
    (probability, position, index) =>
      model.answerActionIndices[position] === action ? posterior[index] : probability,
    0,
  );
  return {
    word: model.actionWords[action],
    reason,
    candidateCount: candidates.length,
    answerProbability: actionProbability,
    entropy,
    topCandidates: topCandidateDetails(model, candidates, posterior),
  };
}

export async function solveBoard(model, rows, onProgress = () => {}) {
  const started = performance.now();
  const candidates = filterCandidates(model, rows);
  if (candidates.length === 0) return { noCandidates: true, candidateCount: 0 };

  const turn = rows.length + 1;
  const primaryPosterior = normalizeCandidateWeights(model.primaryPrior, candidates);
  if (turn === 1) {
    const starter = model.actionIndex.get(model.policy.starter);
    const entropy = entropyForAction(model, starter, candidates, primaryPosterior);
    const result = resultForAction(
      model,
      starter,
      "starter",
      candidates,
      primaryPosterior,
      entropy,
    );
    return {
      ...result,
      utility: entropy + model.policy.direct_hit_weight * result.answerProbability,
      hitWeight: model.policy.direct_hit_weight,
      effectiveHitProbability: result.answerProbability,
      durationMs: performance.now() - started,
    };
  }

  let adjustedPosterior = primaryPosterior;
  if (
    model.policy.tail_wordfreq_weight > 0 &&
    turn >= model.policy.tail_wordfreq_start_turn &&
    candidates.length >= 2
  ) {
    const lexical = candidates.map((position) => model.answerLexicalZ[position]).sort((a, b) => b - a);
    if (lexical[0] - lexical[1] >= model.policy.tail_wordfreq_gap) {
      adjustedPosterior = primaryPosterior.map(
        (value, index) =>
          value * Math.exp(model.policy.tail_wordfreq_weight * model.answerLexicalZ[candidates[index]]),
      );
      const total = adjustedPosterior.reduce((sum, value) => sum + value, 0);
      adjustedPosterior = adjustedPosterior.map((value) => value / total);
    }
  }

  const order = adjustedPosterior
    .map((probability, index) => ({ probability, index }))
    .sort((left, right) => right.probability - left.probability);
  const mapPosition = candidates[order[0].index];
  const mapAction = model.answerActionIndices[mapPosition];

  if (candidates.length === 1) {
    return {
      ...resultForAction(model, mapAction, "only-candidate", candidates, adjustedPosterior, 0),
      durationMs: performance.now() - started,
    };
  }

  if (candidates.length <= model.policy.exact_endgame_limit) {
    onProgress({ phase: "endgame", completed: 0, total: model.actionWords.length });
    const exactPrior = Float64Array.from(model.primaryPrior);
    candidates.forEach((position, index) => {
      exactPrior[position] = adjustedPosterior[index];
    });
    const exactDecision = exactEndgameAction(model, candidates, exactPrior);
    return {
      ...resultForAction(
        model,
        exactDecision.action,
        "exact-endgame",
        candidates,
        adjustedPosterior,
        entropyForAction(model, exactDecision.action, candidates, adjustedPosterior),
      ),
      expectedGuesses: exactDecision.expectedGuesses,
      durationMs: performance.now() - started,
    };
  }

  const remainingAfterGuess = 6 - turn;
  if (candidates.length <= remainingAfterGuess + 1) {
    return {
      ...resultForAction(
        model,
        mapAction,
        "safe-map",
        candidates,
        adjustedPosterior,
        entropyForAction(model, mapAction, candidates, adjustedPosterior),
      ),
      durationMs: performance.now() - started,
    };
  }

  const policy = model.policy;
  if (
    policy.expanded_language_override &&
    turn === policy.expanded_language_turn &&
    candidates.length >= policy.expanded_language_min_candidates &&
    candidates.length <= policy.expanded_language_max_candidates
  ) {
    const linguisticPosterior = normalizeCandidateWeights(model.auxiliaryPrior, candidates);
    const top = linguisticPosterior.reduce(
      (best, probability, index) => (probability > linguisticPosterior[best] ? index : best),
      0,
    );
    const action = model.answerActionIndices[candidates[top]];
    if (
      action >= model.classicSolutionCount &&
      linguisticPosterior[top] >= policy.expanded_language_min_probability &&
      primaryPosterior[top] >= policy.expanded_language_editorial_min &&
      primaryPosterior[top] <= policy.expanded_language_editorial_max
    ) {
      return {
        ...resultForAction(
          model,
          action,
          "linguistic-tail",
          candidates,
          primaryPosterior,
          entropyForAction(model, action, candidates, primaryPosterior),
        ),
        durationMs: performance.now() - started,
      };
    }
  }

  if (order[0].probability >= policy.exploit_threshold) {
    return {
      ...resultForAction(
        model,
        mapAction,
        "high-confidence",
        candidates,
        adjustedPosterior,
        entropyForAction(model, mapAction, candidates, adjustedPosterior),
      ),
      durationMs: performance.now() - started,
    };
  }

  const candidateProbability = new Map();
  candidates.forEach((position, index) => {
    candidateProbability.set(model.answerActionIndices[position], adjustedPosterior[index]);
  });
  const hitWeight = policy.direct_hit_weight + policy.late_hit_weight * Math.max(turn - 2, 0);
  let bestAction = mapAction;
  let bestScore = Number.NEGATIVE_INFINITY;
  let bestEntropy = 0;
  let bestEffectiveHitProbability = 0;
  for (let action = 0; action < model.actionWords.length; action += 1) {
    const entropy = entropyForAction(model, action, candidates, adjustedPosterior);
    let hitProbability = candidateProbability.get(action) || 0;
    if (action >= model.classicSolutionCount) hitProbability *= policy.expanded_direct_hit_factor;
    const score = entropy + hitWeight * hitProbability;
    if (score > bestScore) {
      bestScore = score;
      bestAction = action;
      bestEntropy = entropy;
      bestEffectiveHitProbability = hitProbability;
    }
    if (action % 256 === 0) {
      onProgress({ phase: "search", completed: action, total: model.actionWords.length });
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  onProgress({ phase: "search", completed: model.actionWords.length, total: model.actionWords.length });
  return {
    ...resultForAction(model, bestAction, "information", candidates, adjustedPosterior, bestEntropy),
    utility: bestScore,
    hitWeight,
    effectiveHitProbability: bestEffectiveHitProbability,
    durationMs: performance.now() - started,
  };
}
