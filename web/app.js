const STATE_NAMES = ["absent", "present", "correct"];
const STATE_LABELS = {
  absent: "absent",
  present: "present in another position",
  correct: "correct position",
};
const board = document.querySelector("#board");
const solveButton = document.querySelector("#solve-button");
const resetButton = document.querySelector("#reset-button");
const boardError = document.querySelector("#board-error");
const copyButton = document.querySelector("#copy-button");
const states = {
  loading: document.querySelector("#loading-state"),
  empty: document.querySelector("#empty-state"),
  working: document.querySelector("#working-state"),
  recommendation: document.querySelector("#recommendation-state"),
  noCandidates: document.querySelector("#no-candidates-state"),
  solved: document.querySelector("#solved-state"),
};

const rows = Array.from({ length: 6 }, () => ({ word: "", feedback: [0, 0, 0, 0, 0] }));
let modelReady = false;
let activeRequest = 0;
let recommendedWord = "";
let modelMetadata = null;

function showState(name) {
  Object.entries(states).forEach(([key, element]) => {
    element.hidden = key !== name;
  });
}

function updateTile(tile, letter, value, rowNumber, position) {
  const state = STATE_NAMES[value];
  const nextState = STATE_NAMES[(value + 1) % STATE_NAMES.length];
  tile.textContent = letter?.toUpperCase() || "";
  tile.dataset.state = letter ? state : "empty";
  tile.disabled = !letter;
  tile.setAttribute(
    "aria-label",
    letter
      ? `Guess ${rowNumber}, letter ${position + 1}, ${letter.toUpperCase()}: ${STATE_LABELS[state]}. Click for ${STATE_LABELS[nextState]}`
      : `Guess ${rowNumber}, empty letter ${position + 1}`,
  );
  tile.title = letter
    ? `${letter.toUpperCase()}: ${STATE_LABELS[state]}. Click for ${STATE_LABELS[nextState]}.`
    : "";
}

function createBoard() {
  rows.forEach((row, rowIndex) => {
    const rowElement = document.createElement("div");
    rowElement.className = "board-row";

    const input = document.createElement("input");
    input.className = "guess-input";
    input.type = "text";
    input.inputMode = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.maxLength = 5;
    input.placeholder = `GUESS ${rowIndex + 1}`;
    input.setAttribute("aria-label", `Guess ${rowIndex + 1}`);

    const tiles = Array.from({ length: 5 }, (_, position) => {
      const tile = document.createElement("button");
      tile.className = "feedback-tile";
      tile.type = "button";
      updateTile(tile, "", 0, rowIndex + 1, position);
      tile.addEventListener("click", () => {
        row.feedback[position] = (row.feedback[position] + 1) % 3;
        updateTile(tile, row.word[position], row.feedback[position], rowIndex + 1, position);
      });
      return tile;
    });

    input.addEventListener("input", () => {
      const next = input.value.toLowerCase().replace(/[^a-z]/g, "").slice(0, 5);
      input.value = next.toUpperCase();
      if (next !== row.word) row.feedback = [0, 0, 0, 0, 0];
      row.word = next;
      tiles.forEach((tile, position) => {
        updateTile(tile, row.word[position], row.feedback[position], rowIndex + 1, position);
      });
      boardError.textContent = "";
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && modelReady) solveButton.click();
    });

    rowElement.append(input, ...tiles);
    board.append(rowElement);
  });
}

function populatedRows() {
  return rows
    .filter((row) => row.word.length > 0)
    .map((row) => ({ word: row.word, feedback: [...row.feedback] }));
}

function validateBoard() {
  const usedRows = rows.filter((row) => row.word.length > 0);
  if (usedRows.some((row) => row.word.length !== 5)) {
    return "Every entered guess must contain five letters.";
  }
  const firstEmpty = rows.findIndex((row) => row.word.length === 0);
  if (firstEmpty >= 0 && rows.slice(firstEmpty + 1).some((row) => row.word.length > 0)) {
    return "Fill guesses in chronological order without empty rows between them.";
  }
  if (usedRows.length >= 6) return "All six rows are already filled.";
  return "";
}

function percentage(value) {
  if (value <= 0) return "0%";
  if (value < 0.001) return "<0.1%";
  return `${(value * 100).toFixed(value < 0.01 ? 2 : 1)}%`;
}

function displayDate(isoDate) {
  return new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${isoDate}T00:00:00Z`));
}

function explainRecommendation(result) {
  const word = result.word.toUpperCase();
  const probability = percentage(result.answerProbability);
  const turn = populatedRows().length + 1;
  const guessesLeft = 7 - turn;

  switch (result.reason) {
    case "starter":
      return {
        lead: `Start with ${word}.`,
        detail:
          "It is the opening action fixed by the evaluated policy; no board evidence is available yet.",
      };
    case "only-candidate":
      return {
        lead: `Play ${word} — it is the only answer left.`,
        detail:
          "Exact Wordle feedback filtering, including repeated-letter rules, eliminates every other candidate.",
        equation: "P(answer = this word | board) = 100%",
      };
    case "exact-endgame":
      return {
        lead: `Play ${word} — it minimizes the expected guesses to finish.`,
        detail: `Only ${result.candidateCount} answers remain, so the solver evaluates the complete endgame tree rather than using the approximate information score.`,
        equation: `E[guesses to finish | ${word}] = ${result.expectedGuesses.toFixed(2)}`,
      };
    case "safe-map":
      return {
        lead: `Play ${word} — spend this turn on the most likely answer.`,
        detail: `${result.candidateCount} candidates remain with ${guessesLeft} guesses available, so direct coverage is safer than a separate information probe. The posterior probability of ${word} is ${probability}.`,
      };
    case "linguistic-tail":
      return {
        lead: `Play ${word} — the linguistic prior breaks a narrow tie.`,
        detail: `On this third-turn expanded-vocabulary edge case, the auxiliary orthographic and pronunciation model selects ${word}; its editorial posterior is ${probability}.`,
      };
    case "high-confidence":
      return {
        lead: `Play ${word} — its posterior has crossed the decision threshold.`,
        detail: `The model assigns ${word} a ${probability} chance of being the answer, above the policy's 50% exploitation threshold.`,
        equation: `${probability} ≥ 50% exploit threshold`,
      };
    default: {
      const entropy = result.entropy.toFixed(2);
      const hitMass = result.effectiveHitProbability.toFixed(6);
      if (result.answerProbability === 0) {
        const leadingAnswer = result.topCandidates[0];
        return {
          lead: `Use ${word} as an information probe — not as a predicted answer.`,
          detail:
            `${word} is not among the ${result.candidateCount} possible answers. It is selected ` +
            `only because its feedback patterns provide the most expected information across all ` +
            `${modelMetadata.actionCount.toLocaleString()} permitted guesses. If you prefer to ` +
            `guess an answer directly, the posterior leader is ${leadingAnswer.word.toUpperCase()} ` +
            `at ${percentage(leadingAnswer.probability)}.`,
          equation: `H(feedback | ${word}) = ${entropy} bits; direct-hit mass = 0; U = ${result.utility.toFixed(2)}`,
        };
      }
      return {
        lead: `Play ${word} — it has the highest decision utility.`,
        detail: `Across all ${modelMetadata.actionCount.toLocaleString()} permitted guesses, ${word} gives the best measured trade-off between expected feedback information and the chance of solving immediately.`,
        equation: `U(${word}) = ${entropy} bits + ${result.hitWeight.toFixed(1)} × ${hitMass} effective hit mass = ${result.utility.toFixed(2)}`,
      };
    }
  }
}

function renderRecommendation(result) {
  if (result.noCandidates) {
    showState("noCandidates");
    return;
  }
  recommendedWord = result.word.toUpperCase();
  const explanation = explainRecommendation(result);
  document.querySelector("#recommended-word").textContent = recommendedWord;
  document.querySelector("#recommendation-lead").textContent = explanation.lead;
  document.querySelector("#recommendation-reason").textContent = explanation.detail;
  const equation = document.querySelector("#decision-equation");
  equation.hidden = !explanation.equation;
  document.querySelector("#decision-equation-value").textContent = explanation.equation || "";
  document.querySelector("#candidate-count").textContent = result.candidateCount.toLocaleString();
  document.querySelector("#answer-probability").textContent =
    result.answerProbability === 0 ? "Not a candidate" : percentage(result.answerProbability);
  document.querySelector("#information-score").textContent =
    result.entropy === null ? "—" : `${result.entropy.toFixed(2)} bits`;

  const candidateList = document.querySelector("#candidate-list");
  candidateList.replaceChildren(
    ...result.topCandidates.map((candidate) => {
      const item = document.createElement("li");
      const word = document.createElement("strong");
      const probability = document.createElement("span");
      word.textContent = candidate.word.toUpperCase();
      probability.textContent = percentage(candidate.probability);
      item.append(word, probability);
      return item;
    }),
  );
  showState("recommendation");
}

function resetBoard() {
  rows.forEach((row) => {
    row.word = "";
    row.feedback = [0, 0, 0, 0, 0];
  });
  board.replaceChildren();
  createBoard();
  boardError.textContent = "";
  recommendedWord = "";
  if (modelReady) showState("empty");
}

const worker = new Worker("./solver-worker.js", { type: "module" });

worker.addEventListener("message", (event) => {
  const { type, requestId, metadata, result, progress, message } = event.data;
  if (type === "ready") {
    modelReady = true;
    modelMetadata = metadata;
    solveButton.disabled = false;
    document.querySelector("#action-count").textContent = metadata.actionCount.toLocaleString();
    document.querySelector("#answer-count").textContent = metadata.answerCount.toLocaleString();
    document.querySelector("#snapshot-date").textContent = displayDate(metadata.trainedThrough);
    document.querySelector("#model-cutoff").textContent = displayDate(metadata.trainedThrough);
    document.querySelector("#prediction-date").textContent = displayDate(metadata.predictionDate);
    document.querySelector("#starter-word").textContent = metadata.starter.toUpperCase();
    document.querySelector("#starter-explanation").textContent =
      `${metadata.starter.toUpperCase()} is the fixed opener used by the evaluated policy. ` +
      `Under this snapshot it yields ${metadata.starterEntropy.toFixed(2)} bits of expected ` +
      `feedback information, with a ${percentage(metadata.starterAnswerProbability)} direct-hit probability.`;
    showState("empty");
    return;
  }
  if (requestId !== undefined && requestId !== activeRequest) return;
  if (type === "progress") {
    const ratio = progress.total ? progress.completed / progress.total : 0;
    document.querySelector("#progress-bar").style.width = `${Math.max(4, ratio * 100)}%`;
    document.querySelector("#progress-copy").textContent =
      progress.phase === "endgame"
        ? "Solving the small state exactly."
        : `${Math.round(ratio * 100)}% of playable words scored.`;
    return;
  }
  if (type === "result") {
    solveButton.disabled = false;
    renderRecommendation(result);
    return;
  }
  if (type === "error") {
    solveButton.disabled = !modelReady;
    boardError.textContent = message || "The local solver could not complete this board.";
    showState(modelReady ? "empty" : "loading");
  }
});

solveButton.addEventListener("click", () => {
  const validationError = validateBoard();
  if (validationError) {
    boardError.textContent = validationError;
    return;
  }
  if (populatedRows().some((row) => row.feedback.every((state) => state === 2))) {
    boardError.textContent = "";
    showState("solved");
    return;
  }
  boardError.textContent = "";
  activeRequest += 1;
  solveButton.disabled = true;
  document.querySelector("#progress-bar").style.width = "4%";
  document.querySelector("#progress-copy").textContent = "Filtering candidate answers.";
  showState("working");
  worker.postMessage({ type: "solve", requestId: activeRequest, rows: populatedRows() });
});

resetButton.addEventListener("click", resetBoard);
copyButton.addEventListener("click", async () => {
  if (!recommendedWord) return;
  try {
    await navigator.clipboard.writeText(recommendedWord);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1200);
  } catch {
    boardError.textContent = "Copy access is unavailable; select the recommendation manually.";
  }
});

createBoard();
