import { prepareModel, solveBoard } from "./solver-core.js";

let model;

async function loadModel() {
  const response = await fetch("./model.json", { cache: "no-cache" });
  if (!response.ok) throw new Error(`Model request failed with status ${response.status}.`);
  model = prepareModel(await response.json());
  const opening = await solveBoard(model, []);
  self.postMessage({
    type: "ready",
    metadata: {
      actionCount: model.actionWords.length,
      answerCount: model.answerActionIndices.length,
      trainedThrough: model.trainedThrough,
      predictionDate: model.predictionDate,
      starter: model.policy.starter,
      starterEntropy: opening.entropy,
      starterAnswerProbability: opening.answerProbability,
    },
  });
}

loadModel().catch((error) => {
  self.postMessage({ type: "error", message: error.message });
});

self.addEventListener("message", async (event) => {
  if (event.data?.type !== "solve") return;
  const { requestId, rows } = event.data;
  if (!model) {
    self.postMessage({ type: "error", requestId, message: "The model is still loading." });
    return;
  }
  try {
    const result = await solveBoard(model, rows, (progress) => {
      self.postMessage({ type: "progress", requestId, progress });
    });
    self.postMessage({ type: "result", requestId, result });
  } catch (error) {
    self.postMessage({ type: "error", requestId, message: error.message });
  }
});
