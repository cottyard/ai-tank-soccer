import { createNeuralStrategy } from './neuralStrategy';
import type { NeuralWorkerRequest, NeuralWorkerResponse } from './neuralWorkerMessages';

type WorkerContext = {
  onmessage: ((event: MessageEvent<NeuralWorkerRequest>) => void) | null;
  postMessage(message: NeuralWorkerResponse): void;
};

const worker = globalThis as unknown as WorkerContext;

worker.onmessage = (event) => {
  const request = event.data;
  try {
    const strategy = createNeuralStrategy({
      weights: request.weights,
      name: request.name,
      tacticalRollout: request.tacticalRollout
    });
    worker.postMessage({
      id: request.id,
      team: request.team,
      frame: request.state.frame,
      commands: strategy.decide(request.state, request.team)
    });
  } catch (error) {
    worker.postMessage({
      id: request.id,
      team: request.team,
      frame: request.state.frame,
      commands: {},
      error: error instanceof Error ? error.message : String(error)
    });
  }
};
