import type { GameState, Team } from '../game/model';
import type { CommandMap, Strategy } from '../game/strategy';
import { defaultNeuralWeights, type NeuralWeights } from './neuralWeights';
import type { NeuralWorkerRequest, NeuralWorkerResponse } from './neuralWorkerMessages';

export type AsyncNeuralStrategyOptions = {
  weights?: NeuralWeights | (() => NeuralWeights);
  name?: string;
  tacticalRollout?: boolean;
  onError?: (message: string) => void;
};

export type AsyncNeuralStrategy = Strategy & {
  dispose(): void;
};

export function createAsyncNeuralStrategy(options: AsyncNeuralStrategyOptions = {}): AsyncNeuralStrategy {
  const name = options.name ?? 'neural-policy';
  const tacticalRollout = options.tacticalRollout ?? true;
  const resolveWeights = weightResolver(options.weights);
  let worker: Worker | undefined;
  let nextRequestId = 0;
  let inFlight = false;
  let disposed = false;
  let latestCommands: CommandMap = {};

  function ensureWorker(): Worker {
    if (!worker) {
      worker = new Worker(new URL('./neuralWorker.ts', import.meta.url), {
        type: 'module'
      });
      worker.onmessage = (event: MessageEvent<NeuralWorkerResponse>) => {
        const response = event.data;
        if (disposed || response.id !== nextRequestId) {
          return;
        }

        inFlight = false;
        if (response.error) {
          options.onError?.(response.error);
          return;
        }

        latestCommands = response.commands;
      };
      worker.onerror = (event) => {
        inFlight = false;
        worker?.terminate();
        worker = undefined;
        options.onError?.(event.message);
      };
    }

    return worker;
  }

  return {
    name,
    decide(state: Readonly<GameState>, team: Team): CommandMap {
      if (!disposed && !inFlight) {
        inFlight = true;
        const request: NeuralWorkerRequest = {
          id: nextRequestId + 1,
          state: state as GameState,
          team,
          weights: [...resolveWeights()],
          name,
          tacticalRollout
        };
        nextRequestId = request.id;
        try {
          ensureWorker().postMessage(request);
        } catch (error) {
          inFlight = false;
          options.onError?.(error instanceof Error ? error.message : String(error));
        }
      }

      return latestCommands;
    },
    dispose(): void {
      disposed = true;
      inFlight = false;
      worker?.terminate();
      worker = undefined;
      latestCommands = {};
    }
  };
}

function weightResolver(weights: AsyncNeuralStrategyOptions['weights']): () => NeuralWeights {
  if (typeof weights === 'function') {
    return weights;
  }

  if (weights) {
    return () => weights;
  }

  return defaultNeuralWeights;
}
