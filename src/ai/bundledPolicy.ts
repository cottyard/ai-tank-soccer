import { NEURAL_WEIGHT_COUNT, type NeuralWeights } from './neuralWeights';

export const BUNDLED_POLICY_URL = '/models/neural-best.json';
export const BUNDLED_POLICY_VERSION = 'native-shuffle-e18-20260501';

export type BundledPolicyPayload = {
  weights: number[];
  metadata?: {
    cycle?: number;
    bestCycle?: number;
    selectionScore?: number;
    seed?: number;
    replaySamples?: number;
    selfPlaySamples?: number;
    loss?: number;
  };
};

export async function loadBundledPolicy(fetcher: typeof fetch = fetch): Promise<BundledPolicyPayload | null> {
  const response = await fetcher(BUNDLED_POLICY_URL);
  if (!response.ok) {
    return null;
  }

  const parsed = await response.json() as unknown;
  if (!isRecord(parsed) || !Array.isArray(parsed.weights)) {
    return null;
  }
  if (!isNeuralWeights(parsed.weights)) {
    return null;
  }

  return {
    weights: parsed.weights,
    metadata: isRecord(parsed.metadata) ? parsed.metadata : undefined
  };
}

function isNeuralWeights(weights: readonly unknown[]): weights is NeuralWeights {
  return weights.length === NEURAL_WEIGHT_COUNT &&
    weights.every((weight) => typeof weight === 'number' && Number.isFinite(weight));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}
