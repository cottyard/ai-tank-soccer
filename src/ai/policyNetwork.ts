import { POLICY_ACTION_COUNT } from './policyActions';

export const POLICY_INPUT_COUNT = 36;
export const POLICY_HIDDEN_LAYER_SIZES = [64, 64] as const;
export const POLICY_HIDDEN_COUNT = POLICY_HIDDEN_LAYER_SIZES[0];
export const POLICY_OUTPUT_COUNT = POLICY_ACTION_COUNT;
const POLICY_LAYER_SIZES = [
  POLICY_INPUT_COUNT,
  ...POLICY_HIDDEN_LAYER_SIZES,
  POLICY_OUTPUT_COUNT
] as const;
export const POLICY_WEIGHT_COUNT = layerWeightCount(POLICY_LAYER_SIZES);

export type PolicyWeights = readonly number[];

export type PolicySample = {
  inputs: readonly number[];
  actionIndex: number;
  weight?: number;
};

export type PolicyGradientSample = {
  inputs: readonly number[];
  actionIndex: number;
  advantage: number;
  oldProbability?: number;
};

export type TrainPolicyOptions = {
  learningRate?: number;
  l2?: number;
  gradientClip?: number;
  ppoClip?: number;
};

export type TrainPolicyResult = {
  weights: number[];
  loss: number;
};

export function createPolicyWeights(seed = 1): number[] {
  const random = createSeededRandom(seed);
  return Array.from({ length: POLICY_WEIGHT_COUNT }, () => (random() - 0.5) * 0.12);
}

export function evaluatePolicy(inputs: readonly number[], weights: PolicyWeights): number[] {
  validateInputs(inputs);
  validateWeights(weights);
  return forwardPass(inputs, weights).logits;
}

export function policyProbabilities(logits: readonly number[]): number[] {
  if (logits.length !== POLICY_OUTPUT_COUNT) {
    throw new Error(`Expected ${POLICY_OUTPUT_COUNT} policy logits, received ${logits.length}`);
  }
  return softmax(logits);
}

export function softmax(logits: readonly number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0) || 1;
  return exps.map((value) => value / sum);
}

export function trainPolicyBatch(
  weights: PolicyWeights,
  samples: readonly PolicySample[],
  options: TrainPolicyOptions = {}
): TrainPolicyResult {
  validateWeights(weights);
  if (samples.length === 0) {
    return { weights: [...weights], loss: 0 };
  }

  const learningRate = options.learningRate ?? 0.04;
  const l2 = options.l2 ?? 0.0002;
  const gradientClip = options.gradientClip ?? 1.5;
  const gradient = Array.from({ length: POLICY_WEIGHT_COUNT }, () => 0);
  let totalLoss = 0;
  let totalWeight = 0;

  for (const sample of samples) {
    validateInputs(sample.inputs);
    const sampleWeight = Math.max(0, sample.weight ?? 1);
    if (sampleWeight === 0) {
      continue;
    }

    const target = clampActionIndex(sample.actionIndex);
    const forward = forwardPass(sample.inputs, weights);
    const logits = forward.logits;
    const probabilities = softmax(logits);
    totalLoss += -Math.log(Math.max(1e-9, probabilities[target])) * sampleWeight;
    totalWeight += sampleWeight;

    let deltas = probabilities.map((probability, index) =>
      (probability - (index === target ? 1 : 0)) * sampleWeight
    );

    for (let layer = POLICY_LAYER_SIZES.length - 2; layer >= 0; layer -= 1) {
      const inputActivations = forward.activations[layer];
      const inputCount = POLICY_LAYER_SIZES[layer];
      const outputCount = POLICY_LAYER_SIZES[layer + 1];
      const offset = layerOffset(layer);
      const previousActivationDeltas =
        layer > 0 ? Array.from({ length: inputCount }, () => 0) : [];

      for (let output = 0; output < outputCount; output += 1) {
        const rowOffset = offset + output * (inputCount + 1);
        for (let input = 0; input < inputCount; input += 1) {
          gradient[rowOffset + input] += deltas[output] * inputActivations[input];
          if (layer > 0) {
            previousActivationDeltas[input] += deltas[output] * weights[rowOffset + input];
          }
        }
        gradient[rowOffset + inputCount] += deltas[output];
      }

      if (layer > 0) {
        deltas = previousActivationDeltas.map((delta, index) => {
          const activation = inputActivations[index];
          return delta * (1 - activation * activation);
        });
      }
    }
  }

  const divisor = totalWeight || 1;
  const nextWeights = weights.map((weight, index) => {
    const normalizedGradient = gradient[index] / divisor + l2 * weight;
    const clippedGradient = clamp(normalizedGradient, -gradientClip, gradientClip);
    return clamp(weight - learningRate * clippedGradient, -4, 4);
  });

  return {
    weights: nextWeights,
    loss: totalLoss / divisor
  };
}

export function trainPolicyGradientBatch(
  weights: PolicyWeights,
  samples: readonly PolicyGradientSample[],
  options: TrainPolicyOptions = {}
): TrainPolicyResult {
  validateWeights(weights);
  if (samples.length === 0) {
    return { weights: [...weights], loss: 0 };
  }

  const learningRate = options.learningRate ?? 0.01;
  const l2 = options.l2 ?? 0.0001;
  const gradientClip = options.gradientClip ?? 1.5;
  const ppoClip = Math.max(0, options.ppoClip ?? 0);
  const gradient = Array.from({ length: POLICY_WEIGHT_COUNT }, () => 0);
  let totalLoss = 0;
  let totalWeight = 0;

  for (const sample of samples) {
    validateInputs(sample.inputs);
    if (!Number.isFinite(sample.advantage) || sample.advantage === 0) {
      continue;
    }

    const target = clampActionIndex(sample.actionIndex);
    const forward = forwardPass(sample.inputs, weights);
    const probabilities = softmax(forward.logits);
    const advantage = sample.advantage;
    const sampleWeight = Math.abs(advantage);
    const oldProbability = sample.oldProbability && sample.oldProbability > 0
      ? sample.oldProbability
      : undefined;
    const ratio = oldProbability
      ? probabilities[target] / oldProbability
      : 1;
    if (ppoClip > 0 && oldProbability && clipsPpoUpdate(ratio, advantage, ppoClip)) {
      totalWeight += sampleWeight;
      continue;
    }
    totalLoss += -advantage * Math.log(Math.max(1e-9, probabilities[target]));
    totalWeight += sampleWeight;

    let deltas = probabilities.map((probability, index) =>
      (probability - (index === target ? 1 : 0)) * advantage
    );

    for (let layer = POLICY_LAYER_SIZES.length - 2; layer >= 0; layer -= 1) {
      const inputActivations = forward.activations[layer];
      const inputCount = POLICY_LAYER_SIZES[layer];
      const outputCount = POLICY_LAYER_SIZES[layer + 1];
      const offset = layerOffset(layer);
      const previousActivationDeltas =
        layer > 0 ? Array.from({ length: inputCount }, () => 0) : [];

      for (let output = 0; output < outputCount; output += 1) {
        const rowOffset = offset + output * (inputCount + 1);
        for (let input = 0; input < inputCount; input += 1) {
          gradient[rowOffset + input] += deltas[output] * inputActivations[input];
          if (layer > 0) {
            previousActivationDeltas[input] += deltas[output] * weights[rowOffset + input];
          }
        }
        gradient[rowOffset + inputCount] += deltas[output];
      }

      if (layer > 0) {
        deltas = previousActivationDeltas.map((delta, index) => {
          const activation = inputActivations[index];
          return delta * (1 - activation * activation);
        });
      }
    }
  }

  const divisor = totalWeight || 1;
  const nextWeights = weights.map((weight, index) => {
    const normalizedGradient = gradient[index] / divisor + l2 * weight;
    const clippedGradient = clamp(normalizedGradient, -gradientClip, gradientClip);
    return clamp(weight - learningRate * clippedGradient, -4, 4);
  });

  return {
    weights: nextWeights,
    loss: totalLoss / divisor
  };
}

function forwardPass(inputs: readonly number[], weights: PolicyWeights): {
  activations: number[][];
  logits: number[];
} {
  const activations: number[][] = [[...inputs]];
  let current = [...inputs];

  for (let layer = 0; layer < POLICY_LAYER_SIZES.length - 1; layer += 1) {
    const inputCount = POLICY_LAYER_SIZES[layer];
    const outputCount = POLICY_LAYER_SIZES[layer + 1];
    const outputIsLogits = layer === POLICY_LAYER_SIZES.length - 2;
    const offset = layerOffset(layer);
    const next = Array.from({ length: outputCount }, (_, output) => {
      const rowOffset = offset + output * (inputCount + 1);
      let sum = weights[rowOffset + inputCount];
      for (let input = 0; input < inputCount; input += 1) {
        sum += current[input] * weights[rowOffset + input];
      }
      return outputIsLogits ? sum : Math.tanh(sum);
    });

    if (outputIsLogits) {
      return { activations, logits: next };
    }

    current = next;
    activations.push(current);
  }

  return { activations, logits: [] };
}

function validateInputs(inputs: readonly number[]): void {
  if (inputs.length !== POLICY_INPUT_COUNT) {
    throw new Error(`Expected ${POLICY_INPUT_COUNT} policy inputs, received ${inputs.length}`);
  }
}

function validateWeights(weights: PolicyWeights): void {
  if (weights.length !== POLICY_WEIGHT_COUNT) {
    throw new Error(`Expected ${POLICY_WEIGHT_COUNT} policy weights, received ${weights.length}`);
  }
}

function layerOffset(layer: number): number {
  let offset = 0;
  for (let index = 0; index < layer; index += 1) {
    offset += POLICY_LAYER_SIZES[index + 1] * (POLICY_LAYER_SIZES[index] + 1);
  }
  return offset;
}

function layerWeightCount(sizes: readonly number[]): number {
  let count = 0;
  for (let index = 0; index < sizes.length - 1; index += 1) {
    count += sizes[index + 1] * (sizes[index] + 1);
  }
  return count;
}

function clampActionIndex(index: number): number {
  if (!Number.isFinite(index)) {
    return 4;
  }
  return Math.max(0, Math.min(POLICY_OUTPUT_COUNT - 1, Math.round(index)));
}

function clipsPpoUpdate(ratio: number, advantage: number, ppoClip: number): boolean {
  if (!Number.isFinite(ratio)) {
    return true;
  }
  if (advantage > 0) {
    return ratio > 1 + ppoClip;
  }
  if (advantage < 0) {
    return ratio < 1 - ppoClip;
  }
  return true;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
