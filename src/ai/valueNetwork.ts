export const VALUE_INPUT_COUNT = 36;
export const VALUE_HIDDEN_LAYER_SIZES = [32, 32] as const;
const VALUE_LAYER_SIZES = [VALUE_INPUT_COUNT, ...VALUE_HIDDEN_LAYER_SIZES, 1] as const;
export const VALUE_WEIGHT_COUNT = layerWeightCount(VALUE_LAYER_SIZES);

/**
 * State value network.
 *
 * The tactical rollout scores a candidate action by the change in
 * `evaluatePosition`, which is ten hand-weighted linear terms. Measurement on
 * the large-sample benchmark showed that deeper search, wider search, and a
 * reacting opponent model all fail to improve play, which points at that
 * terminal valuation rather than at the search.
 *
 * This network predicts the same quantity the search actually needs: a signed,
 * discounted estimate of who scores next from a state, in units of goals and
 * bounded to [-1, 1] by a tanh head. It consumes the identical 36 team-relative
 * inputs as the policy network, so it stays browser-compatible and needs no new
 * feature plumbing.
 */

export type ValueWeights = readonly number[];

export type ValueSample = {
  inputs: readonly number[];
  target: number;
  weight?: number;
};

export type TrainValueOptions = {
  learningRate?: number;
  l2?: number;
  gradientClip?: number;
  beta1?: number;
  beta2?: number;
  epsilon?: number;
  /** Adam moment state, carried across batches by the caller. */
  moments?: ValueAdamMoments;
};

export type ValueAdamMoments = {
  first: number[];
  second: number[];
  step: number;
};

export type TrainValueResult = {
  weights: number[];
  loss: number;
  moments: ValueAdamMoments;
};

export function createValueWeights(seed = 1): number[] {
  const random = createSeededRandom(seed);
  const weights: number[] = [];
  for (let layer = 0; layer < VALUE_LAYER_SIZES.length - 1; layer += 1) {
    const inputCount = VALUE_LAYER_SIZES[layer];
    const outputCount = VALUE_LAYER_SIZES[layer + 1];
    // Xavier-style scale keeps early tanh activations off the saturated tails.
    const scale = Math.sqrt(6 / (inputCount + outputCount));
    for (let output = 0; output < outputCount; output += 1) {
      for (let input = 0; input < inputCount; input += 1) {
        weights.push((random() * 2 - 1) * scale);
      }
      weights.push(0);
    }
  }
  return weights;
}

export function createValueAdamMoments(): ValueAdamMoments {
  return {
    first: Array.from({ length: VALUE_WEIGHT_COUNT }, () => 0),
    second: Array.from({ length: VALUE_WEIGHT_COUNT }, () => 0),
    step: 0
  };
}

export function evaluateValue(inputs: readonly number[], weights: ValueWeights): number {
  validateInputs(inputs);
  validateWeights(weights);
  return forwardPass(inputs, weights).output;
}

/**
 * One Adam step on mean squared error. Adam is used rather than plain SGD
 * because the input features have very different scales (bounded bearings next
 * to sparse pressure signals) and a single learning rate otherwise either
 * diverges or stalls.
 */
/**
 * Mean squared error and its gradient. Exposed separately so the analytic
 * gradient can be finite-difference checked in tests.
 */
export function valueLossGradients(
  samples: readonly ValueSample[],
  weights: ValueWeights
): { gradients: number[]; loss: number; totalWeight: number } {
  validateWeights(weights);
  const gradients = Array.from({ length: VALUE_WEIGHT_COUNT }, () => 0);
  let loss = 0;
  let totalWeight = 0;

  for (const sample of samples) {
    validateInputs(sample.inputs);
    const sampleWeight = sample.weight ?? 1;
    if (sampleWeight === 0) {
      continue;
    }
    totalWeight += sampleWeight;

    const pass = forwardPass(sample.inputs, weights);
    const error = pass.output - sample.target;
    loss += sampleWeight * error * error;
    accumulateGradients(gradients, pass, weights, 2 * error * sampleWeight);
  }

  if (totalWeight === 0) {
    return { gradients, loss: 0, totalWeight: 0 };
  }

  return {
    gradients: gradients.map((value) => value / totalWeight),
    loss: loss / totalWeight,
    totalWeight
  };
}

export function trainValueBatch(
  samples: readonly ValueSample[],
  weights: ValueWeights,
  options: TrainValueOptions = {}
): TrainValueResult {
  validateWeights(weights);
  const learningRate = options.learningRate ?? 0.002;
  const l2 = options.l2 ?? 0;
  const gradientClip = options.gradientClip ?? 4;
  const beta1 = options.beta1 ?? 0.9;
  const beta2 = options.beta2 ?? 0.999;
  const epsilon = options.epsilon ?? 1e-8;
  const moments = options.moments ?? createValueAdamMoments();

  const { gradients, loss, totalWeight } = valueLossGradients(samples, weights);
  if (totalWeight === 0) {
    return { weights: [...weights], loss: 0, moments };
  }

  const next = [...weights];
  const step = moments.step + 1;
  const biasCorrection1 = 1 - beta1 ** step;
  const biasCorrection2 = 1 - beta2 ** step;

  for (let index = 0; index < VALUE_WEIGHT_COUNT; index += 1) {
    let gradient = gradients[index] + l2 * next[index];
    gradient = Math.max(-gradientClip, Math.min(gradientClip, gradient));
    moments.first[index] = beta1 * moments.first[index] + (1 - beta1) * gradient;
    moments.second[index] = beta2 * moments.second[index] + (1 - beta2) * gradient * gradient;
    const first = moments.first[index] / biasCorrection1;
    const second = moments.second[index] / biasCorrection2;
    next[index] -= learningRate * first / (Math.sqrt(second) + epsilon);
  }

  moments.step = step;
  return { weights: next, loss, moments };
}

type ForwardPass = {
  activations: number[][];
  output: number;
};

function forwardPass(inputs: readonly number[], weights: ValueWeights): ForwardPass {
  const activations: number[][] = [[...inputs]];
  let current = [...inputs];

  for (let layer = 0; layer < VALUE_LAYER_SIZES.length - 1; layer += 1) {
    const inputCount = VALUE_LAYER_SIZES[layer];
    const outputCount = VALUE_LAYER_SIZES[layer + 1];
    const offset = layerOffset(layer);
    const next = Array.from({ length: outputCount }, (_unused, output) => {
      const rowOffset = offset + output * (inputCount + 1);
      let sum = weights[rowOffset + inputCount];
      for (let input = 0; input < inputCount; input += 1) {
        sum += current[input] * weights[rowOffset + input];
      }
      return Math.tanh(sum);
    });
    current = next;
    activations.push(current);
  }

  return { activations, output: current[0] };
}

function accumulateGradients(
  gradients: number[],
  pass: ForwardPass,
  weights: ValueWeights,
  outputGradient: number
): void {
  // Every layer, including the head, uses tanh, so d/dx tanh = 1 - tanh^2.
  let delta = [outputGradient * (1 - pass.output * pass.output)];

  for (let layer = VALUE_LAYER_SIZES.length - 2; layer >= 0; layer -= 1) {
    const inputCount = VALUE_LAYER_SIZES[layer];
    const outputCount = VALUE_LAYER_SIZES[layer + 1];
    const offset = layerOffset(layer);
    const layerInputs = pass.activations[layer];
    const nextDelta = layer > 0 ? Array.from({ length: inputCount }, () => 0) : [];

    for (let output = 0; output < outputCount; output += 1) {
      const rowOffset = offset + output * (inputCount + 1);
      const outputDelta = delta[output];
      if (outputDelta === 0) {
        continue;
      }
      for (let input = 0; input < inputCount; input += 1) {
        gradients[rowOffset + input] += outputDelta * layerInputs[input];
        if (layer > 0) {
          nextDelta[input] += outputDelta * weights[rowOffset + input];
        }
      }
      gradients[rowOffset + inputCount] += outputDelta;
    }

    if (layer > 0) {
      const activation = pass.activations[layer];
      delta = nextDelta.map((value, index) => value * (1 - activation[index] * activation[index]));
    }
  }
}

function validateInputs(inputs: readonly number[]): void {
  if (inputs.length !== VALUE_INPUT_COUNT) {
    throw new Error(`Expected ${VALUE_INPUT_COUNT} value inputs, received ${inputs.length}`);
  }
}

function validateWeights(weights: ValueWeights): void {
  if (weights.length !== VALUE_WEIGHT_COUNT) {
    throw new Error(`Expected ${VALUE_WEIGHT_COUNT} value weights, received ${weights.length}`);
  }
}

function layerOffset(layer: number): number {
  let offset = 0;
  for (let index = 0; index < layer; index += 1) {
    offset += VALUE_LAYER_SIZES[index + 1] * (VALUE_LAYER_SIZES[index] + 1);
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
