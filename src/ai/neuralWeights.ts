import {
  POLICY_HIDDEN_LAYER_SIZES,
  POLICY_HIDDEN_COUNT,
  POLICY_INPUT_COUNT,
  POLICY_OUTPUT_COUNT,
  POLICY_WEIGHT_COUNT,
  createPolicyWeights
} from './policyNetwork';

export const NEURAL_INPUT_COUNT = POLICY_INPUT_COUNT;
export const NEURAL_HIDDEN_COUNT = POLICY_HIDDEN_COUNT;
export const NEURAL_OUTPUT_COUNT = POLICY_OUTPUT_COUNT;
export const NEURAL_WEIGHT_COUNT = POLICY_WEIGHT_COUNT;

export type NeuralWeights = readonly number[];

export const ZERO_NEURAL_WEIGHTS: NeuralWeights = Object.freeze(
  Array.from({ length: NEURAL_WEIGHT_COUNT }, () => 0)
);

const inputIndexByName = {
  stamina: 7,
  ballDx: 8,
  ballDy: 9,
  ballVelocityX: 10,
  ballDistance: 12,
  ballBearingForward: 13,
  ballBearingLateral: 14,
  ballForwardClose: 15,
  ballLateralClose: 16,
  finishingPressure: 21,
  ownGoalPressure: 22,
  sideWallPressure: 23,
  attackCornerPressure: 25,
  ownCornerPressure: 26,
  nearestOpponentDistance: 29,
  targetForward: 32,
  targetLateral: 33,
  targetBearingForward: 34,
  targetBearingLateral: 35
} as const;

type InputName = keyof typeof inputIndexByName;

export function defaultNeuralWeights(): number[] {
  const weights = createPolicyWeights(101);

  for (let hidden = 0; hidden < 24; hidden += 1) {
    setHiddenBridge(weights, hidden, hidden, 1.35, 0);
  }

  setHidden(weights, 0, {
    targetBearingForward: 2.2,
    targetForward: 0.35,
    stamina: 0.28
  }, 0.05);
  setHidden(weights, 1, {
    targetBearingForward: -2.1,
    targetForward: -0.25
  }, -0.05);
  setHidden(weights, 2, { targetBearingLateral: 2.8 }, 0);
  setHidden(weights, 3, { targetBearingLateral: -2.8 }, 0);
  setHidden(weights, 4, {
    ballBearingForward: 1.2,
    ballForwardClose: 0.75,
    finishingPressure: 0.65
  }, 0);
  setHidden(weights, 5, {
    ballBearingLateral: 1.8,
    ballLateralClose: 0.4
  }, 0);
  setHidden(weights, 6, {
    ballBearingLateral: -1.8,
    ballLateralClose: -0.4
  }, 0);
  setHidden(weights, 7, {
    ownGoalPressure: 2,
    ballVelocityX: -0.65
  }, -0.08);
  setHidden(weights, 8, {
    sideWallPressure: 1.6,
    attackCornerPressure: 1.1,
    targetBearingLateral: 0.75
  }, -0.1);
  setHidden(weights, 9, {
    sideWallPressure: 1.6,
    attackCornerPressure: 1.1,
    targetBearingLateral: -0.75
  }, -0.1);
  setHidden(weights, 10, {
    ownCornerPressure: 1.5,
    targetBearingForward: 0.8
  }, -0.05);
  setHidden(weights, 11, {
    nearestOpponentDistance: -1.4,
    ballDistance: -0.3
  }, 0.12);
  setHidden(weights, 12, {
    stamina: -2,
    ballDistance: 0.75,
    ownGoalPressure: -0.4
  }, 0.95);
  setHidden(weights, 13, {
    targetLateral: 0.8,
    ballDy: 0.35
  }, 0);
  setHidden(weights, 14, {
    targetLateral: -0.8,
    ballDy: -0.35
  }, 0);
  setHidden(weights, 15, {
    ballDx: 0.45,
    finishingPressure: 1
  }, 0);
  setHidden(weights, 16, {
    ballDx: -0.45,
    ownGoalPressure: 1
  }, 0);
  setHidden(weights, 17, {
    targetForward: 1.2,
    ballBearingForward: 0.55
  }, 0);
  setHidden(weights, 18, {
    ballDistance: -1.2,
    finishingPressure: 0.7
  }, 0.15);
  setHidden(weights, 19, {
    ballDistance: -1.2,
    ownGoalPressure: 0.75
  }, 0.15);
  setHidden(weights, 20, {
    targetBearingLateral: 1.5,
    sideWallPressure: 0.35
  }, 0);
  setHidden(weights, 21, {
    targetBearingLateral: -1.5,
    sideWallPressure: 0.35
  }, 0);
  setHidden(weights, 22, {
    ballVelocityX: 0.85,
    finishingPressure: 0.45
  }, 0);
  setHidden(weights, 23, {
    ballVelocityX: -0.85,
    ownGoalPressure: 0.45
  }, 0);

  // Forward action [1, 1].
  setOutput(weights, 8, {
    0: 1.45,
    4: 0.42,
    7: 0.38,
    10: 0.28,
    15: 0.26,
    17: 0.3,
    18: 0.35,
    19: 0.35,
    22: 0.18
  }, 0.08);

  // Pivot and arc actions steer toward the target side.
  setOutput(weights, 7, { 0: 0.75, 2: 1.05, 5: 0.32, 8: 0.26, 20: 0.32 }, 0.02);
  setOutput(weights, 5, { 0: 0.75, 3: 1.05, 6: 0.32, 9: 0.26, 21: 0.32 }, 0.02);
  setOutput(weights, 6, { 2: 0.9, 5: 0.2, 20: 0.35 }, -0.04);
  setOutput(weights, 2, { 3: 0.9, 6: 0.2, 21: 0.35 }, -0.04);

  // Reverse is useful when the target is behind the tank.
  setOutput(weights, 0, { 1: 1.05, 12: 0.18 }, -0.05);
  setOutput(weights, 1, { 1: 0.75, 3: 0.55, 21: 0.18 }, -0.08);
  setOutput(weights, 3, { 1: 0.75, 2: 0.55, 20: 0.18 }, -0.08);

  // Stop becomes more attractive when low stamina is not urgent.
  setOutput(weights, 4, { 12: 1.1 }, -0.12);

  return weights;
}

function setHidden(
  weights: number[],
  hidden: number,
  inputs: Partial<Record<InputName, number>>,
  bias: number
): void {
  const offset = hidden * (NEURAL_INPUT_COUNT + 1);
  for (const [name, value] of Object.entries(inputs) as [InputName, number][]) {
    weights[offset + inputIndexByName[name]] = value;
  }
  weights[offset + NEURAL_INPUT_COUNT] = bias;
}

function setHiddenBridge(
  weights: number[],
  sourceHidden: number,
  targetHidden: number,
  value: number,
  bias: number
): void {
  const offset = firstHiddenWeightCount() + targetHidden * (POLICY_HIDDEN_LAYER_SIZES[0] + 1);
  weights[offset + sourceHidden] = value;
  weights[offset + POLICY_HIDDEN_LAYER_SIZES[0]] = bias;
}

function setOutput(
  weights: number[],
  output: number,
  hidden: Partial<Record<number, number>>,
  bias: number
): void {
  const offset =
    firstHiddenWeightCount() +
    POLICY_HIDDEN_LAYER_SIZES[1] * (POLICY_HIDDEN_LAYER_SIZES[0] + 1) +
    output * (NEURAL_HIDDEN_COUNT + 1);

  for (const [index, value] of Object.entries(hidden) as [string, number][]) {
    weights[offset + Number(index)] = value;
  }
  weights[offset + NEURAL_HIDDEN_COUNT] = bias;
}

function firstHiddenWeightCount(): number {
  return POLICY_HIDDEN_LAYER_SIZES[0] * (NEURAL_INPUT_COUNT + 1);
}
