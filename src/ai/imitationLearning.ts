import { FIELD, type GameState, type Tank, type Team } from '../game/model';
import type { TankCommand } from '../game/strategy';
import { commandToActionIndex } from './policyActions';
import { extractTankInputs } from './neuralStrategy';
import {
  type PolicyWeights,
  trainPolicyBatch
} from './policyNetwork';

export type LearningTag =
  | 'corner'
  | 'sideWall'
  | 'ownDanger'
  | 'finish'
  | 'lowStamina'
  | 'contact'
  | 'contest';

export type LearningSample = {
  inputs: number[];
  actionIndex: number;
  team: Team;
  frame: number;
  tags: LearningTag[];
  weight: number;
};

export type ReplayBufferOptions = {
  maxSamples?: number;
  idleKeepEvery?: number;
};

export type OnlineTrainingOptions = {
  batchSize?: number;
  learningRate?: number;
  seed?: number;
};

export type OfflineTrainingOptions = {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  l2?: number;
  gradientClip?: number;
  seed?: number;
};

export type LearningTrainingResult = {
  weights: number[];
  loss: number;
  trainedSamples: number;
  modelVersion: number;
  epochs?: number;
  batches?: number;
};

export class LearningReplayBuffer {
  private readonly maxSamples: number;
  private readonly idleKeepEvery: number;
  private repeatedIdle = 0;
  private version = 0;
  readonly samples: LearningSample[] = [];

  constructor(options: ReplayBufferOptions = {}) {
    this.maxSamples = options.maxSamples ?? 2000;
    this.idleKeepEvery = options.idleKeepEvery ?? 6;
  }

  add(sample: LearningSample): boolean {
    if (sample.actionIndex === 4) {
      this.repeatedIdle += 1;
      if (this.repeatedIdle > 1 && (this.repeatedIdle - 1) % this.idleKeepEvery !== 0) {
        return false;
      }
    } else {
      this.repeatedIdle = 0;
    }

    this.samples.push(sample);
    while (this.samples.length > this.maxSamples) {
      this.samples.shift();
    }
    return true;
  }

  load(samples: readonly LearningSample[]): void {
    this.samples.length = 0;
    this.repeatedIdle = 0;
    for (const sample of samples.slice(-this.maxSamples)) {
      this.samples.push(cloneSample(sample));
      this.repeatedIdle = sample.actionIndex === 4 ? this.repeatedIdle + 1 : 0;
    }
  }

  markTrained(): number {
    this.version += 1;
    return this.version;
  }

  weightedBatch(size: number, seed = 1): LearningSample[] {
    if (this.samples.length === 0 || size <= 0) {
      return [];
    }

    const random = createSeededRandom(seed + this.version * 101);
    return Array.from({ length: Math.min(size, this.samples.length) }, () =>
      weightedPick(this.samples, random)
    );
  }
}

function cloneSample(sample: LearningSample): LearningSample {
  return {
    inputs: [...sample.inputs],
    actionIndex: sample.actionIndex,
    team: sample.team,
    frame: sample.frame,
    tags: [...sample.tags],
    weight: sample.weight
  };
}

export function createLearningSample(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  command: TankCommand
): LearningSample {
  const tags = tagLearningSample(state, team, tank);
  return {
    inputs: extractTankInputs(state, team, tank),
    actionIndex: commandToActionIndex(command),
    team,
    frame: state.frame,
    tags,
    weight: weightForTags(tags)
  };
}

export function tagLearningSample(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank
): LearningTag[] {
  const tags: LearningTag[] = [];
  const ball = state.ball;
  const sideWallDistance = Math.min(
    ball.position.y - FIELD.ballRadius,
    FIELD.width - FIELD.ballRadius - ball.position.y
  );
  const endWallDistance = Math.min(
    ball.position.x - FIELD.ballRadius,
    FIELD.length - FIELD.ballRadius - ball.position.x
  );
  const sign = team === 'red' ? 1 : -1;
  const ownGoalX = team === 'red' ? 0 : FIELD.length;
  const ownDistance = (ball.position.x - ownGoalX) * sign;
  const lane = Math.abs(ball.position.y - FIELD.width / 2) < FIELD.goalMouth * 0.72;
  const incoming = ball.velocity.x * sign < -50;
  const attackProgress = ((ball.position.x - FIELD.length / 2) * sign) / (FIELD.length / 2);
  const contactDistance = tank.radius + ball.radius + 30;
  const nearOwnEnd = ownDistance < 250;
  const ownCornerThreat = nearOwnEnd && sideWallDistance < FIELD.ballRadius + 72;
  const opponent = nearestOpponent(state, team);
  const ballDistance = Math.hypot(tank.position.x - ball.position.x, tank.position.y - ball.position.y);
  const opponentBallDistance = opponent
    ? Math.hypot(opponent.position.x - ball.position.x, opponent.position.y - ball.position.y)
    : Number.POSITIVE_INFINITY;
  const ballSpeed = Math.hypot(ball.velocity.x, ball.velocity.y);
  const midfieldLooseBall =
    Math.abs(ball.position.x - FIELD.length / 2) < FIELD.length * 0.26 &&
    ballSpeed < 170;
  const opponentPressure = opponentBallDistance < FIELD.tankRadius * 2.6;
  const closeEnoughToContest = ballDistance < FIELD.tankRadius * 4.2;

  if (sideWallDistance < FIELD.ballRadius + 42) {
    tags.push('sideWall');
  }
  if (sideWallDistance < FIELD.ballRadius + 42 && endWallDistance < FIELD.ballRadius + 80) {
    tags.push('corner');
  }
  if (nearOwnEnd && (lane || incoming || ownCornerThreat)) {
    tags.push('ownDanger');
  }
  if (attackProgress > 0.48 && lane) {
    tags.push('finish');
  }
  if (tank.maxStamina > 0 && tank.stamina / tank.maxStamina < 0.5) {
    tags.push('lowStamina');
  }
  if (Math.hypot(tank.position.x - ball.position.x, tank.position.y - ball.position.y) <= contactDistance) {
    tags.push('contact');
  }
  if (closeEnoughToContest && (opponentPressure || midfieldLooseBall || ownCornerThreat)) {
    tags.push('contest');
  }

  return tags;
}

export function trainOnlineFromReplay(
  weights: PolicyWeights,
  replay: LearningReplayBuffer,
  options: OnlineTrainingOptions = {}
): LearningTrainingResult {
  const batch = replay.weightedBatch(options.batchSize ?? 8, options.seed ?? 1);
  if (batch.length === 0) {
    return {
      weights: [...weights],
      loss: 0,
      trainedSamples: 0,
      modelVersion: replay.markTrained()
    };
  }

  const result = trainPolicyBatch(weights, batch, {
    learningRate: options.learningRate ?? 0.035,
    l2: 0.0002,
    gradientClip: 2
  });

  return {
    weights: result.weights,
    loss: result.loss,
    trainedSamples: batch.length,
    modelVersion: replay.markTrained()
  };
}

export function trainReplayBatch(
  weights: PolicyWeights,
  replay: LearningReplayBuffer,
  epochs = 8
): LearningTrainingResult {
  return trainOfflineFromReplay(weights, replay, {
    epochs,
    batchSize: 32,
    learningRate: 0.028,
    l2: 0.00025,
    gradientClip: 1.6,
    seed: 1000
  });
}

export function trainOfflineFromReplay(
  weights: PolicyWeights,
  replay: LearningReplayBuffer,
  options: OfflineTrainingOptions = {}
): LearningTrainingResult {
  let nextWeights = [...weights];
  let loss = 0;
  let trainedSamples = 0;
  let batches = 0;
  const epochs = Math.max(0, Math.floor(options.epochs ?? 80));
  const batchSize = Math.max(1, Math.floor(options.batchSize ?? 64));
  const seed = options.seed ?? 1000;
  const learningRate = options.learningRate ?? 0.026;
  const l2 = options.l2 ?? 0.00022;
  const gradientClip = options.gradientClip ?? 1.8;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const shuffled = shuffledSamples(replay.samples, seed + epoch * 7919);
    if (shuffled.length === 0) {
      break;
    }

    for (let start = 0; start < shuffled.length; start += batchSize) {
      const batch = shuffled.slice(start, start + batchSize);
      const result = trainPolicyBatch(nextWeights, batch, {
        learningRate,
        l2,
        gradientClip
      });
      nextWeights = result.weights;
      loss = result.loss;
      trainedSamples += batch.length;
      batches += 1;
    }
  }

  return {
    weights: nextWeights,
    loss,
    trainedSamples,
    modelVersion: replay.markTrained(),
    epochs,
    batches
  };
}

function shuffledSamples(samples: readonly LearningSample[], seed: number): LearningSample[] {
  const random = createSeededRandom(seed);
  const shuffled = samples.map(cloneSample);

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function weightForTags(tags: readonly LearningTag[]): number {
  let weight = 1;
  if (tags.includes('corner')) {
    weight += 1.4;
  }
  if (tags.includes('contact')) {
    weight += 0.8;
  }
  if (tags.includes('contest')) {
    weight += 0.85;
  }
  if (tags.includes('ownDanger')) {
    weight += 0.7;
  }
  if (tags.includes('finish')) {
    weight += 0.45;
  }
  if (tags.includes('lowStamina')) {
    weight += 0.25;
  }
  return weight;
}

function nearestOpponent(state: Readonly<GameState>, team: Team): Tank | undefined {
  let best: Tank | undefined;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const candidate of state.tanks) {
    if (candidate.team === team) {
      continue;
    }
    const distance = Math.hypot(
      candidate.position.x - state.ball.position.x,
      candidate.position.y - state.ball.position.y
    );
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }
  return best;
}

function weightedPick(samples: readonly LearningSample[], random: () => number): LearningSample {
  const total = samples.reduce((sum, sample) => sum + sample.weight, 0) || samples.length;
  let cursor = random() * total;
  for (const sample of samples) {
    cursor -= sample.weight;
    if (cursor <= 0) {
      return sample;
    }
  }
  return samples[samples.length - 1];
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
