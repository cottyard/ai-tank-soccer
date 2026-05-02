import { stepGame } from '../game/simulation';
import { AI_HZ, FIXED_DT, PHYSICS_HZ } from '../game/match';
import { FIELD, cloneState, createInitialState, type GameState, type Team } from '../game/model';
import type { CommandMap } from '../game/strategy';
import { actionIndexToCommand, POLICY_ACTION_COUNT } from './policyActions';
import { extractTankInputs } from './neuralStrategy';
import {
  evaluatePolicy,
  policyProbabilities,
  trainPolicyGradientBatch,
  type PolicyGradientSample,
  type PolicyWeights
} from './policyNetwork';

export type PolicyGradientDecision = PolicyGradientSample & {
  team: Team;
  frame: number;
  probability: number;
  logProbability: number;
  return: number;
  trainable: boolean;
  startStateMode: ActualStartStateMode;
};

export type PolicyGradientStartStateMode = ActualStartStateMode | 'mixed';
export type PolicyGradientAdvantageBaseline = 'global' | 'start-team-time';
type ActualStartStateMode = 'open' | 'outcome-curriculum';

export type PolicyGradientCollectionOptions = {
  weights: PolicyWeights;
  opponentWeights?: PolicyWeights;
  matches?: number;
  frames?: number;
  seed?: number;
  temperature?: number;
  discount?: number;
  goalReward?: number;
  winReward?: number;
  normalizeAdvantages?: boolean;
  advantageBaseline?: PolicyGradientAdvantageBaseline;
  startStateMode?: PolicyGradientStartStateMode;
  initialStateFactory?: (match: number, random: () => number) => GameState;
};

export type PolicyGradientCollectionResult = {
  decisions: PolicyGradientDecision[];
  samples: PolicyGradientSample[];
  frames: number;
  redGoals: number;
  blueGoals: number;
  finalState: GameState;
};

export type PolicyGradientTrainingOptions = PolicyGradientCollectionOptions & {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
  l2?: number;
  gradientClip?: number;
  ppoClip?: number;
};

export type PolicyGradientTrainingResult = {
  weights: number[];
  loss: number;
  trainedSamples: number;
  samples: number;
  frames: number;
  redGoals: number;
  blueGoals: number;
  finalState: GameState;
};

type PendingDecision = Omit<PolicyGradientDecision, 'advantage' | 'return'> & {
  rewards: number[];
};

type GoalSnapshot = {
  frame: number;
  team: Team;
};

const DEFAULT_TEMPERATURE = 1.08;
const DEFAULT_DISCOUNT = 0.992;
const DEFAULT_GOAL_REWARD = 1;
const DEFAULT_WIN_REWARD = 1.4;

export function collectPolicyGradientSelfPlay(
  options: PolicyGradientCollectionOptions
): PolicyGradientCollectionResult {
  const matches = Math.max(1, Math.floor(options.matches ?? 8));
  const framesPerMatch = Math.max(1, Math.floor(options.frames ?? PHYSICS_HZ * 45));
  const framesPerDecision = Math.max(1, Math.round(PHYSICS_HZ / AI_HZ));
  const random = createSeededRandom(options.seed ?? 1);
  const opponentWeights = options.opponentWeights ?? options.weights;
  const temperature = Math.max(0.05, options.temperature ?? DEFAULT_TEMPERATURE);
  const discount = clamp01(options.discount ?? DEFAULT_DISCOUNT);
  const goalReward = options.goalReward ?? DEFAULT_GOAL_REWARD;
  const winReward = options.winReward ?? DEFAULT_WIN_REWARD;
  const normalizeAdvantages = options.normalizeAdvantages ?? true;
  const advantageBaseline = options.advantageBaseline ?? 'global';
  const decisions: PolicyGradientDecision[] = [];
  let redGoals = 0;
  let blueGoals = 0;
  let completedFrames = 0;
  let finalState = createInitialState();

  for (let match = 0; match < matches; match += 1) {
    const startStateMode = resolveStartStateMode(options.startStateMode ?? 'open', match);
    const state = options.initialStateFactory
      ? options.initialStateFactory(match, random)
      : createSeededInitialState(random, match, startStateMode);
    const pending: PendingDecision[] = [];
    const goals: GoalSnapshot[] = [];
    let commands: CommandMap = {};

    for (let frame = 0; frame < framesPerMatch; frame += 1) {
      if (state.frame % framesPerDecision === 0) {
        const redDecision = sampleTeamDecision(state, 'red', options.weights, temperature, random, startStateMode);
        const blueDecision = sampleTeamDecision(state, 'blue', opponentWeights, temperature, random, startStateMode);
        commands = {
          ...redDecision.commands,
          ...blueDecision.commands
        };
        if (redDecision.decision) {
          pending.push({
            ...redDecision.decision,
            trainable: true
          });
        }
        if (blueDecision.decision) {
          pending.push({
            ...blueDecision.decision,
            trainable: opponentWeights === options.weights
          });
        }
      }

      stepGame(state, commands, FIXED_DT);
      if (state.lastGoal?.frame === state.frame - 1) {
        goals.push({
          frame: state.lastGoal.frame,
          team: state.lastGoal.team
        });
      }
    }

    const redDiff = state.score.red - state.score.blue;
    const matchDecisions = finalizeMatchDecisions(
      pending,
      goals,
      redDiff,
      discount,
      goalReward,
      winReward
    );
    decisions.push(...matchDecisions);
    redGoals += state.score.red;
    blueGoals += state.score.blue;
    completedFrames += framesPerMatch;
    finalState = cloneState(state);
  }

  const normalizedDecisions = normalizeAdvantages
    ? withNormalizedAdvantages(decisions, advantageBaseline)
    : decisions;

  return {
    decisions: normalizedDecisions,
    samples: normalizedDecisions
      .filter((decision) => decision.trainable)
      .map((decision) => ({
        inputs: decision.inputs,
        actionIndex: decision.actionIndex,
        advantage: decision.advantage,
        oldProbability: decision.probability
      })),
    frames: completedFrames,
    redGoals,
    blueGoals,
    finalState
  };
}

export function trainPolicyGradientSelfPlay(
  options: PolicyGradientTrainingOptions
): PolicyGradientTrainingResult {
  const collection = collectPolicyGradientSelfPlay(options);
  const epochs = Math.max(0, Math.floor(options.epochs ?? 4));
  const batchSize = Math.max(1, Math.floor(options.batchSize ?? 64));
  const random = createSeededRandom((options.seed ?? 1) + 90_017);
  let weights = [...options.weights];
  let loss = 0;
  let trainedSamples = 0;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    const shuffled = shuffle(collection.samples, random);
    for (let index = 0; index < shuffled.length; index += batchSize) {
      const batch = shuffled.slice(index, index + batchSize);
      const trained = trainPolicyGradientBatch(weights, batch, {
        learningRate: options.learningRate ?? 0.006,
        l2: options.l2 ?? 0.00008,
        gradientClip: options.gradientClip ?? 1.2,
        ppoClip: options.ppoClip ?? 0.2
      });
      weights = trained.weights;
      loss = trained.loss;
      trainedSamples += batch.length;
    }
  }

  return {
    weights,
    loss,
    trainedSamples,
    samples: collection.samples.length,
    frames: collection.frames,
    redGoals: collection.redGoals,
    blueGoals: collection.blueGoals,
    finalState: collection.finalState
  };
}

function sampleTeamDecision(
  state: Readonly<GameState>,
  team: Team,
  weights: PolicyWeights,
  temperature: number,
  random: () => number,
  startStateMode: ActualStartStateMode
): { commands: CommandMap; decision?: PendingDecision } {
  const tank = state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  if (!tank) {
    return { commands: {} };
  }

  const inputs = extractTankInputs(state, team, tank);
  const logits = evaluatePolicy(inputs, weights).map((logit) => logit / temperature);
  const probabilities = policyProbabilities(logits);
  const actionIndex = sampleAction(probabilities, random);
  const probability = Math.max(1e-9, probabilities[actionIndex]);

  return {
    commands: {
      [tank.id]: actionIndexToCommand(actionIndex)
    },
    decision: {
      inputs,
      actionIndex,
      team,
      frame: state.frame,
      probability,
      logProbability: Math.log(probability),
      trainable: true,
      startStateMode,
      rewards: []
    }
  };
}

function finalizeMatchDecisions(
  decisions: readonly PendingDecision[],
  goals: readonly GoalSnapshot[],
  redDiff: number,
  discount: number,
  goalReward: number,
  winReward: number
): PolicyGradientDecision[] {
  return decisions.map((decision) => {
    let totalReturn = 0;
    for (const goal of goals) {
      if (goal.frame < decision.frame) {
        continue;
      }
      const sign = goal.team === decision.team ? 1 : -1;
      const ticksAhead = Math.max(0, goal.frame - decision.frame) / Math.max(1, PHYSICS_HZ / AI_HZ);
      totalReturn += sign * goalReward * discount ** ticksAhead;
    }

    const teamDiff = decision.team === 'red' ? redDiff : -redDiff;
    totalReturn += Math.sign(teamDiff) * winReward;

    return {
      inputs: decision.inputs,
      actionIndex: decision.actionIndex,
      team: decision.team,
      frame: decision.frame,
      probability: decision.probability,
      logProbability: decision.logProbability,
      return: totalReturn,
      advantage: totalReturn,
      trainable: decision.trainable,
      startStateMode: decision.startStateMode
    };
  });
}

function withNormalizedAdvantages(
  decisions: readonly PolicyGradientDecision[],
  baseline: PolicyGradientAdvantageBaseline
): PolicyGradientDecision[] {
  if (decisions.length === 0) {
    return [];
  }

  const population = decisions.filter((decision) => decision.trainable);
  const globalStats = advantageStats(population.length > 0 ? population : decisions);
  const groupedStats = baseline === 'start-team-time'
    ? buildGroupedAdvantageStats(population.length > 0 ? population : decisions)
    : new Map<string, AdvantageStats>();

  return decisions.map((decision) => ({
    ...decision,
    advantage: normalizeReturn(
      decision.return,
      groupedStats.get(advantageGroupKey(decision)) ?? globalStats
    )
  }));
}

type AdvantageStats = {
  count: number;
  mean: number;
  std: number;
};

function advantageStats(decisions: readonly PolicyGradientDecision[]): AdvantageStats {
  if (decisions.length === 0) {
    return { count: 0, mean: 0, std: 0 };
  }

  const mean = decisions.reduce((sum, decision) => sum + decision.return, 0) / decisions.length;
  const variance = decisions.reduce(
    (sum, decision) => sum + (decision.return - mean) ** 2,
    0
  ) / decisions.length;
  return {
    count: decisions.length,
    mean,
    std: Math.sqrt(variance)
  };
}

function buildGroupedAdvantageStats(
  decisions: readonly PolicyGradientDecision[]
): Map<string, AdvantageStats> {
  const grouped = new Map<string, PolicyGradientDecision[]>();
  for (const decision of decisions) {
    const key = advantageGroupKey(decision);
    grouped.set(key, [...(grouped.get(key) ?? []), decision]);
  }

  const stats = new Map<string, AdvantageStats>();
  for (const [key, values] of grouped.entries()) {
    const groupStats = advantageStats(values);
    if (groupStats.count >= 2) {
      stats.set(key, groupStats);
    }
  }
  return stats;
}

function advantageGroupKey(decision: Pick<PolicyGradientDecision, 'startStateMode' | 'team' | 'frame'>): string {
  const timeBucket = Math.floor(decision.frame / (PHYSICS_HZ * 5));
  return `${decision.startStateMode}:${decision.team}:${timeBucket}`;
}

function normalizeReturn(value: number, stats: AdvantageStats): number {
  if (stats.std < 1e-6) {
    return value - stats.mean;
  }
  return (value - stats.mean) / stats.std;
}

function sampleAction(probabilities: readonly number[], random: () => number): number {
  let cursor = random();
  for (let index = 0; index < probabilities.length; index += 1) {
    cursor -= probabilities[index];
    if (cursor <= 0) {
      return index;
    }
  }
  return POLICY_ACTION_COUNT - 1;
}

function createSeededInitialState(
  random: () => number,
  match: number,
  mode: ActualStartStateMode
): GameState {
  const state = createInitialState();
  const yJitter = (random() - 0.5) * FIELD.width * 0.32;
  const xJitter = (random() - 0.5) * FIELD.length * 0.18;

  state.ball.position = {
    x: FIELD.length / 2 + xJitter,
    y: FIELD.width / 2 + yJitter
  };
  state.ball.velocity = {
    x: (random() - 0.5) * 120,
    y: (random() - 0.5) * 120
  };

  const red = state.tanks.find((tank) => tank.team === 'red' && tank.index === 0);
  const blue = state.tanks.find((tank) => tank.team === 'blue' && tank.index === 0);
  if (red) {
    red.position = {
      x: 160 + random() * 110,
      y: FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.18
    };
    red.angle = (random() - 0.5) * 0.4;
  }
  if (blue) {
    blue.position = {
      x: FIELD.length - 160 - random() * 110,
      y: FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.18
    };
    blue.angle = Math.PI + (random() - 0.5) * 0.4;
  }

  if (mode === 'outcome-curriculum') {
    placeOutcomeCurriculumState(state, match, random);
  }

  return state;
}

function resolveStartStateMode(mode: PolicyGradientStartStateMode, match: number): ActualStartStateMode {
  if (mode === 'mixed') {
    return match % 2 === 0 ? 'open' : 'outcome-curriculum';
  }
  return mode;
}

function placeOutcomeCurriculumState(
  state: GameState,
  match: number,
  random: () => number
): void {
  const attackingTeam: Team = match % 2 === 0 ? 'red' : 'blue';
  const scenario = match % 4;

  if (scenario === 0) {
    placeBallInTeamFrame(
      state,
      attackingTeam,
      FIELD.length - FIELD.ballRadius - 3,
      FIELD.width / 2 + (random() - 0.5) * FIELD.goalMouth * 0.45,
      190 + random() * 70,
      (random() - 0.5) * 20
    );
  } else if (scenario === 1) {
    placeBallInTeamFrame(
      state,
      attackingTeam === 'red' ? 'blue' : 'red',
      FIELD.length - FIELD.ballRadius - 3,
      FIELD.width / 2 + (random() - 0.5) * FIELD.goalMouth * 0.45,
      190 + random() * 70,
      (random() - 0.5) * 20
    );
  } else if (scenario === 2) {
    const side = random() < 0.5 ? -1 : 1;
    placeBallInTeamFrame(
      state,
      attackingTeam,
      FIELD.length - FIELD.ballRadius - 28 - random() * 24,
      side < 0
        ? FIELD.ballRadius + 8 + random() * 18
        : FIELD.width - FIELD.ballRadius - 8 - random() * 18,
      40 + random() * 60,
      side * (random() * 40)
    );
  } else {
    placeBallInTeamFrame(
      state,
      attackingTeam,
      FIELD.length / 2 + random() * 80,
      FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.25,
      80 + random() * 80,
      (random() - 0.5) * 60
    );
  }

  const red = state.tanks.find((tank) => tank.team === 'red' && tank.index === 0);
  const blue = state.tanks.find((tank) => tank.team === 'blue' && tank.index === 0);
  if (red) {
    red.position = fieldPoint('red', 210 + random() * 70, FIELD.width / 2 + (random() - 0.5) * 140);
    red.angle = fieldAngle('red', (random() - 0.5) * 0.5);
    red.velocity = { x: 0, y: 0 };
    red.angularVelocity = 0;
  }
  if (blue) {
    blue.position = fieldPoint('blue', 210 + random() * 70, FIELD.width / 2 + (random() - 0.5) * 140);
    blue.angle = fieldAngle('blue', (random() - 0.5) * 0.5);
    blue.velocity = { x: 0, y: 0 };
    blue.angularVelocity = 0;
  }
}

function placeBallInTeamFrame(
  state: GameState,
  team: Team,
  attackFrameX: number,
  attackFrameY: number,
  attackFrameVelocityX: number,
  attackFrameVelocityY: number
): void {
  state.ball.position = fieldPoint(team, attackFrameX, attackFrameY);
  state.ball.velocity = fieldVector(team, attackFrameVelocityX, attackFrameVelocityY);
}

function fieldPoint(team: Team, attackFrameX: number, attackFrameY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackFrameX : FIELD.length - attackFrameX,
    y: team === 'red' ? attackFrameY : FIELD.width - attackFrameY
  };
}

function fieldVector(team: Team, attackFrameX: number, attackFrameY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackFrameX : -attackFrameX,
    y: team === 'red' ? attackFrameY : -attackFrameY
  };
}

function fieldAngle(team: Team, attackFrameAngle: number): number {
  return normalizeAngle(team === 'red' ? attackFrameAngle : attackFrameAngle + Math.PI);
}

function shuffle<T>(items: readonly T[], random: () => number): T[] {
  const result = [...items];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
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

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function normalizeAngle(angle: number): number {
  let normalized = angle;
  while (normalized <= -Math.PI) {
    normalized += Math.PI * 2;
  }
  while (normalized > Math.PI) {
    normalized -= Math.PI * 2;
  }
  return normalized;
}
