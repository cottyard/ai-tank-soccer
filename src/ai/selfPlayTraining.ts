import { stepGame } from '../game/simulation';
import { AI_HZ, FIXED_DT, PHYSICS_HZ } from '../game/match';
import { FIELD, cloneState, createInitialState, type GameState, type Tank, type Team } from '../game/model';
import type { CommandMap } from '../game/strategy';
import { actionIndexToCommand, POLICY_ACTION_COUNT } from './policyActions';
import { tagLearningSample, LearningReplayBuffer, trainOfflineFromReplay, type LearningSample } from './imitationLearning';
import { evaluatePolicy, policyProbabilities, type PolicyWeights } from './policyNetwork';
import { extractTankInputs } from './neuralStrategy';
import { evaluatePosition, evaluatePositionDelta } from './positionEvaluation';

export type SelfPlayCollectionOptions = {
  weights: PolicyWeights;
  opponentWeights?: PolicyWeights;
  matches?: number;
  frames?: number;
  seed?: number;
  exploration?: number;
  rolloutFrames?: number;
};

export type SelfPlayCollectionResult = {
  samples: LearningSample[];
  frames: number;
  redGoals: number;
  blueGoals: number;
  finalState: GameState;
};

export type SelfPlayTrainingOptions = SelfPlayCollectionOptions & {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
};

export type SelfPlayTrainingResult = {
  weights: number[];
  loss: number;
  trainedSamples: number;
  modelVersion: number;
  samples: number;
  frames: number;
  redGoals: number;
  blueGoals: number;
  finalState: GameState;
};

type PendingSample = LearningSample & {
  startBallX: number;
  rolloutScore: number;
};

const DEFAULT_EXPLORATION = 0.18;
const DEFAULT_ROLLOUT_FRAMES = 18;
const LOW_STAMINA_CURRICULUM_RATIO = 0.34;
const CRITICAL_STAMINA_RATIO = 0.22;

export function collectSelfPlaySamples(options: SelfPlayCollectionOptions): SelfPlayCollectionResult {
  const matches = Math.max(1, Math.floor(options.matches ?? 8));
  const framesPerMatch = Math.max(1, Math.floor(options.frames ?? PHYSICS_HZ * 45));
  const framesPerDecision = Math.max(1, Math.round(PHYSICS_HZ / AI_HZ));
  const random = createSeededRandom(options.seed ?? 1);
  const opponentWeights = options.opponentWeights ?? options.weights;
  const exploration = clamp01(options.exploration ?? DEFAULT_EXPLORATION);
  const rolloutFrames = Math.max(1, Math.floor(options.rolloutFrames ?? DEFAULT_ROLLOUT_FRAMES));
  const samples: LearningSample[] = [];
  let redGoals = 0;
  let blueGoals = 0;
  let completedFrames = 0;
  let finalState = createInitialState();

  for (let match = 0; match < matches; match += 1) {
    const state = createSeededInitialState(random, match);
    const pending: PendingSample[] = [];
    let commands: CommandMap = {};

    for (let frame = 0; frame < framesPerMatch; frame += 1) {
      if (state.frame % framesPerDecision === 0) {
        const redDecisions = decideSelfPlayTeam(
          state,
          'red',
          options.weights,
          opponentWeights,
          exploration,
          rolloutFrames,
          random
        );
        const blueDecisions = decideSelfPlayTeam(
          state,
          'blue',
          opponentWeights,
          options.weights,
          exploration,
          rolloutFrames,
          random
        );
        commands = {
          ...redDecisions.commands,
          ...blueDecisions.commands
        };
        pending.push(...redDecisions.samples, ...blueDecisions.samples);
      }

      stepGame(state, commands, FIXED_DT);
    }

    const redDiff = state.score.red - state.score.blue;
    for (const sample of pending) {
      samples.push(weightSelfPlaySample(sample, state, redDiff));
    }

    redGoals += state.score.red;
    blueGoals += state.score.blue;
    completedFrames += framesPerMatch;
    finalState = cloneState(state);
  }

  return {
    samples,
    frames: completedFrames,
    redGoals,
    blueGoals,
    finalState
  };
}

export function trainSelfPlayPolicy(options: SelfPlayTrainingOptions): SelfPlayTrainingResult {
  const collection = collectSelfPlaySamples(options);
  const replay = new LearningReplayBuffer({ maxSamples: collection.samples.length || 1 });
  replay.load(collection.samples);
  const trained = trainOfflineFromReplay(options.weights, replay, {
    epochs: options.epochs ?? 36,
    batchSize: options.batchSize ?? 64,
    learningRate: options.learningRate ?? 0.018,
    l2: 0.00028,
    gradientClip: 1.4,
    seed: options.seed ?? 1
  });

  return {
    weights: trained.weights,
    loss: trained.loss,
    trainedSamples: trained.trainedSamples,
    modelVersion: trained.modelVersion,
    samples: collection.samples.length,
    frames: collection.frames,
    redGoals: collection.redGoals,
    blueGoals: collection.blueGoals,
    finalState: collection.finalState
  };
}

function decideSelfPlayTeam(
  state: Readonly<GameState>,
  team: Team,
  weights: PolicyWeights,
  opponentWeights: PolicyWeights,
  exploration: number,
  rolloutFrames: number,
  random: () => number
): { commands: CommandMap; samples: PendingSample[] } {
  const tank = state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  if (!tank) {
    return { commands: {}, samples: [] };
  }

  const inputs = extractTankInputs(state, team, tank);
  const improved = chooseRolloutAction(state, team, weights, opponentWeights, rolloutFrames);
  const executedActionIndex =
    random() < exploration
      ? Math.floor(random() * POLICY_ACTION_COUNT)
      : improved.actionIndex;
  const command = actionIndexToCommand(executedActionIndex);
  const tags = tagLearningSample(state, team, tank);
  const sample: PendingSample = {
    inputs,
    actionIndex: improved.actionIndex,
    team,
    frame: state.frame,
    tags,
    weight: 1,
    startBallX: state.ball.position.x,
    rolloutScore: improved.score
  };

  return {
    commands: {
      [tank.id]: command
    },
    samples: [sample]
  };
}

function chooseRolloutAction(
  state: Readonly<GameState>,
  team: Team,
  weights: PolicyWeights,
  opponentWeights: PolicyWeights,
  rolloutFrames: number
): { actionIndex: number; score: number } {
  let best = {
    actionIndex: greedyPolicyAction(extractTankInputsForTeam(state, team), weights),
    score: Number.NEGATIVE_INFINITY
  };

  for (let actionIndex = 0; actionIndex < POLICY_ACTION_COUNT; actionIndex += 1) {
    const score = scoreActionRollout(state, team, actionIndex, opponentWeights, rolloutFrames);
    if (score > best.score) {
      best = { actionIndex, score };
    }
  }

  return best;
}

function scoreActionRollout(
  state: Readonly<GameState>,
  team: Team,
  actionIndex: number,
  opponentWeights: PolicyWeights,
  rolloutFrames: number
): number {
  const simulated = cloneState(state as GameState);
  const controlled = simulated.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  const opponentTeam = team === 'red' ? 'blue' : 'red';
  const opponent = simulated.tanks.find((candidate) => candidate.team === opponentTeam && candidate.index === 0);
  if (!controlled || !opponent) {
    return Number.NEGATIVE_INFINITY;
  }

  const command = actionIndexToCommand(actionIndex);
  const initialControlled = (state as GameState).tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  const initialTags = initialControlled ? tagLearningSample(state, team, initialControlled) : [];
  const initialStaminaRatio = initialControlled
    ? initialControlled.stamina / Math.max(1, initialControlled.maxStamina)
    : 1;
  const opponentCommand = actionIndexToCommand(
    greedyPolicyAction(extractTankInputs(simulated, opponentTeam, opponent), opponentWeights)
  );
  const commands: CommandMap = {
    [controlled.id]: command,
    [opponent.id]: opponentCommand
  };
  const before = evaluatePosition(simulated, team).total;

  for (let frame = 0; frame < rolloutFrames; frame += 1) {
    stepGame(simulated, commands, FIXED_DT);
  }

  const after = evaluatePosition(simulated, team).total;
  const delta = evaluatePositionDelta(simulated, state as GameState, team);
  const activeTracks = Math.abs(command.leftTrack) + Math.abs(command.rightTrack);
  const criticalStamina = initialStaminaRatio < CRITICAL_STAMINA_RATIO;
  const urgent =
    initialTags.includes('ownDanger') ||
    (initialTags.includes('finish') && isClinchingFinish(state, team)) ||
    (initialTags.includes('contest') && !criticalStamina);
  const lowStamina = initialStaminaRatio < 0.5;
  const criticalCost = criticalStamina && activeTracks > 1 && !urgent ? 0.34 : 0;
  const staminaCost = activeTracks * (lowStamina && !urgent ? 0.08 : 0.006) + criticalCost;
  const restBonus = lowStamina && !urgent && activeTracks === 0 ? 0.11 : 0;
  const cornerBonus = initialTags.includes('corner') ? delta.breakdown.cornerEscape * 0.85 : 0;
  const contestBonus = initialTags.includes('contest') ? delta.breakdown.contest * 1.15 : 0;
  return after - before + cornerBonus + contestBonus + restBonus - staminaCost;
}

function greedyPolicyAction(
  inputs: readonly number[],
  weights: PolicyWeights,
): number {
  const probabilities = policyProbabilities(evaluatePolicy(inputs, weights));
  return probabilities.reduce(
    (best, value, index) => value > probabilities[best] ? index : best,
    0
  );
}

function extractTankInputsForTeam(state: Readonly<GameState>, team: Team): number[] {
  const tank = state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  if (!tank) {
    return Array.from({ length: 36 }, () => 0);
  }
  return extractTankInputs(state, team, tank);
}

function weightSelfPlaySample(sample: PendingSample, finalState: GameState, redDiff: number): LearningSample {
  const sign = sample.team === 'red' ? 1 : -1;
  const teamDiff = redDiff * sign;
  const startAttackX = attackX(sample.team, sample.startBallX);
  const endAttackX = attackX(sample.team, finalState.ball.position.x);
  const progress = (endAttackX - startAttackX) / FIELD.length;
  const winnerBonus = teamDiff > 0 ? 1.9 : teamDiff < 0 ? 0.45 : 1;
  const progressBonus = clampRange(1 + progress * 1.4, 0.55, 1.7);
  const rolloutBonus = clampRange(1 + sample.rolloutScore * 0.7, 0.45, 1.65);
  const tacticalBonus =
    sample.tags.includes('finish') ||
    sample.tags.includes('ownDanger') ||
    sample.tags.includes('corner') ||
    sample.tags.includes('contest')
      ? 1.25
      : 1;

  return {
    inputs: [...sample.inputs],
    actionIndex: sample.actionIndex,
    team: sample.team,
    frame: sample.frame,
    tags: [...sample.tags],
    weight: clampRange(winnerBonus * progressBonus * rolloutBonus * tacticalBonus, 0.2, 3.4)
  };
}

function createSeededInitialState(random: () => number, match: number): GameState {
  const state = createInitialState();
  const red = tank(state, 'red');
  const blue = tank(state, 'blue');
  const yJitter = (random() - 0.5) * FIELD.width * 0.28;
  const xJitter = (random() - 0.5) * FIELD.length * 0.16;

  state.ball.position = {
    x: FIELD.length / 2 + xJitter,
    y: FIELD.width / 2 + yJitter
  };
  state.ball.velocity = {
    x: (random() - 0.5) * 140,
    y: (random() - 0.5) * 140
  };

  red.position = {
    x: 160 + random() * 80,
    y: FIELD.width / 2 - yJitter * 0.22 + (random() - 0.5) * 30
  };
  red.angle = (random() - 0.5) * 0.28;

  blue.position = {
    x: FIELD.length - 160 - random() * 80,
    y: FIELD.width / 2 + yJitter * 0.22 + (random() - 0.5) * 30
  };
  blue.angle = Math.PI + (random() - 0.5) * 0.28;

  const scenario = match % 6;

  if (scenario === 1) {
    placeKickoffContest(state, red, blue, random);
  } else if (scenario === 2) {
    placeOwnCornerContest(state, red, blue, 'red', random);
  } else if (scenario === 3) {
    placeOwnCornerContest(state, red, blue, 'blue', random);
  } else if (scenario === 4) {
    state.ball.position.x = FIELD.length - FIELD.ballRadius - 55 - random() * 30;
    state.ball.position.y = random() < 0.5
      ? FIELD.ballRadius + 26 + random() * 28
      : FIELD.width - FIELD.ballRadius - 26 - random() * 28;
  } else if (scenario === 5) {
    state.ball.position.x = FIELD.ballRadius + 55 + random() * 30;
    state.ball.position.y = FIELD.width / 2 + (random() - 0.5) * FIELD.goalMouth * 0.8;
  } else if (match % 12 === 6) {
    state.ball.position.x = FIELD.length - FIELD.ballRadius - 44 - random() * 28;
    state.ball.position.y = random() < 0.5
      ? FIELD.ballRadius + 18 + random() * 22
      : FIELD.width - FIELD.ballRadius - 18 - random() * 22;
    red.position = {
      x: state.ball.position.x - 160 - random() * 24,
      y: clampRange(state.ball.position.y + (state.ball.position.y < FIELD.width / 2 ? 92 : -92), FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
    };
    red.angle = 0;
    blue.position = {
      x: state.ball.position.x - 58 - random() * 28,
      y: clampRange(state.ball.position.y + (state.ball.position.y < FIELD.width / 2 ? 54 : -54), FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
    };
    blue.angle = Math.PI;
    red.stamina = red.maxStamina * LOW_STAMINA_CURRICULUM_RATIO;
    blue.stamina = blue.maxStamina * LOW_STAMINA_CURRICULUM_RATIO;
  }

  return state;
}

function placeKickoffContest(
  state: GameState,
  red: Tank,
  blue: Tank,
  random: () => number
): void {
  const side = random() < 0.5 ? -1 : 1;
  state.ball.position = {
    x: FIELD.length / 2 + (random() - 0.5) * 80,
    y: FIELD.width / 2 + side * (20 + random() * 86)
  };
  state.ball.velocity = {
    x: (random() - 0.5) * 60,
    y: side * (random() - 0.5) * 70
  };
  red.position = {
    x: state.ball.position.x - (74 + random() * 58),
    y: clampRange(state.ball.position.y - side * (18 + random() * 38), FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
  red.angle = Math.atan2(state.ball.position.y - red.position.y, state.ball.position.x - red.position.x);
  blue.position = {
    x: state.ball.position.x + (58 + random() * 54),
    y: clampRange(state.ball.position.y + side * (10 + random() * 36), FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
  blue.angle = Math.atan2(state.ball.position.y - blue.position.y, state.ball.position.x - blue.position.x);
  if (random() < 0.42) {
    red.stamina = red.maxStamina * (0.32 + random() * 0.24);
  }
  if (random() < 0.42) {
    blue.stamina = blue.maxStamina * (0.32 + random() * 0.24);
  }
}

function placeOwnCornerContest(
  state: GameState,
  red: Tank,
  blue: Tank,
  defendingTeam: Team,
  random: () => number
): void {
  const side = random() < 0.5 ? -1 : 1;
  const ownX = defendingTeam === 'red'
    ? FIELD.ballRadius + 22 + random() * 46
    : FIELD.length - FIELD.ballRadius - 22 - random() * 46;
  const wallY = side < 0
    ? FIELD.ballRadius + 15 + random() * 30
    : FIELD.width - FIELD.ballRadius - 15 - random() * 30;
  const defender = defendingTeam === 'red' ? red : blue;
  const attacker = defendingTeam === 'red' ? blue : red;
  const defenderSign = defendingTeam === 'red' ? 1 : -1;

  state.ball.position = { x: ownX, y: wallY };
  state.ball.velocity = {
    x: defenderSign * (random() - 0.5) * 38,
    y: side * (random() - 0.5) * 38
  };
  defender.position = {
    x: clampRange(state.ball.position.x + defenderSign * (96 + random() * 76), FIELD.tankRadius, FIELD.length - FIELD.tankRadius),
    y: clampRange(state.ball.position.y - side * (84 + random() * 58), FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
  attacker.position = {
    x: clampRange(state.ball.position.x + defenderSign * (54 + random() * 54), FIELD.tankRadius, FIELD.length - FIELD.tankRadius),
    y: clampRange(state.ball.position.y - side * (36 + random() * 44), FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
  defender.angle = Math.atan2(
    state.ball.position.y - defender.position.y,
    state.ball.position.x - defender.position.x
  );
  attacker.angle = Math.atan2(
    state.ball.position.y - attacker.position.y,
    state.ball.position.x - attacker.position.x
  );
  if (random() < 0.58) {
    defender.stamina = defender.maxStamina * (0.3 + random() * 0.24);
  }
}

function tank(state: GameState, team: Team): Tank {
  const found = state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  if (!found) {
    throw new Error(`Missing ${team}-0`);
  }
  return found;
}

function attackX(team: Team, x: number): number {
  return team === 'red' ? x : FIELD.length - x;
}

function isClinchingFinish(state: Readonly<GameState>, team: Team): boolean {
  const ballAttackX = attackX(team, state.ball.position.x);
  const lane = Math.abs(state.ball.position.y - FIELD.width / 2) < FIELD.goalMouth * 0.46;
  const attackVelocity = state.ball.velocity.x * (team === 'red' ? 1 : -1);
  return ballAttackX > FIELD.length - 270 && lane && attackVelocity > -25;
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

function clampRange(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
