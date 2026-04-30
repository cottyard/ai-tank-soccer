import { FIXED_DT } from '../game/match';
import { FIELD, cloneState, createInitialState, type GameState, type Tank, type Team } from '../game/model';
import { stepGame } from '../game/simulation';
import type { CommandMap } from '../game/strategy';
import { actionIndexToCommand, POLICY_ACTION_COUNT } from './policyActions';
import { LearningReplayBuffer, tagLearningSample, trainOfflineFromReplay, type LearningSample } from './imitationLearning';
import { evaluatePolicy, policyProbabilities, type PolicyWeights } from './policyNetwork';
import { extractTankInputs } from './neuralStrategy';
import { evaluatePosition, evaluatePositionDelta } from './positionEvaluation';

type ScenarioName = 'finish' | 'defense' | 'corner' | 'duel' | 'kickoff' | 'ownCornerContest';

export type CurriculumCollectionOptions = {
  weights: PolicyWeights;
  scenarios?: number;
  rolloutFrames?: number;
  seed?: number;
};

export type CurriculumCollectionResult = {
  samples: LearningSample[];
  byScenario: Record<ScenarioName, number>;
};

export type CurriculumTrainingOptions = CurriculumCollectionOptions & {
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
};

export type CurriculumTrainingResult = {
  weights: number[];
  loss: number;
  trainedSamples: number;
  modelVersion: number;
  samples: number;
  byScenario: Record<ScenarioName, number>;
};

const SCENARIOS: ScenarioName[] = ['finish', 'defense', 'corner', 'duel', 'kickoff', 'ownCornerContest'];
const DEFAULT_ROLLOUT_FRAMES = 16;

export function collectCurriculumSamples(options: CurriculumCollectionOptions): CurriculumCollectionResult {
  const scenarioCount = Math.max(1, Math.floor(options.scenarios ?? 256));
  const rolloutFrames = Math.max(1, Math.floor(options.rolloutFrames ?? DEFAULT_ROLLOUT_FRAMES));
  const random = createSeededRandom(options.seed ?? 1);
  const samples: LearningSample[] = [];
  const byScenario: Record<ScenarioName, number> = {
    finish: 0,
    defense: 0,
    corner: 0,
    duel: 0,
    kickoff: 0,
    ownCornerContest: 0
  };

  for (let index = 0; index < scenarioCount; index += 1) {
    const scenario = SCENARIOS[index % SCENARIOS.length];
    const team: Team = index % 2 === 0 ? 'red' : 'blue';
    const state = createScenarioState(scenario, team, random);
    const sample = createExpertSample(state, team, options.weights, scenario, rolloutFrames);
    if (sample) {
      samples.push(sample);
      byScenario[scenario] += 1;
    }
  }

  return { samples, byScenario };
}

export function trainCurriculumPolicy(options: CurriculumTrainingOptions): CurriculumTrainingResult {
  const collection = collectCurriculumSamples(options);
  const replay = new LearningReplayBuffer({ maxSamples: collection.samples.length || 1 });
  replay.load(collection.samples);
  const trained = trainOfflineFromReplay(options.weights, replay, {
    epochs: options.epochs ?? 18,
    batchSize: options.batchSize ?? 64,
    learningRate: options.learningRate ?? 0.008,
    l2: 0.00032,
    gradientClip: 1,
    seed: options.seed ?? 1
  });

  return {
    weights: trained.weights,
    loss: trained.loss,
    trainedSamples: trained.trainedSamples,
    modelVersion: trained.modelVersion,
    samples: collection.samples.length,
    byScenario: collection.byScenario
  };
}

function createExpertSample(
  state: GameState,
  team: Team,
  weights: PolicyWeights,
  scenario: ScenarioName,
  rolloutFrames: number
): LearningSample | null {
  const tank = controlledTank(state, team);
  if (!tank) {
    return null;
  }

  const policyAction = greedyPolicyAction(extractTankInputs(state, team, tank), weights);
  let best = {
    actionIndex: policyAction,
    score: Number.NEGATIVE_INFINITY
  };

  for (let actionIndex = 0; actionIndex < POLICY_ACTION_COUNT; actionIndex += 1) {
    const score = scoreAction(state, team, actionIndex, rolloutFrames);
    if (score > best.score) {
      best = { actionIndex, score };
    }
  }

  const tags = tagLearningSample(state, team, tank);
  return {
    inputs: extractTankInputs(state, team, tank),
    actionIndex: best.actionIndex,
    team,
    frame: state.frame,
    tags,
    weight: sampleWeight(tags, scenario, best.score)
  };
}

function scoreAction(
  state: Readonly<GameState>,
  team: Team,
  actionIndex: number,
  rolloutFrames: number
): number {
  const initial = state as GameState;
  const simulated = cloneState(initial);
  const tank = controlledTank(simulated, team);
  if (!tank) {
    return Number.NEGATIVE_INFINITY;
  }

  const commands: CommandMap = {
    [tank.id]: actionIndexToCommand(actionIndex)
  };
  const before = evaluatePosition(simulated, team).total;
  for (let frame = 0; frame < rolloutFrames; frame += 1) {
    stepGame(simulated, commands, FIXED_DT);
  }
  const after = evaluatePosition(simulated, team).total;
  const delta = evaluatePositionDelta(simulated, initial, team);
  const command = actionIndexToCommand(actionIndex);
  const activeTracks = Math.abs(command.leftTrack) + Math.abs(command.rightTrack);
  const forwardBias = command.leftTrack === 1 && command.rightTrack === 1
    ? forwardActionBonus(initial, simulated, team)
    : 0;
  const turnPenalty = command.leftTrack !== command.rightTrack ? 0.035 : 0;
  const trackCost = activeTracks * 0.006 + turnPenalty;

  return after - before + delta.breakdown.cornerEscape * 0.7 + forwardBias - trackCost;
}

function forwardActionBonus(initial: Readonly<GameState>, simulated: Readonly<GameState>, team: Team): number {
  const ballProgress =
    (attackX(team, simulated.ball.position.x) - attackX(team, initial.ball.position.x)) /
    FIELD.length;
  const tank = controlledTank(initial, team);
  if (!tank) {
    return 0;
  }
  const ballDistance = Math.hypot(
    tank.position.x - initial.ball.position.x,
    tank.position.y - initial.ball.position.y
  );
  const contactReady = ballDistance < FIELD.tankRadius + FIELD.ballRadius + 42 ? 0.08 : 0;
  return clampRange(ballProgress * 2.4 + contactReady, 0, 0.16);
}

function sampleWeight(tags: readonly string[], scenario: ScenarioName, score: number): number {
  let weight = 1 + clamp01(Math.abs(score)) * 0.5;
  if (scenario === 'finish') {
    weight += 0.65;
  }
  if (scenario === 'defense') {
    weight += 0.55;
  }
  if (scenario === 'corner') {
    weight += 0.6;
  }
  if (scenario === 'kickoff') {
    weight += 0.45;
  }
  if (scenario === 'ownCornerContest') {
    weight += 0.85;
  }
  if (tags.includes('contest')) {
    weight += 0.5;
  }
  if (tags.includes('contact')) {
    weight += 0.55;
  }
  if (tags.includes('ownDanger') || tags.includes('finish') || tags.includes('corner')) {
    weight += 0.45;
  }
  return clampRange(weight, 0.6, 3.5);
}

function createScenarioState(scenario: ScenarioName, team: Team, random: () => number): GameState {
  if (scenario === 'finish') {
    return createFinishState(team, random);
  }
  if (scenario === 'defense') {
    return createDefenseState(team, random);
  }
  if (scenario === 'corner') {
    return createCornerState(team, random);
  }
  if (scenario === 'kickoff') {
    return createKickoffContestState(team, random);
  }
  if (scenario === 'ownCornerContest') {
    return createOwnCornerContestState(team, random);
  }
  return createDuelState(team, random);
}

function createFinishState(team: Team, random: () => number): GameState {
  const state = createInitialState();
  const side = random() < 0.5 ? -1 : 1;
  const ballX = FIELD.length - 175 - random() * 105;
  const ballY = FIELD.width / 2 + side * (random() * FIELD.goalMouth * 0.32);
  placeBall(state, team, ballX, ballY, random() * 65, side * (random() - 0.5) * 35);
  placeTank(state, team, ballX - 155 - random() * 36, ballY + side * (random() - 0.2) * 58, side * (random() - 0.5) * 0.32);
  placeOpponent(state, team, FIELD.length - 130 - random() * 45, FIELD.width / 2 - side * (155 + random() * 80), Math.PI);
  return state;
}

function createDefenseState(team: Team, random: () => number): GameState {
  const state = createInitialState();
  const side = random() < 0.5 ? -1 : 1;
  const ballX = 145 + random() * 115;
  const ballY = FIELD.width / 2 + side * (random() * FIELD.goalMouth * 0.42);
  placeBall(state, team, ballX, ballY, -120 - random() * 210, side * (random() - 0.5) * 90);
  placeTank(state, team, 95 + random() * 90, FIELD.width / 2 + side * (random() * 90), side * (random() - 0.2) * 0.7);
  placeOpponent(state, team, 310 + random() * 140, FIELD.width / 2 - side * (70 + random() * 95), Math.PI);
  return state;
}

function createCornerState(team: Team, random: () => number): GameState {
  const state = createInitialState();
  const attackCorner = random() < 0.72;
  const side = random() < 0.5 ? -1 : 1;
  const ballX = attackCorner
    ? FIELD.length - FIELD.ballRadius - 20 - random() * 58
    : FIELD.ballRadius + 35 + random() * 86;
  const ballY = side < 0
    ? FIELD.ballRadius + 14 + random() * 34
    : FIELD.width - FIELD.ballRadius - 14 - random() * 34;
  placeBall(state, team, ballX, ballY, (random() - 0.5) * 35, (random() - 0.5) * 35);
  placeTank(state, team, ballX + (attackCorner ? -150 : 105) + (random() - 0.5) * 45, ballY - side * (82 + random() * 44), side * (random() - 0.3) * 0.55);
  placeOpponent(state, team, ballX + (attackCorner ? -60 : 135), ballY - side * (48 + random() * 32), Math.PI);
  return state;
}

function createDuelState(team: Team, random: () => number): GameState {
  const state = createInitialState();
  const side = random() < 0.5 ? -1 : 1;
  const ballX = FIELD.length / 2 + (random() - 0.25) * FIELD.length * 0.32;
  const ballY = FIELD.width / 2 + side * (30 + random() * 125);
  placeBall(state, team, ballX, ballY, (random() - 0.2) * 130, side * (random() - 0.5) * 105);
  placeTank(state, team, ballX - 130 - random() * 70, ballY + side * (random() * 78), side * (random() - 0.5) * 0.55);
  placeOpponent(state, team, ballX + 85 + random() * 95, ballY - side * (random() * 82), Math.PI + side * (random() - 0.5) * 0.55);
  return state;
}

function createKickoffContestState(team: Team, random: () => number): GameState {
  const state = createInitialState();
  const side = random() < 0.5 ? -1 : 1;
  const ballX = FIELD.length / 2 + (random() - 0.5) * 90;
  const ballY = FIELD.width / 2 + side * random() * 95;
  placeBall(state, team, ballX, ballY, (random() - 0.5) * 70, side * (random() - 0.5) * 70);
  placeTank(state, team, 190 + random() * 90, FIELD.width / 2 - side * (random() * 45), side * (random() - 0.3) * 0.36);
  placeOpponent(state, team, ballX + 45 + random() * 95, ballY + side * (random() * 48), Math.PI + side * (random() - 0.5) * 0.42);
  const tank = controlledTank(state, team);
  if (tank && random() < 0.45) {
    tank.stamina = tank.maxStamina * (0.34 + random() * 0.22);
  }
  return state;
}

function createOwnCornerContestState(team: Team, random: () => number): GameState {
  const state = createInitialState();
  const side = random() < 0.5 ? -1 : 1;
  const ballX = FIELD.ballRadius + 18 + random() * 42;
  const ballY = side < 0
    ? FIELD.ballRadius + 14 + random() * 28
    : FIELD.width - FIELD.ballRadius - 14 - random() * 28;
  placeBall(state, team, ballX, ballY, (random() - 0.5) * 22, (random() - 0.5) * 22);
  placeTank(state, team, ballX + 105 + random() * 70, ballY - side * (92 + random() * 46), side < 0 ? Math.PI / 2 : -Math.PI / 2);
  placeOpponent(state, team, ballX + 62 + random() * 60, ballY - side * (42 + random() * 38), Math.PI);
  return state;
}

function placeBall(
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

function placeTank(state: GameState, team: Team, attackFrameX: number, attackFrameY: number, angle: number): void {
  const tank = controlledTank(state, team);
  if (!tank) {
    throw new Error(`Missing ${team} tank`);
  }
  tank.position = fieldPoint(team, attackFrameX, attackFrameY);
  tank.velocity = { x: 0, y: 0 };
  tank.angle = fieldAngle(team, angle);
  tank.angularVelocity = 0;
  tank.stamina = tank.maxStamina;
}

function placeOpponent(state: GameState, team: Team, attackFrameX: number, attackFrameY: number, angle: number): void {
  const opponent = controlledTank(state, team === 'red' ? 'blue' : 'red');
  if (!opponent) {
    throw new Error(`Missing opponent for ${team}`);
  }
  opponent.position = fieldPoint(team, attackFrameX, attackFrameY);
  opponent.velocity = { x: 0, y: 0 };
  opponent.angle = fieldAngle(team === 'red' ? 'blue' : 'red', angle);
  opponent.angularVelocity = 0;
  opponent.stamina = opponent.maxStamina;
}

function controlledTank(state: Readonly<GameState>, team: Team): Tank | undefined {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
}

function attackX(team: Team, fieldX: number): number {
  return team === 'red' ? fieldX : FIELD.length - fieldX;
}

function greedyPolicyAction(inputs: readonly number[], weights: PolicyWeights): number {
  const probabilities = policyProbabilities(evaluatePolicy(inputs, weights));
  return probabilities.reduce(
    (best, value, index) => value > probabilities[best] ? index : best,
    0
  );
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

function clampRange(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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
