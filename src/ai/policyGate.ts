import {
  evaluateNeuralWeights,
  type EvaluationOptions,
  type EvaluationResult
} from './neuralTraining';
import { createNeuralStrategy } from './neuralStrategy';
import type { NeuralWeights } from './neuralWeights';
import { traditionalStrategy } from './traditionalStrategy';
import { FIELD, createInitialState, type GameState, type Team } from '../game/model';
import { simulateMatch } from '../game/match';

export type PolicyGateOptions = EvaluationOptions & {
  minDelta?: number;
  evaluate?: (weights: NeuralWeights, options: EvaluationOptions) => EvaluationResult;
};

export type RuntimeEvaluationOptions = {
  seed?: number;
  matches?: number;
  frames?: number;
};

export type RuntimeEvaluationResult = EvaluationResult & {
  goalsFor: number;
  goalsAgainst: number;
  winProxy: number;
};

export type PolicyGateResult = {
  accepted: boolean;
  currentScore: number;
  candidateScore: number;
  current: EvaluationResult;
  candidate: EvaluationResult;
};

export type SelectedPolicyResult = PolicyGateResult & {
  weights: number[];
  source: 'current' | 'candidate';
};

export function evaluatePolicyGate(
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  options: PolicyGateOptions = {}
): PolicyGateResult {
  const { evaluate = evaluateNeuralWeights, minDelta = 0, ...evaluationOptions } = options;
  const current = evaluate(currentWeights, evaluationOptions);
  const candidate = evaluate(candidateWeights, evaluationOptions);

  return {
    accepted: candidate.score > current.score + minDelta,
    currentScore: current.score,
    candidateScore: candidate.score,
    current,
    candidate
  };
}

export function selectAcceptedPolicy(
  currentWeights: NeuralWeights,
  candidateWeights: NeuralWeights,
  options: PolicyGateOptions = {}
): SelectedPolicyResult {
  const gate = evaluatePolicyGate(currentWeights, candidateWeights, options);
  return {
    ...gate,
    weights: gate.accepted ? [...candidateWeights] : [...currentWeights],
    source: gate.accepted ? 'candidate' : 'current'
  };
}

export function evaluateRuntimePolicy(
  weights: NeuralWeights,
  options: RuntimeEvaluationOptions = {}
): RuntimeEvaluationResult {
  const seed = options.seed ?? 1;
  const matches = Math.max(1, Math.floor(options.matches ?? 8));
  const frames = Math.max(1, Math.floor(options.frames ?? 30 * 30));
  const neural = createNeuralStrategy({
    weights,
    name: 'neural-runtime-gate',
    tacticalRollout: true
  });
  let goalsFor = 0;
  let goalsAgainst = 0;
  let wins = 0;
  let ballProgress = 0;

  for (let match = 0; match < matches; match += 1) {
    const team: Team = match % 2 === 0 ? 'red' : 'blue';
    const initialState = createSeededInitialState(seed, match, team);
    const result = simulateMatch({
      red: team === 'red' ? neural : traditionalStrategy,
      blue: team === 'blue' ? neural : traditionalStrategy,
      frames,
      initialState
    }).state;
    const forGoals = team === 'red' ? result.score.red : result.score.blue;
    const againstGoals = team === 'red' ? result.score.blue : result.score.red;

    goalsFor += forGoals;
    goalsAgainst += againstGoals;
    wins += forGoals > againstGoals ? 1 : forGoals === againstGoals ? 0.5 : 0;
    ballProgress += attackProgress(result, initialState, team);
  }

  const goalDiff = goalsFor - goalsAgainst;
  const averageBallProgress = ballProgress / matches;
  const winProxy = wins / matches;
  return {
    score: goalDiff * 100 + goalsFor * 35 - goalsAgainst * 55 + averageBallProgress * 80 + winProxy * 45,
    goalDiff,
    ballProgress: averageBallProgress,
    goalsFor,
    goalsAgainst,
    winProxy
  };
}

function createSeededInitialState(seed: number, match: number, team: Team): GameState {
  const random = createSeededRandom(seed + match * 4099);
  const state = createInitialState();
  const attackFrameX = FIELD.length / 2 + (random() - 0.5) * FIELD.length * 0.12;
  const attackFrameY = FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.22;

  state.ball.position = fieldPoint(team, attackFrameX, attackFrameY);
  state.ball.velocity = fieldVector(team, (random() - 0.5) * 120, (random() - 0.5) * 120);
  return state;
}

function attackProgress(state: GameState, initialState: GameState, team: Team): number {
  return (attackX(team, state.ball.position.x) - attackX(team, initialState.ball.position.x)) / FIELD.length;
}

function attackX(team: Team, fieldX: number): number {
  return team === 'red' ? fieldX : FIELD.length - fieldX;
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
