import { FIXED_DT } from '../game/match';
import { FIELD, cloneState, type GameState, type Tank, type Team } from '../game/model';
import { stepGame } from '../game/simulation';
import type { CommandMap } from '../game/strategy';
import { POLICY_ACTION_COUNT, actionIndexToCommand } from './policyActions';
import { evaluatePosition, evaluatePositionDelta } from './positionEvaluation';

export type TacticalActionChoice = {
  actionIndex: number;
  score: number;
  policyScore: number;
  actionScores: number[];
};

export type TacticalActionOptions = {
  state: Readonly<GameState>;
  team: Team;
  policyActionIndex: number;
  opponentActionIndex?: number;
  rolloutFrames?: number;
  improvementMargin?: number;
};

const DEFAULT_ROLLOUT_FRAMES = 18;
const PINNED_ATTACK_CORNER_ROLLOUT_FRAMES = 120;
const DEFAULT_IMPROVEMENT_MARGIN = 0.018;

export function chooseTacticalAction(options: TacticalActionOptions): TacticalActionChoice {
  const rolloutFrames = Math.max(1, Math.floor(options.rolloutFrames ?? defaultRolloutFrames(options)));
  const policyActionIndex = clampActionIndex(options.policyActionIndex);
  const policyScore = scoreTacticalAction({
    ...options,
    actionIndex: policyActionIndex,
    rolloutFrames
  });
  const actionScores = Array.from({ length: POLICY_ACTION_COUNT }, () => Number.NEGATIVE_INFINITY);
  actionScores[policyActionIndex] = policyScore;
  let best = {
    actionIndex: policyActionIndex,
    score: policyScore
  };

  for (let actionIndex = 0; actionIndex < POLICY_ACTION_COUNT; actionIndex += 1) {
    if (actionIndex === policyActionIndex) {
      continue;
    }

    const score = scoreTacticalAction({
      ...options,
      actionIndex,
      rolloutFrames
    });
    actionScores[actionIndex] = score;
    if (score > best.score + 1e-9) {
      best = { actionIndex, score };
    }
  }

  const margin = options.improvementMargin ?? DEFAULT_IMPROVEMENT_MARGIN;
  const actionIndex = best.score > policyScore + margin
    ? best.actionIndex
    : policyActionIndex;

  return {
    actionIndex,
    score: actionIndex === best.actionIndex ? best.score : policyScore,
    policyScore,
    actionScores
  };
}

function defaultRolloutFrames(options: TacticalActionOptions): number {
  return isSlowPinnedAttackingCorner(options.state, options.team)
    ? PINNED_ATTACK_CORNER_ROLLOUT_FRAMES
    : DEFAULT_ROLLOUT_FRAMES;
}

type ScoreOptions = TacticalActionOptions & {
  actionIndex: number;
  rolloutFrames: number;
};

function scoreTacticalAction(options: ScoreOptions): number {
  const initial = options.state as GameState;
  const simulated = cloneState(initial);
  const controlled = controlledTank(simulated, options.team);
  if (!controlled) {
    return Number.NEGATIVE_INFINITY;
  }

  const opponentTeam = options.team === 'red' ? 'blue' : 'red';
  const opponent = controlledTank(simulated, opponentTeam);
  const commands: CommandMap = {
    [controlled.id]: actionIndexToCommand(options.actionIndex)
  };
  if (opponent && options.opponentActionIndex !== undefined) {
    commands[opponent.id] = actionIndexToCommand(options.opponentActionIndex);
  }

  const before = evaluatePosition(simulated, options.team).total;
  for (let frame = 0; frame < options.rolloutFrames; frame += 1) {
    stepGame(simulated, commands, FIXED_DT);
  }

  const after = evaluatePosition(simulated, options.team).total;
  const delta = evaluatePositionDelta(simulated, initial, options.team);
  const action = actionIndexToCommand(options.actionIndex);
  const trackCost = Math.abs(action.leftTrack) + Math.abs(action.rightTrack);
  return after - before +
    delta.breakdown.cornerEscape * 0.45 +
    -trackCost * 0.004;
}

function isSlowPinnedAttackingCorner(state: Readonly<GameState>, team: Team): boolean {
  const ball = state.ball;
  const ballSpeed = Math.hypot(ball.velocity.x, ball.velocity.y);
  if (ballSpeed > 50) {
    return false;
  }

  return attackX(team, ball.position.x) > FIELD.length - FIELD.ballRadius - 115 &&
    sideWallDistance(ball.position.y) < FIELD.ballRadius + 58;
}

function controlledTank(state: GameState, team: Team): Tank | undefined {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
}

function clampActionIndex(index: number): number {
  if (!Number.isFinite(index)) {
    return 4;
  }
  return Math.max(0, Math.min(POLICY_ACTION_COUNT - 1, Math.round(index)));
}

function attackX(team: Team, x: number): number {
  return team === 'red' ? x : FIELD.length - x;
}

function sideWallDistance(y: number): number {
  return Math.min(y - FIELD.ballRadius, FIELD.width - FIELD.ballRadius - y);
}
