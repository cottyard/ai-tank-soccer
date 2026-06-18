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
const URGENT_OWN_GOAL_ROLLOUT_FRAMES = 72;
const ATTACK_STALL_SEQUENCE_FIRST_FRAMES = 24;
const ATTACK_STALL_SEQUENCE_SECOND_FRAMES = 66;
const CENTRAL_FINISH_SEQUENCE_FIRST_FRAMES = 6;
const CENTRAL_FINISH_SEQUENCE_SECOND_FRAMES = 54;
const DEFAULT_IMPROVEMENT_MARGIN = 0.018;

export function chooseTacticalAction(options: TacticalActionOptions): TacticalActionChoice {
  const rolloutFrames = Math.max(1, Math.floor(options.rolloutFrames ?? defaultRolloutFrames(options)));
  const sequenceProfile = options.rolloutFrames === undefined
    ? tacticalSequenceProfile(options.state, options.team)
    : undefined;
  const policyActionIndex = clampActionIndex(options.policyActionIndex);
  const policyScore = sequenceProfile
    ? scoreTacticalActionSequence({
        ...options,
        actionIndex: policyActionIndex,
        sequenceProfile
      })
    : scoreTacticalAction({
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

    const score = sequenceProfile
      ? scoreTacticalActionSequence({
          ...options,
          actionIndex,
          sequenceProfile
        })
      : scoreTacticalAction({
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
  if (isUrgentOwnGoalThreat(options.state, options.team)) {
    return URGENT_OWN_GOAL_ROLLOUT_FRAMES;
  }

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

type TacticalSequenceProfile = {
  firstFrames: number;
  secondFrames: number;
};

function scoreTacticalActionSequence(
  options: Omit<ScoreOptions, 'rolloutFrames'> & { sequenceProfile: TacticalSequenceProfile }
): number {
  let best = Number.NEGATIVE_INFINITY;
  for (let followupActionIndex = 0; followupActionIndex < POLICY_ACTION_COUNT; followupActionIndex += 1) {
    const score = scoreTacticalActionSequencePair({
      ...options,
      followupActionIndex
    });
    if (score > best) {
      best = score;
    }
  }
  return best;
}

function scoreTacticalActionSequencePair(
  options: Omit<ScoreOptions, 'rolloutFrames'> & {
    followupActionIndex: number;
    sequenceProfile: TacticalSequenceProfile;
  }
): number {
  const initial = options.state as GameState;
  const simulated = cloneState(initial);
  const controlled = controlledTank(simulated, options.team);
  if (!controlled) {
    return Number.NEGATIVE_INFINITY;
  }

  const before = evaluatePosition(simulated, options.team).total;
  const firstCommands: CommandMap = {
    [controlled.id]: actionIndexToCommand(options.actionIndex)
  };
  const followupCommands: CommandMap = {
    [controlled.id]: actionIndexToCommand(options.followupActionIndex)
  };
  const opponentTeam = options.team === 'red' ? 'blue' : 'red';
  const opponent = controlledTank(simulated, opponentTeam);
  if (opponent && options.opponentActionIndex !== undefined) {
    const opponentCommand = actionIndexToCommand(options.opponentActionIndex);
    firstCommands[opponent.id] = opponentCommand;
    followupCommands[opponent.id] = opponentCommand;
  }

  for (let frame = 0; frame < options.sequenceProfile.firstFrames; frame += 1) {
    stepGame(simulated, firstCommands, FIXED_DT);
  }
  for (let frame = 0; frame < options.sequenceProfile.secondFrames; frame += 1) {
    stepGame(simulated, followupCommands, FIXED_DT);
  }

  const after = evaluatePosition(simulated, options.team).total;
  const delta = evaluatePositionDelta(simulated, initial, options.team);
  const firstAction = actionIndexToCommand(options.actionIndex);
  const followupAction = actionIndexToCommand(options.followupActionIndex);
  const trackCost =
    (Math.abs(firstAction.leftTrack) + Math.abs(firstAction.rightTrack)) * 0.55 +
    (Math.abs(followupAction.leftTrack) + Math.abs(followupAction.rightTrack)) * 0.45;
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

function isUrgentOwnGoalThreat(state: Readonly<GameState>, team: Team): boolean {
  const sign = team === 'red' ? 1 : -1;
  const ballAttackX = attackX(team, state.ball.position.x);
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  const incomingSpeed = -state.ball.velocity.x * sign;
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);

  return ballAttackX < FIELD.length * 0.28 &&
    lane > 0.72 &&
    incomingSpeed > 120 &&
    ballSpeed > 140;
}

function tacticalSequenceProfile(state: Readonly<GameState>, team: Team): TacticalSequenceProfile | undefined {
  if (shouldUseCentralFinishSequence(state, team)) {
    return {
      firstFrames: CENTRAL_FINISH_SEQUENCE_FIRST_FRAMES,
      secondFrames: CENTRAL_FINISH_SEQUENCE_SECOND_FRAMES
    };
  }

  if (shouldUseAttackStallSequence(state, team)) {
    return {
      firstFrames: ATTACK_STALL_SEQUENCE_FIRST_FRAMES,
      secondFrames: ATTACK_STALL_SEQUENCE_SECOND_FRAMES
    };
  }

  return undefined;
}

function shouldUseAttackStallSequence(state: Readonly<GameState>, team: Team): boolean {
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (ballSpeed > 18) {
    return false;
  }

  const tank = controlledTank(state as GameState, team);
  if (!tank) {
    return false;
  }
  if (staminaRatio(tank) < 0.5) {
    return false;
  }

  const ballDistance = Math.hypot(
    tank.position.x - state.ball.position.x,
    tank.position.y - state.ball.position.y
  );
  if (ballDistance > FIELD.tankRadius * 3.1) {
    return false;
  }

  const ballAttackX = attackX(team, state.ball.position.x);
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  const sideWallPinned = sideWallDistance(state.ball.position.y) < FIELD.ballRadius + 16;
  return (ballAttackX > FIELD.length - 245 && lane > 0.45) ||
    (ballAttackX > FIELD.length - 210 && sideWallPinned);
}

function shouldUseCentralFinishSequence(state: Readonly<GameState>, team: Team): boolean {
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (ballSpeed > 12) {
    return false;
  }

  const tank = controlledTank(state as GameState, team);
  if (!tank || staminaRatio(tank) < 0.78) {
    return false;
  }

  const ballDistance = Math.hypot(
    tank.position.x - state.ball.position.x,
    tank.position.y - state.ball.position.y
  );
  if (ballDistance < FIELD.tankRadius * 1.08 || ballDistance > FIELD.tankRadius * 1.55) {
    return false;
  }

  const ballAttackX = attackX(team, state.ball.position.x);
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  return ballAttackX > FIELD.length - 135 && lane > 0.9;
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

function staminaRatio(tank: Tank): number {
  return tank.maxStamina > 0 ? clamp01(tank.stamina / tank.maxStamina) : 0;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
