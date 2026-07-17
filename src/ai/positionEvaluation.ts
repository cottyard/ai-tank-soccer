import { FIELD, type GameState, type Tank, type Team } from '../game/model';

export type PositionBreakdown = {
  goal: number;
  ballProgress: number;
  shotLane: number;
  finishThreat: number;
  shotVelocity: number;
  contest: number;
  possession: number;
  ownDanger: number;
  cornerEscape: number;
  stamina: number;
};

export type PositionEvaluation = {
  total: number;
  breakdown: PositionBreakdown;
};

export function evaluatePosition(state: Readonly<GameState>, team: Team): PositionEvaluation {
  const sign = team === 'red' ? 1 : -1;
  const ballAttackX = attackX(team, state.ball.position.x);
  const lane = goalLaneScore(state.ball.position.y);
  const goalDiff = team === 'red'
    ? state.score.red - state.score.blue
    : state.score.blue - state.score.red;
  const ballProgress = normalizeSigned(ballAttackX - FIELD.length / 2, FIELD.length / 2);
  const goalProximity = clamp01((ballAttackX - FIELD.length * 0.68) / (FIELD.length * 0.32));
  const shotVelocity = clamp(state.ball.velocity.x * sign / 420);
  const finishThreat = finishThreatScore(state, team, lane, goalProximity);
  const ownGoalProximity = clamp01((FIELD.length * 0.34 - ballAttackX) / (FIELD.length * 0.34));
  const incomingOwnGoal = clamp01((-state.ball.velocity.x * sign) / 340);
  const ownDanger = clamp01((ownGoalProximity * 0.74 + incomingOwnGoal * 0.42) * lane);
  const tank = controlledTank(state, team);
  const possession = tank ? possessionScore(state, team, tank) : -0.6;
  const contest = tank ? contestScore(state, team, tank) : -0.4;
  const cornerEscape = cornerPositionScore(state, team);
  const stamina = tank ? staminaScore(tank, ownDanger, goalProximity, possession) : 0;

  const breakdown = {
    goal: goalDiff,
    ballProgress,
    shotLane: lane * goalProximity,
    finishThreat,
    shotVelocity,
    contest,
    possession,
    ownDanger,
    cornerEscape,
    stamina
  };
  const total =
    breakdown.goal * 12 +
    breakdown.ballProgress * 1.7 +
    breakdown.shotLane * 0.95 +
    breakdown.finishThreat * 1.05 +
    breakdown.shotVelocity * 0.42 +
    breakdown.contest * 0.7 +
    breakdown.possession * 1.1 -
    breakdown.ownDanger * 2.2 +
    breakdown.cornerEscape * 0.85 +
    breakdown.stamina * 0.22;

  return { total, breakdown };
}

export function evaluatePositionDelta(
  state: Readonly<GameState>,
  initial: Readonly<GameState>,
  team: Team
): PositionEvaluation {
  const after = evaluatePosition(state, team);
  const before = evaluatePosition(initial, team);
  const cornerEscape = cornerEscapeGain(state, initial, team);
  const breakdown = {
    goal: after.breakdown.goal - before.breakdown.goal,
    ballProgress: after.breakdown.ballProgress - before.breakdown.ballProgress,
    shotLane: after.breakdown.shotLane - before.breakdown.shotLane,
    finishThreat: after.breakdown.finishThreat - before.breakdown.finishThreat,
    shotVelocity: after.breakdown.shotVelocity - before.breakdown.shotVelocity,
    contest: after.breakdown.contest - before.breakdown.contest,
    possession: after.breakdown.possession - before.breakdown.possession,
    ownDanger: after.breakdown.ownDanger - before.breakdown.ownDanger,
    cornerEscape,
    stamina: after.breakdown.stamina - before.breakdown.stamina
  };
  const total =
    breakdown.goal * 12 +
    breakdown.ballProgress * 1.9 +
    breakdown.shotLane * 1.05 +
    breakdown.finishThreat * 1.25 +
    breakdown.shotVelocity * 0.5 +
    breakdown.contest * 0.85 +
    breakdown.possession * 1.25 -
    breakdown.ownDanger * 2.45 +
    breakdown.cornerEscape * 1.15 +
    breakdown.stamina * 0.18;

  return { total, breakdown };
}

function finishThreatScore(
  state: Readonly<GameState>,
  team: Team,
  lane: number,
  goalProximity: number
): number {
  const sign = team === 'red' ? 1 : -1;
  const attackVelocity = state.ball.velocity.x * sign;
  const speedTowardGoal = clamp01(attackVelocity / 260);
  const nearGoal = clamp01((attackX(team, state.ball.position.x) - (FIELD.length - 260)) / 160);
  const goalFactor = 0.45 + goalProximity * 0.55;
  const yVelocityDrift = Math.abs(state.ball.velocity.y) / Math.max(1, Math.abs(attackVelocity));
  const straightShot = clamp01(1 - yVelocityDrift * 1.15);

  return clamp01(lane * nearGoal * goalFactor * (0.2 + speedTowardGoal * 0.8) * straightShot);
}

function possessionScore(state: Readonly<GameState>, team: Team, tank: Tank): number {
  const ball = state.ball.position;
  const goal = goalPoint(team);
  const shot = unitVector(goal.x - ball.x, goal.y - ball.y);
  const setupDistance = FIELD.ballRadius + FIELD.tankRadius + 10;
  const setup = {
    x: clampRange(ball.x - shot.x * setupDistance, FIELD.tankRadius, FIELD.length - FIELD.tankRadius),
    y: clampRange(ball.y - shot.y * setupDistance, FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
  const setupDistanceRatio = Math.hypot(tank.position.x - setup.x, tank.position.y - setup.y) / FIELD.length;
  const ballDistance = Math.hypot(tank.position.x - ball.x, tank.position.y - ball.y);
  const contact = ballDistance <= FIELD.ballRadius + FIELD.tankRadius + 20 ? 0.48 : 0;
  const tankToBall = unitVector(ball.x - tank.position.x, ball.y - tank.position.y);
  const behindBall = clamp(tankToBall.x * shot.x + tankToBall.y * shot.y);
  const headingToBall = Math.atan2(ball.y - tank.position.y, ball.x - tank.position.x);
  const headingAlignment = Math.cos(normalizeAngle(headingToBall - tank.angle));
  const shotAlignment = Math.cos(normalizeAngle(Math.atan2(shot.y, shot.x) - tank.angle));
  const opponent = nearestOpponent(state, team);
  const opponentPressure = opponent
    ? clamp01(1 - Math.hypot(opponent.position.x - ball.x, opponent.position.y - ball.y) / (FIELD.tankRadius * 3.2))
    : 0;

  return clamp(
    -setupDistanceRatio * 1.6 +
    contact +
    behindBall * 0.36 +
    headingAlignment * 0.22 +
    shotAlignment * 0.2 -
    opponentPressure * 0.22
  );
}

function contestScore(state: Readonly<GameState>, team: Team, tank: Tank): number {
  const ball = state.ball.position;
  const ballDistance = Math.hypot(tank.position.x - ball.x, tank.position.y - ball.y);
  const close = 1 - clamp01(ballDistance / (FIELD.tankRadius * 3.2));
  const headingToBall = Math.atan2(ball.y - tank.position.y, ball.x - tank.position.x);
  const noseAlignment = Math.cos(normalizeAngle(headingToBall - tank.angle));
  const opponent = nearestOpponent(state, team);
  const opponentPressure = opponent
    ? clamp01(1 - Math.hypot(opponent.position.x - ball.x, opponent.position.y - ball.y) / (FIELD.tankRadius * 2.4))
    : 0;
  const pinned = isCornerPinned(state, team) ? 0.42 : 0;
  const looseMidfield =
    Math.abs(ball.x - FIELD.length / 2) < FIELD.length * 0.24 &&
    Math.hypot(state.ball.velocity.x, state.ball.velocity.y) < 140
      ? 0.28
      : 0;

  return clamp(close * 0.75 + noseAlignment * 0.38 + opponentPressure * 0.26 + pinned + looseMidfield - 0.34);
}

function cornerPositionScore(state: Readonly<GameState>, team: Team): number {
  const ballAttackX = attackX(team, state.ball.position.x);
  const attackingCorner = ballAttackX > FIELD.length - FIELD.ballRadius - 90;
  const defendingCorner = ballAttackX < FIELD.ballRadius + 115;
  if (!attackingCorner && !defendingCorner) {
    return 0;
  }

  const sideDistance = sideWallDistance(state.ball.position.y);
  const centerScore = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.width / 2));
  const wallPenalty = clamp01((FIELD.ballRadius + 48 - sideDistance) / (FIELD.ballRadius + 48));
  return clamp(centerScore - wallPenalty * 0.68);
}

function cornerEscapeGain(
  state: Readonly<GameState>,
  initial: Readonly<GameState>,
  team: Team
): number {
  const initiallyPinned = isCornerPinned(initial, team);
  if (!initiallyPinned) {
    return 0;
  }

  const wallClearGain = (sideWallDistance(state.ball.position.y) - sideWallDistance(initial.ball.position.y)) / (FIELD.width / 2);
  const centerGain =
    (Math.abs(initial.ball.position.y - FIELD.width / 2) -
      Math.abs(state.ball.position.y - FIELD.width / 2)) /
    (FIELD.width / 2);
  const progressGain = (attackX(team, state.ball.position.x) - attackX(team, initial.ball.position.x)) / FIELD.length;
  return clamp(wallClearGain * 1.35 + centerGain * 1.05 + progressGain * 0.45);
}

function isCornerPinned(state: Readonly<GameState>, team: Team): boolean {
  const ballAttackX = attackX(team, state.ball.position.x);
  const nearEnd = ballAttackX > FIELD.length - FIELD.ballRadius - 100 ||
    ballAttackX < FIELD.ballRadius + 100;
  return nearEnd && sideWallDistance(state.ball.position.y) < FIELD.ballRadius + 58;
}

function staminaScore(tank: Tank, ownDanger: number, goalProximity: number, possession: number): number {
  const ratio = tank.maxStamina > 0 ? tank.stamina / tank.maxStamina : 0;
  const urgent = ownDanger > 0.45 || goalProximity > 0.55 || possession > 0.25;
  return urgent ? ratio * 0.2 : ratio - 0.5;
}

function controlledTank(state: Readonly<GameState>, team: Team): Tank | undefined {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
}

function nearestOpponent(state: Readonly<GameState>, team: Team): Tank | undefined {
  let best: Tank | undefined;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const tank of state.tanks) {
    if (tank.team === team) {
      continue;
    }
    const distance = Math.hypot(tank.position.x - state.ball.position.x, tank.position.y - state.ball.position.y);
    if (distance < bestDistance) {
      best = tank;
      bestDistance = distance;
    }
  }
  return best;
}

function goalPoint(team: Team): { x: number; y: number } {
  return {
    x: team === 'red' ? FIELD.length : 0,
    y: FIELD.width / 2
  };
}

function goalLaneScore(y: number): number {
  return 1 - clamp01(Math.abs(y - FIELD.width / 2) / (FIELD.goalMouth * 0.72));
}

function sideWallDistance(y: number): number {
  return Math.min(y - FIELD.ballRadius, FIELD.width - FIELD.ballRadius - y);
}

function attackX(team: Team, x: number): number {
  return team === 'red' ? x : FIELD.length - x;
}

function unitVector(x: number, y: number): { x: number; y: number } {
  const length = Math.hypot(x, y) || 1;
  return { x: x / length, y: y / length };
}

function normalizeSigned(value: number, scale: number): number {
  return clamp(value / scale);
}

function clamp(value: number): number {
  return Math.max(-1, Math.min(1, value));
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
