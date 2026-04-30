import { FIELD, type GameState, type Tank, type Team, type Vec2 } from '../game/model';
import type { Strategy, TankCommand, TrackPower } from '../game/strategy';

const CENTER: Vec2 = { x: FIELD.length / 2, y: FIELD.width / 2 };
const ATTACK: Record<Team, 1 | -1> = { red: 1, blue: -1 };
const GOAL_X: Record<Team, number> = { red: FIELD.length, blue: 0 };
const OWN_GOAL_X: Record<Team, number> = { red: 0, blue: FIELD.length };
const TRACKS = [-1, 0, 1] as const;

const DEFENSE_X = 108;
const DANGER_DEPTH = 265;
const OWN_GOAL_AVOID_DEPTH = 205;
const BALL_PREDICT_SECONDS = 0.75;
const STRIKE_SETUP_DISTANCE = FIELD.ballRadius + FIELD.tankLength + 42;
const STRIKE_APPROACH_TOLERANCE = FIELD.tankRadius + 54;
const STRIKE_LATERAL_TOLERANCE = FIELD.ballRadius + FIELD.tankWidth * 0.58;
const STAMINA_CONSERVE_RATIO = 0.5;
const NEAR_BALL_BUFFER = 24;
const SIDE_WALL_DEPTH = FIELD.ballRadius + 54;
const OPPONENT_CORNER_DEPTH = FIELD.ballRadius + FIELD.tankLength + 72;
const STRAIGHT_HEADING_TOLERANCE = 0.18;

export const traditionalStrategy: Strategy = {
  name: 'traditional',
  decide(state, team) {
    const tank = singleTankForTeam(state, team);
    const target = tacticalTarget(state, team, tank);
    const urgentDefense = ballThreatensOwnGoal(state, team);
    const spendLowStamina = urgentDefense || usefulBallContact(tank, target, state, team);

    return {
      [`${team}-0`]: driveTo(tank, target, spendLowStamina)
    };
  }
};

export default traditionalStrategy;

function singleTankForTeam(state: Readonly<GameState>, team: Team): Tank {
  const tank = state.tanks.find((candidate) => candidate.id === `${team}-0`);
  if (!tank) {
    throw new Error(`traditionalStrategy requires ${team}-0`);
  }
  return tank;
}

function tacticalTarget(state: Readonly<GameState>, team: Team, tank: Tank): Vec2 {
  const ball = state.ball.position;

  if (isWrongSideOwnGoalPush(tank, ball, team)) {
    return ownGoalEscapeTarget(tank, ball, team);
  }

  if (ballThreatensOwnGoal(state, team)) {
    return defensiveTarget(state, team);
  }

  if (opponentCornerTrap(ball, team)) {
    return opponentCornerTarget(state, team, tank);
  }

  if (ballNearSideWall(ball) && tankNearBall(tank, state)) {
    return sideWallRecycleTarget(ball, team);
  }

  return attackTarget(state, team, tank);
}

function defensiveTarget(state: Readonly<GameState>, team: Team): Vec2 {
  const direction = ATTACK[team];
  const ownX = OWN_GOAL_X[team];
  return clampPoint({
    x: ownX + direction * DEFENSE_X,
    y: predictGoalLaneY(state, team)
  });
}

function attackTarget(state: Readonly<GameState>, team: Team, tank: Tank): Vec2 {
  const plan = attackPlan(state, team);
  const ball = state.ball.position;
  const readiness = shotReadiness(tank, ball, plan.shot);
  const approachTarget = clampPoint({
    x: ball.x - plan.shot.x * STRIKE_SETUP_DISTANCE,
    y: ball.y - plan.shot.y * STRIKE_SETUP_DISTANCE
  });
  const aligned =
    readiness.forward > FIELD.ballRadius * 0.35 &&
    Math.abs(readiness.lateral) < STRIKE_LATERAL_TOLERANCE;
  const closeToSetup =
    distance(tank.position, approachTarget) < STRIKE_APPROACH_TOLERANCE ||
    tankNearBall(tank, state);

  if (aligned && closeToSetup) {
    return plan.finishTarget;
  }

  return approachTarget;
}

function attackPlan(state: Readonly<GameState>, team: Team): { shot: Vec2; finishTarget: Vec2 } {
  const finishTarget = { x: GOAL_X[team], y: CENTER.y };
  return {
    shot: unitVector(state.ball.position, finishTarget, ATTACK[team]),
    finishTarget
  };
}

function opponentCornerTarget(state: Readonly<GameState>, team: Team, tank: Tank): Vec2 {
  const ball = state.ball.position;
  const direction = ATTACK[team];
  const inward = ball.y < CENTER.y ? 1 : -1;

  if (tankNearBall(tank, state) && direction * (ball.x - tank.position.x) > -FIELD.ballRadius) {
    return clampPoint({
      x: ball.x - direction * 48,
      y: ball.y + inward * 260
    });
  }

  return clampPoint({
    x: ball.x - direction * (FIELD.ballRadius + FIELD.tankLength + 92),
    y: ball.y + inward * 230
  });
}

function sideWallRecycleTarget(ball: Vec2, team: Team): Vec2 {
  const inward = ball.y < CENTER.y ? 1 : -1;
  return clampPoint({
    x: ball.x - ATTACK[team] * 70,
    y: ball.y + inward * 210
  });
}

function ownGoalEscapeTarget(tank: Tank, ball: Vec2, team: Team): Vec2 {
  const direction = ATTACK[team];
  const side = tank.position.y <= ball.y ? -1 : 1;
  return clampPoint({
    x: ball.x + direction * (FIELD.ballRadius + FIELD.tankLength + 80),
    y: ball.y + side * (FIELD.ballRadius + FIELD.tankWidth + 55)
  });
}

function isWrongSideOwnGoalPush(tank: Tank, ball: Vec2, team: Team): boolean {
  const direction = ATTACK[team];
  const ownX = OWN_GOAL_X[team];
  const ballDeep = direction * (ball.x - ownX) < OWN_GOAL_AVOID_DEPTH;
  const inGoalLane = Math.abs(ball.y - CENTER.y) < FIELD.goalMouth * 0.72;
  const tankOnAttackSideOfBall = direction * (tank.position.x - ball.x) > FIELD.ballRadius * 0.45;
  const facingOwnGoal = Math.cos(tank.angle) * direction < -0.32;

  return ballDeep && inGoalLane && tankOnAttackSideOfBall && facingOwnGoal;
}

function usefulBallContact(
  tank: Tank,
  target: Vec2,
  state: Readonly<GameState>,
  team: Team
): boolean {
  if (!tankNearBall(tank, state)) {
    return false;
  }

  const ball = state.ball.position;
  const plan = attackPlan(state, team);
  const readiness = shotReadiness(tank, ball, plan.shot);
  const toBall = unitVector(tank.position, ball, ATTACK[team]);
  const toTarget = unitVector(tank.position, target, ATTACK[team]);
  const movingThroughBall = dot(toBall, toTarget) > 0.72;
  const attackingContact =
    readiness.forward > -FIELD.ballRadius * 0.25 &&
    Math.abs(readiness.lateral) < STRIKE_LATERAL_TOLERANCE;

  return movingThroughBall && (attackingContact || ballInOwnDangerLane(ball, team));
}

function ballThreatensOwnGoal(state: Readonly<GameState>, team: Team): boolean {
  const direction = ATTACK[team];
  const ball = state.ball;
  const movingTowardOwnGoal = ball.velocity.x * direction < -60;
  const deepInLane = ballInOwnDangerLane(ball.position, team);

  if (deepInLane) {
    return true;
  }

  if (!movingTowardOwnGoal) {
    return false;
  }

  return Math.abs(predictGoalLaneY(state, team) - CENTER.y) < FIELD.goalMouth * 0.72;
}

function ballInOwnDangerLane(ball: Vec2, team: Team): boolean {
  const direction = ATTACK[team];
  const ownX = OWN_GOAL_X[team];
  return direction * (ball.x - ownX) < DANGER_DEPTH &&
    Math.abs(ball.y - CENTER.y) < FIELD.goalMouth * 0.72;
}

function predictGoalLaneY(state: Readonly<GameState>, team: Team): number {
  const direction = ATTACK[team];
  const ball = state.ball;
  const ownBlockX = OWN_GOAL_X[team] + direction * DEFENSE_X;
  const movingTowardOwnGoal = ball.velocity.x * direction < -10;
  const seconds = movingTowardOwnGoal
    ? clamp((ownBlockX - ball.position.x) / ball.velocity.x, 0, BALL_PREDICT_SECONDS)
    : 0;

  return clamp(
    ball.position.y + ball.velocity.y * seconds,
    CENTER.y - FIELD.goalMouth * 0.48,
    CENTER.y + FIELD.goalMouth * 0.48
  );
}

function opponentCornerTrap(ball: Vec2, team: Team): boolean {
  const direction = ATTACK[team];
  const opponentDepth = direction * (GOAL_X[team] - ball.x);
  return opponentDepth < OPPONENT_CORNER_DEPTH && ballNearSideWall(ball);
}

function ballNearSideWall(ball: Vec2): boolean {
  return ball.y < SIDE_WALL_DEPTH || ball.y > FIELD.width - SIDE_WALL_DEPTH;
}

function tankNearBall(tank: Tank, state: Readonly<GameState>): boolean {
  return distance(tank.position, state.ball.position) <=
    tank.radius + state.ball.radius + NEAR_BALL_BUFFER;
}

function shotReadiness(tank: Tank, ball: Vec2, shot: Vec2): { forward: number; lateral: number } {
  const toBall = {
    x: ball.x - tank.position.x,
    y: ball.y - tank.position.y
  };

  return {
    forward: toBall.x * shot.x + toBall.y * shot.y,
    lateral: toBall.x * -shot.y + toBall.y * shot.x
  };
}

function driveTo(tank: Tank, target: Vec2, spendLowStamina = false): TankCommand {
  if (shouldConserveStamina(tank, spendLowStamina)) {
    return { leftTrack: 0, rightTrack: 0 };
  }

  const dx = target.x - tank.position.x;
  const dy = target.y - tank.position.y;
  if (Math.hypot(dx, dy) < 14) {
    return { leftTrack: 0, rightTrack: 0 };
  }

  const desired = Math.atan2(dy, dx);
  const forwardError = normalizeAngle(desired - tank.angle);
  const reverseError = normalizeAngle(desired + Math.PI - tank.angle);
  const reverse = Math.abs(reverseError) + 0.18 < Math.abs(forwardError);
  const headingError = reverse ? reverseError : forwardError;
  const base = reverse ? -1 : 1;

  if (Math.abs(headingError) < STRAIGHT_HEADING_TOLERANCE) {
    return trackPair(base, base);
  }

  if (Math.abs(headingError) > 1.35) {
    return headingError > 0 ? trackPair(1, -1) : trackPair(-1, 1);
  }

  if (headingError > 0) {
    return reverse ? trackPair(0, base) : trackPair(base, 0);
  }

  return reverse ? trackPair(base, 0) : trackPair(0, base);
}

function shouldConserveStamina(tank: Tank, spendLowStamina: boolean): boolean {
  return !spendLowStamina && staminaRatio(tank) < STAMINA_CONSERVE_RATIO;
}

function staminaRatio(tank: Tank): number {
  return tank.maxStamina > 0 ? tank.stamina / tank.maxStamina : 0;
}

function trackPair(left: number, right: number): TankCommand {
  return {
    leftTrack: toTrack(left),
    rightTrack: toTrack(right)
  };
}

function toTrack(value: number): TrackPower {
  if (value > 0) {
    return TRACKS[2];
  }
  if (value < 0) {
    return TRACKS[0];
  }
  return TRACKS[1];
}

function clampPoint(point: Vec2): Vec2 {
  return {
    x: clamp(point.x, FIELD.tankRadius, FIELD.length - FIELD.tankRadius),
    y: clamp(point.y, FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
}

function unitVector(from: Vec2, to: Vec2, fallbackX: 1 | -1): Vec2 {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);

  if (length === 0) {
    return { x: fallbackX, y: 0 };
  }

  return {
    x: dx / length,
    y: dy / length
  };
}

function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function dot(a: Vec2, b: Vec2): number {
  return a.x * b.x + a.y * b.y;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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
