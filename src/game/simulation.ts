import {
  FIELD,
  type Ball,
  type GameState,
  type Tank,
  type Vec2,
  goalTeamForSide,
  isInsideGoalMouth,
  resetForKickoff
} from './model';
import { type CommandMap, type TankCommand, sanitizeCommand } from './strategy';

const EPSILON = 0.0001;
const COLLISION_ITERATIONS = 16;
const POSITION_SLOP = 0.005;
const TANK_TANK_RESTITUTION = 0.04;
const TANK_BALL_RESTITUTION = 0.18;
const TANK_WALL_RESTITUTION = 0.08;
const BALL_WALL_RESTITUTION = FIELD.wallRestitution;
const TANK_IDLE_ANGULAR_FRICTION = 60;
const TANK_DRIVE_ACCELERATION = 3600;
const TANK_LATERAL_GRIP_ACCELERATION = 3600;
const TANK_TRACK_ANGULAR_ACCELERATION = 90;
const FULL_POWER_STAMINA_RATIO = 0.5;

export function stepGame(state: GameState, commands: CommandMap, dt: number): void {
  state.lastGoal = null;

  for (const tank of state.tanks) {
    integrateTank(tank, sanitizeCommand(commands[tank.id]), dt);
  }

  integrateBall(state.ball, dt);
  if (enforceBallBounds(state, true)) {
    finishStep(state, dt);
    return;
  }

  for (let i = 0; i < COLLISION_ITERATIONS; i += 1) {
    for (const tank of state.tanks) {
      enforceTankBounds(tank);
    }
    resolveTankTankCollisions(state.tanks);
    resolveTankBallCollisions(state);
    if (enforceBallBounds(state, true)) {
      finishStep(state, dt);
      return;
    }
  }

  finishStep(state, dt);
}

function finishStep(state: GameState, dt: number): void {
  state.time += dt;
  state.frame += 1;
}

function integrateTank(tank: Tank, command: TankCommand, dt: number): void {
  const activeTracks = Math.abs(command.leftTrack) + Math.abs(command.rightTrack);
  const powerScale = activeTracks === 0 ? 0 : trackPowerScale(tank);

  if (activeTracks > 0) {
    tank.stamina = Math.max(
      0,
      tank.stamina - activeTracks * FIELD.tankStaminaDrainPerTrackSecond * powerScale * dt
    );
    applyPoweredTrackMotion(tank, command, powerScale, dt);
  } else {
    tank.stamina = Math.min(
      tank.maxStamina,
      tank.stamina + FIELD.tankStaminaRecoveryPerSecond * dt
    );
    applyGroundFriction(tank, dt);
  }

  limitTankVelocity(tank);

  tank.angle = normalizeAngle(tank.angle + tank.angularVelocity * dt);
  tank.position.x += tank.velocity.x * dt;
  tank.position.y += tank.velocity.y * dt;
}

function trackPowerScale(tank: Tank): number {
  const ratio = tank.maxStamina > 0 ? tank.stamina / tank.maxStamina : 0;
  if (ratio >= FULL_POWER_STAMINA_RATIO) {
    return 1;
  }

  return clamp01(ratio / FULL_POWER_STAMINA_RATIO);
}

function integrateBall(ball: Ball, dt: number): void {
  ball.position.x += ball.velocity.x * dt;
  ball.position.y += ball.velocity.y * dt;
  applyBallFriction(ball.velocity, dt);
}

function applyPoweredTrackMotion(
  tank: Tank,
  command: TankCommand,
  staminaScale: number,
  dt: number
): void {
  const forward = { x: Math.cos(tank.angle), y: Math.sin(tank.angle) };
  const lateral = { x: -forward.y, y: forward.x };
  const currentForwardSpeed = tank.velocity.x * forward.x + tank.velocity.y * forward.y;
  const currentLateralSpeed = tank.velocity.x * lateral.x + tank.velocity.y * lateral.y;
  const desiredForwardSpeed =
    ((command.leftTrack + command.rightTrack) / 2) * tank.maxTrackSpeed * staminaScale;
  const desiredAngularVelocity =
    ((command.leftTrack - command.rightTrack) * tank.maxTrackSpeed / (2 * tank.width)) *
    staminaScale;

  const nextForwardSpeed = moveToward(
    currentForwardSpeed,
    desiredForwardSpeed,
    TANK_DRIVE_ACCELERATION * dt
  );
  const nextLateralSpeed = moveToward(
    currentLateralSpeed,
    0,
    TANK_LATERAL_GRIP_ACCELERATION * dt
  );

  tank.velocity.x = forward.x * nextForwardSpeed + lateral.x * nextLateralSpeed;
  tank.velocity.y = forward.y * nextForwardSpeed + lateral.y * nextLateralSpeed;
  tank.angularVelocity = moveToward(
    tank.angularVelocity,
    desiredAngularVelocity,
    TANK_TRACK_ANGULAR_ACCELERATION * dt
  );
}

function applyGroundFriction(tank: Tank, dt: number): void {
  const speed = Math.hypot(tank.velocity.x, tank.velocity.y);
  const friction = tank.staticFriction * FIELD.gravity;
  const deceleration = friction * dt;

  if (speed <= deceleration) {
    tank.velocity.x = 0;
    tank.velocity.y = 0;
  } else if (speed > 0) {
    const scale = (speed - deceleration) / speed;
    tank.velocity.x *= scale;
    tank.velocity.y *= scale;
  }

  const angularDeceleration = TANK_IDLE_ANGULAR_FRICTION * dt;
  if (Math.abs(tank.angularVelocity) <= angularDeceleration) {
    tank.angularVelocity = 0;
  } else {
    tank.angularVelocity -= Math.sign(tank.angularVelocity) * angularDeceleration;
  }
}

function limitTankVelocity(tank: Tank): void {
  const speed = Math.hypot(tank.velocity.x, tank.velocity.y);
  if (speed > tank.maxTrackSpeed) {
    const scale = tank.maxTrackSpeed / speed;
    tank.velocity.x *= scale;
    tank.velocity.y *= scale;
  }

  const maxAngularVelocity = tank.maxTrackSpeed * 2 / tank.trackWidth;
  tank.angularVelocity = clamp(tank.angularVelocity, -maxAngularVelocity, maxAngularVelocity);
}

function enforceTankBounds(tank: Tank): void {
  const bounds = tankWorldBounds(tank);

  if (bounds.minX < 0) {
    tank.position.x -= bounds.minX;
    tank.velocity.x = Math.max(0, -tank.velocity.x * TANK_WALL_RESTITUTION);
  } else if (bounds.maxX > FIELD.length) {
    tank.position.x -= bounds.maxX - FIELD.length;
    tank.velocity.x = Math.min(0, -tank.velocity.x * TANK_WALL_RESTITUTION);
  }

  const yBounds = tankWorldBounds(tank);
  if (yBounds.minY < 0) {
    tank.position.y -= yBounds.minY;
    tank.velocity.y = Math.max(0, -tank.velocity.y * TANK_WALL_RESTITUTION);
  } else if (yBounds.maxY > FIELD.width) {
    tank.position.y -= yBounds.maxY - FIELD.width;
    tank.velocity.y = Math.min(0, -tank.velocity.y * TANK_WALL_RESTITUTION);
  }
}

function enforceBallBounds(state: GameState, allowScore: boolean): boolean {
  const ball = state.ball;

  if (ball.position.x <= ball.radius) {
    if (allowScore && isInsideGoalMouth(ball.position.y)) {
      scoreGoal(state, goalTeamForSide('left'));
      return true;
    }
    ball.position.x = ball.radius;
    ball.velocity.x = Math.abs(ball.velocity.x) * BALL_WALL_RESTITUTION;
  }

  if (ball.position.x >= FIELD.length - ball.radius) {
    if (allowScore && isInsideGoalMouth(ball.position.y)) {
      scoreGoal(state, goalTeamForSide('right'));
      return true;
    }
    ball.position.x = FIELD.length - ball.radius;
    ball.velocity.x = -Math.abs(ball.velocity.x) * BALL_WALL_RESTITUTION;
  }

  if (ball.position.y <= ball.radius) {
    ball.position.y = ball.radius;
    ball.velocity.y = Math.abs(ball.velocity.y) * BALL_WALL_RESTITUTION;
  }

  if (ball.position.y >= FIELD.width - ball.radius) {
    ball.position.y = FIELD.width - ball.radius;
    ball.velocity.y = -Math.abs(ball.velocity.y) * BALL_WALL_RESTITUTION;
  }

  return false;
}

function resolveTankTankCollisions(tanks: Tank[]): void {
  for (let i = 0; i < tanks.length; i += 1) {
    for (let j = i + 1; j < tanks.length; j += 1) {
      const a = tanks[i];
      const b = tanks[j];
      const partsA = tankWorldConvexParts(a);
      const partsB = tankWorldConvexParts(b);

      const invMassA = 1 / a.mass;
      const invMassB = 1 / b.mass;
      const invMassSum = invMassA + invMassB;

      for (const partA of partsA) {
        for (const partB of partsB) {
          const collision = convexPolygonCollision(partA, partB);
          if (!collision) {
            continue;
          }

          const correction = Math.max(0, collision.penetration - POSITION_SLOP) / invMassSum;
          a.position.x -= collision.normal.x * correction * invMassA;
          a.position.y -= collision.normal.y * correction * invMassA;
          b.position.x += collision.normal.x * correction * invMassB;
          b.position.y += collision.normal.y * correction * invMassB;

          applyImpulse(
            a.velocity,
            b.velocity,
            collision.normal,
            invMassA,
            invMassB,
            TANK_TANK_RESTITUTION
          );
        }
      }
    }
  }
}

function resolveTankBallCollisions(state: GameState): void {
  const ball = state.ball;

  for (const tank of state.tanks) {
    const collision = tankBallCollision(tank, ball.position, ball.radius);
    if (!collision) {
      continue;
    }

    separateTankAndBall(state, tank, collision.normal, collision.penetration);
    const correctedCollision = tankBallCollision(tank, ball.position, ball.radius);
    applyImpulse(
      tank.velocity,
      ball.velocity,
      correctedCollision?.normal ?? collision.normal,
      1 / tank.mass,
      1 / ball.mass,
      TANK_BALL_RESTITUTION
    );
  }
}

function separateTankAndBall(
  state: GameState,
  tank: Tank,
  normal: { x: number; y: number },
  overlap: number
): void {
  const ball = state.ball;
  const invTank = 1 / tank.mass;
  const invBall = 1 / ball.mass;
  const invSum = invTank + invBall;
  const correction = Math.max(0, overlap - POSITION_SLOP) / invSum;

  tank.position.x -= normal.x * correction * invTank;
  tank.position.y -= normal.y * correction * invTank;
  ball.position.x += normal.x * correction * invBall;
  ball.position.y += normal.y * correction * invBall;
  enforceTankBounds(tank);
  enforceBallBounds(state, false);

  const residual = tankBallCollision(tank, ball.position, ball.radius);
  if (residual && residual.penetration > POSITION_SLOP) {
    tank.position.x -= residual.normal.x * residual.penetration;
    tank.position.y -= residual.normal.y * residual.penetration;
    enforceTankBounds(tank);
  }

  const finalResidual = tankBallCollision(tank, ball.position, ball.radius);
  if (finalResidual && finalResidual.penetration > POSITION_SLOP) {
    ball.position.x += finalResidual.normal.x * finalResidual.penetration;
    ball.position.y += finalResidual.normal.y * finalResidual.penetration;
    enforceBallBounds(state, false);
  }
}

function applyImpulse(
  velocityA: Vec2,
  velocityB: Vec2,
  normal: { x: number; y: number },
  invMassA: number,
  invMassB: number,
  restitution: number
): void {
  const relativeVelocity = {
    x: velocityB.x - velocityA.x,
    y: velocityB.y - velocityA.y
  };
  const speedAlongNormal = relativeVelocity.x * normal.x + relativeVelocity.y * normal.y;

  if (speedAlongNormal > 0) {
    return;
  }

  const impulseMagnitude = -(1 + restitution) * speedAlongNormal / (invMassA + invMassB);
  const impulse = {
    x: impulseMagnitude * normal.x,
    y: impulseMagnitude * normal.y
  };

  velocityA.x -= impulse.x * invMassA;
  velocityA.y -= impulse.y * invMassA;
  velocityB.x += impulse.x * invMassB;
  velocityB.y += impulse.y * invMassB;
}

function convexPolygonCollision(a: Vec2[], b: Vec2[]): { normal: Vec2; penetration: number } | null {
  let bestAxis: Vec2 | null = null;
  let bestOverlap = Number.POSITIVE_INFINITY;
  const axes = [...polygonAxes(a), ...polygonAxes(b)];
  const centerA = polygonCentroid(a);
  const centerB = polygonCentroid(b);
  const centerDelta = {
    x: centerB.x - centerA.x,
    y: centerB.y - centerA.y
  };

  for (const axis of axes) {
    const projectionA = projectPoints(a, axis);
    const projectionB = projectPoints(b, axis);
    const overlap = Math.min(projectionA.max, projectionB.max) -
      Math.max(projectionA.min, projectionB.min);

    if (overlap <= 0) {
      return null;
    }

    if (overlap < bestOverlap) {
      const direction = centerDelta.x * axis.x + centerDelta.y * axis.y >= 0 ? 1 : -1;
      bestAxis = {
        x: axis.x * direction,
        y: axis.y * direction
      };
      bestOverlap = overlap;
    }
  }

  if (!bestAxis) {
    return null;
  }

  return {
    normal: bestAxis,
    penetration: bestOverlap
  };
}

function tankBallCollision(
  tank: Tank,
  ballPosition: Vec2,
  ballRadius: number
): { normal: Vec2; penetration: number } | null {
  let best: { normal: Vec2; penetration: number } | null = null;

  for (const part of tankWorldConvexParts(tank)) {
    const collision = circlePolygonCollision(ballPosition, ballRadius, part);
    if (!collision) {
      continue;
    }
    if (!best || collision.penetration > best.penetration) {
      best = collision;
    }
  }

  return best;
}

function tankWorldConvexParts(tank: Tank): Vec2[][] {
  return tankLocalConvexParts(tank).map((part) =>
    part.map((point) => tankLocalPointToWorld(tank, point))
  );
}

function tankLocalConvexParts(tank: Tank): Vec2[][] {
  const halfLength = tank.length / 2;
  const halfWidth = tank.width / 2;
  const nose = tank.noseLength;

  return [
    [
      { x: -halfLength, y: -halfWidth },
      { x: halfLength, y: -halfWidth },
      { x: halfLength, y: halfWidth },
      { x: -halfLength, y: halfWidth }
    ],
    [
      { x: halfLength, y: -halfWidth },
      { x: halfLength + nose, y: -halfWidth },
      { x: halfLength, y: 0 }
    ],
    [
      { x: halfLength, y: 0 },
      { x: halfLength + nose, y: halfWidth },
      { x: halfLength, y: halfWidth }
    ]
  ];
}

function tankLocalHullPoints(tank: Tank): Vec2[] {
  const halfLength = tank.length / 2;
  const halfWidth = tank.width / 2;
  const nose = tank.noseLength;

  return [
    { x: -halfLength, y: -halfWidth },
    { x: halfLength, y: -halfWidth },
    { x: halfLength + nose, y: -halfWidth },
    { x: halfLength, y: 0 },
    { x: halfLength + nose, y: halfWidth },
    { x: halfLength, y: halfWidth },
    { x: -halfLength, y: halfWidth }
  ];
}

function tankWorldBounds(tank: Tank): {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
} {
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const point of tankLocalHullPoints(tank)) {
    const world = tankLocalPointToWorld(tank, point);
    minX = Math.min(minX, world.x);
    maxX = Math.max(maxX, world.x);
    minY = Math.min(minY, world.y);
    maxY = Math.max(maxY, world.y);
  }

  return { minX, maxX, minY, maxY };
}

function tankLocalPointToWorld(tank: Tank, point: Vec2): Vec2 {
  return {
    x: tank.position.x + Math.cos(tank.angle) * point.x - Math.sin(tank.angle) * point.y,
    y: tank.position.y + Math.sin(tank.angle) * point.x + Math.cos(tank.angle) * point.y
  };
}

function circlePolygonCollision(
  center: Vec2,
  radius: number,
  polygon: Vec2[]
): { normal: Vec2; penetration: number } | null {
  let closestPoint = polygon[0];
  let closestDistance = Number.POSITIVE_INFINITY;
  let closestEdgeNormal = { x: 1, y: 0 };
  let closestInsideDistance = Number.POSITIVE_INFINITY;
  let closestInsideNormal = { x: 1, y: 0 };
  let inside = true;

  for (let index = 0; index < polygon.length; index += 1) {
    const start = polygon[index];
    const end = polygon[(index + 1) % polygon.length];
    const edge = { x: end.x - start.x, y: end.y - start.y };
    const edgeLength = Math.hypot(edge.x, edge.y) || 1;
    const outward = { x: edge.y / edgeLength, y: -edge.x / edgeLength };
    const signedDistance = (center.x - start.x) * outward.x + (center.y - start.y) * outward.y;

    if (signedDistance > 0) {
      inside = false;
    }

    const insideDistance = Math.abs(signedDistance);
    if (insideDistance < closestInsideDistance) {
      closestInsideDistance = insideDistance;
      closestInsideNormal = outward;
    }

    const point = closestPointOnSegment(center, start, end);
    const pointDistance = distance(center, point);
    if (pointDistance < closestDistance) {
      closestDistance = pointDistance;
      closestPoint = point;
      closestEdgeNormal = outward;
    }
  }

  if (inside) {
    return {
      normal: closestInsideNormal,
      penetration: radius + closestInsideDistance
    };
  }

  if (closestDistance >= radius) {
    return null;
  }

  if (closestDistance <= EPSILON) {
    return {
      normal: closestEdgeNormal,
      penetration: radius
    };
  }

  return {
    normal: {
      x: (center.x - closestPoint.x) / closestDistance,
      y: (center.y - closestPoint.y) / closestDistance
    },
    penetration: radius - closestDistance
  };
}

function closestPointOnSegment(point: Vec2, start: Vec2, end: Vec2): Vec2 {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy || 1;
  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared, 0, 1);

  return {
    x: start.x + dx * t,
    y: start.y + dy * t
  };
}

function polygonAxes(points: Vec2[]): Vec2[] {
  return points.map((start, index) => {
    const end = points[(index + 1) % points.length];
    const edge = { x: end.x - start.x, y: end.y - start.y };
    const length = Math.hypot(edge.x, edge.y) || 1;
    return { x: edge.y / length, y: -edge.x / length };
  });
}

function projectPoints(points: Vec2[], axis: Vec2): { min: number; max: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    const value = point.x * axis.x + point.y * axis.y;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { min, max };
}

function polygonCentroid(points: Vec2[]): Vec2 {
  const sum = points.reduce(
    (total, point) => ({ x: total.x + point.x, y: total.y + point.y }),
    { x: 0, y: 0 }
  );

  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
}

function distance(a: Vec2, b: Vec2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function scoreGoal(state: GameState, team: 'red' | 'blue'): void {
  state.score[team] += 1;
  state.lastGoal = { team, frame: state.frame, time: state.time };
  resetForKickoff(state);
}

function applyBallFriction(velocity: Vec2, dt: number): void {
  const damping = FIELD.ballDampingPerSecond ** dt;
  velocity.x *= damping;
  velocity.y *= damping;

  if (Math.hypot(velocity.x, velocity.y) < 4) {
    velocity.x = 0;
    velocity.y = 0;
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function moveToward(value: number, target: number, maxDelta: number): number {
  const delta = target - value;
  if (Math.abs(delta) <= maxDelta) {
    return target;
  }

  return value + Math.sign(delta) * maxDelta;
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
