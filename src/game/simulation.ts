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

/**
 * The collision geometry below is written into reusable buffers instead of fresh
 * objects, and every expression keeps the exact operand order of the original
 * allocating implementation, so trajectories stay bit-identical.
 * `tests/simulationFingerprint.test.ts` pins that guarantee.
 *
 * This is not reentrant, which is safe because `stepGame` never invokes a
 * strategy: match code decides commands first, then steps the world.
 */

const TANK_PART_COUNT = 3;
const TANK_VERTEX_COUNT = 10;
const TANK_PART_START = [0, 4, 7] as const;
const TANK_PART_SIZE = [4, 3, 3] as const;
const TANK_HULL_POINT_COUNT = 7;

/**
 * Separation guard for the bounding-box rejections. World coordinates stay
 * under ~1e3 so double-precision projection error is around 1e-13; requiring a
 * 1e-6 gap keeps an early exit from ever deciding a marginal contact that the
 * full separating-axis test would have resolved differently.
 */
const BOUNDS_REJECT_GUARD = 1e-6;

const tankVerticesA = new Float64Array(TANK_VERTEX_COUNT * 2);
const tankVerticesB = new Float64Array(TANK_VERTEX_COUNT * 2);
const tankPartBoundsA = new Float64Array(TANK_PART_COUNT * 4);
const tankPartBoundsB = new Float64Array(TANK_PART_COUNT * 4);
const tankHullLocals = new Float64Array(TANK_HULL_POINT_COUNT * 2);

let hitNormalX = 0;
let hitNormalY = 0;
let hitPenetration = 0;

let hullMinX = 0;
let hullMaxX = 0;
let hullMinY = 0;
let hullMaxY = 0;

/**
 * Two-entry angle cache. A hit returns exactly the value `Math.cos`/`Math.sin`
 * would return for the same angle, so results are unchanged; it only removes the
 * thousands of duplicate trig calls the per-vertex transform used to make.
 */
let trigAngle0 = Number.NaN;
let trigCos0 = 0;
let trigSin0 = 0;
let trigAngle1 = Number.NaN;
let trigCos1 = 0;
let trigSin1 = 0;
let trigCos = 0;
let trigSin = 0;

function loadTrig(angle: number): void {
  if (angle === trigAngle0) {
    trigCos = trigCos0;
    trigSin = trigSin0;
    return;
  }
  if (angle === trigAngle1) {
    trigCos = trigCos1;
    trigSin = trigSin1;
    return;
  }

  trigAngle1 = trigAngle0;
  trigCos1 = trigCos0;
  trigSin1 = trigSin0;
  trigAngle0 = angle;
  trigCos0 = Math.cos(angle);
  trigSin0 = Math.sin(angle);
  trigCos = trigCos0;
  trigSin = trigSin0;
}

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
  loadTankHullBounds(tank);

  if (hullMinX < 0) {
    tank.position.x -= hullMinX;
    tank.velocity.x = Math.max(0, -tank.velocity.x * TANK_WALL_RESTITUTION);
  } else if (hullMaxX > FIELD.length) {
    tank.position.x -= hullMaxX - FIELD.length;
    tank.velocity.x = Math.min(0, -tank.velocity.x * TANK_WALL_RESTITUTION);
  }

  // The horizontal correction above only moves `position.x`, so the vertical
  // extents are unchanged and can be reused instead of recomputed.
  if (hullMinY < 0) {
    tank.position.y -= hullMinY;
    tank.velocity.y = Math.max(0, -tank.velocity.y * TANK_WALL_RESTITUTION);
  } else if (hullMaxY > FIELD.width) {
    tank.position.y -= hullMaxY - FIELD.width;
    tank.velocity.y = Math.min(0, -tank.velocity.y * TANK_WALL_RESTITUTION);
  }
}

function loadTankHullBounds(tank: Tank): void {
  const halfLength = tank.length / 2;
  const halfWidth = tank.width / 2;
  const nose = tank.noseLength;

  tankHullLocals[0] = -halfLength;
  tankHullLocals[1] = -halfWidth;
  tankHullLocals[2] = halfLength;
  tankHullLocals[3] = -halfWidth;
  tankHullLocals[4] = halfLength + nose;
  tankHullLocals[5] = -halfWidth;
  tankHullLocals[6] = halfLength;
  tankHullLocals[7] = 0;
  tankHullLocals[8] = halfLength + nose;
  tankHullLocals[9] = halfWidth;
  tankHullLocals[10] = halfLength;
  tankHullLocals[11] = halfWidth;
  tankHullLocals[12] = -halfLength;
  tankHullLocals[13] = halfWidth;

  loadTrig(tank.angle);
  const cos = trigCos;
  const sin = trigSin;
  const originX = tank.position.x;
  const originY = tank.position.y;

  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < TANK_HULL_POINT_COUNT; index += 1) {
    const localX = tankHullLocals[index * 2];
    const localY = tankHullLocals[index * 2 + 1];
    const worldX = originX + cos * localX - sin * localY;
    const worldY = originY + sin * localX + cos * localY;
    minX = Math.min(minX, worldX);
    maxX = Math.max(maxX, worldX);
    minY = Math.min(minY, worldY);
    maxY = Math.max(maxY, worldY);
  }

  hullMinX = minX;
  hullMaxX = maxX;
  hullMinY = minY;
  hullMaxY = maxY;
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
      // Both hulls are captured once per pair. The original implementation also
      // built its vertex lists before resolving any part pair, so later part
      // pairs intentionally test against pre-correction geometry.
      writeTankGeometry(a, tankVerticesA, tankPartBoundsA);
      writeTankGeometry(b, tankVerticesB, tankPartBoundsB);

      const invMassA = 1 / a.mass;
      const invMassB = 1 / b.mass;
      const invMassSum = invMassA + invMassB;

      for (let partA = 0; partA < TANK_PART_COUNT; partA += 1) {
        for (let partB = 0; partB < TANK_PART_COUNT; partB += 1) {
          if (partBoundsSeparated(tankPartBoundsA, partA, tankPartBoundsB, partB)) {
            continue;
          }
          if (!satPartCollision(
            tankVerticesA,
            TANK_PART_START[partA],
            TANK_PART_SIZE[partA],
            tankVerticesB,
            TANK_PART_START[partB],
            TANK_PART_SIZE[partB]
          )) {
            continue;
          }

          const normalX = hitNormalX;
          const normalY = hitNormalY;
          const correction = Math.max(0, hitPenetration - POSITION_SLOP) / invMassSum;
          a.position.x -= normalX * correction * invMassA;
          a.position.y -= normalY * correction * invMassA;
          b.position.x += normalX * correction * invMassB;
          b.position.y += normalY * correction * invMassB;

          applyImpulse(
            a.velocity,
            b.velocity,
            normalX,
            normalY,
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
    if (!tankBallCollision(tank, ball.position.x, ball.position.y, ball.radius)) {
      continue;
    }

    const normalX = hitNormalX;
    const normalY = hitNormalY;
    const penetration = hitPenetration;

    separateTankAndBall(state, tank, normalX, normalY, penetration);
    const corrected = tankBallCollision(tank, ball.position.x, ball.position.y, ball.radius);
    applyImpulse(
      tank.velocity,
      ball.velocity,
      corrected ? hitNormalX : normalX,
      corrected ? hitNormalY : normalY,
      1 / tank.mass,
      1 / ball.mass,
      TANK_BALL_RESTITUTION
    );
  }
}

function separateTankAndBall(
  state: GameState,
  tank: Tank,
  normalX: number,
  normalY: number,
  overlap: number
): void {
  const ball = state.ball;
  const invTank = 1 / tank.mass;
  const invBall = 1 / ball.mass;
  const invSum = invTank + invBall;
  const correction = Math.max(0, overlap - POSITION_SLOP) / invSum;

  tank.position.x -= normalX * correction * invTank;
  tank.position.y -= normalY * correction * invTank;
  ball.position.x += normalX * correction * invBall;
  ball.position.y += normalY * correction * invBall;
  enforceTankBounds(tank);
  enforceBallBounds(state, false);

  if (
    tankBallCollision(tank, ball.position.x, ball.position.y, ball.radius) &&
    hitPenetration > POSITION_SLOP
  ) {
    tank.position.x -= hitNormalX * hitPenetration;
    tank.position.y -= hitNormalY * hitPenetration;
    enforceTankBounds(tank);
  }

  if (
    tankBallCollision(tank, ball.position.x, ball.position.y, ball.radius) &&
    hitPenetration > POSITION_SLOP
  ) {
    ball.position.x += hitNormalX * hitPenetration;
    ball.position.y += hitNormalY * hitPenetration;
    enforceBallBounds(state, false);
  }
}

function applyImpulse(
  velocityA: Vec2,
  velocityB: Vec2,
  normalX: number,
  normalY: number,
  invMassA: number,
  invMassB: number,
  restitution: number
): void {
  const relativeVelocityX = velocityB.x - velocityA.x;
  const relativeVelocityY = velocityB.y - velocityA.y;
  const speedAlongNormal = relativeVelocityX * normalX + relativeVelocityY * normalY;

  if (speedAlongNormal > 0) {
    return;
  }

  const impulseMagnitude = -(1 + restitution) * speedAlongNormal / (invMassA + invMassB);
  const impulseX = impulseMagnitude * normalX;
  const impulseY = impulseMagnitude * normalY;

  velocityA.x -= impulseX * invMassA;
  velocityA.y -= impulseY * invMassA;
  velocityB.x += impulseX * invMassB;
  velocityB.y += impulseY * invMassB;
}

/**
 * Writes the tank's three convex parts into `vertices` as interleaved world
 * coordinates and records a per-part bounding box for cheap rejection. Vertex
 * order matches the original part list exactly, because the separating-axis
 * result depends on edge order.
 */
function writeTankGeometry(tank: Tank, vertices: Float64Array, partBounds: Float64Array): void {
  const halfLength = tank.length / 2;
  const halfWidth = tank.width / 2;
  const nose = tank.noseLength;

  loadTrig(tank.angle);
  const cos = trigCos;
  const sin = trigSin;
  const originX = tank.position.x;
  const originY = tank.position.y;

  writeVertex(vertices, 0, originX, originY, cos, sin, -halfLength, -halfWidth);
  writeVertex(vertices, 1, originX, originY, cos, sin, halfLength, -halfWidth);
  writeVertex(vertices, 2, originX, originY, cos, sin, halfLength, halfWidth);
  writeVertex(vertices, 3, originX, originY, cos, sin, -halfLength, halfWidth);

  writeVertex(vertices, 4, originX, originY, cos, sin, halfLength, -halfWidth);
  writeVertex(vertices, 5, originX, originY, cos, sin, halfLength + nose, -halfWidth);
  writeVertex(vertices, 6, originX, originY, cos, sin, halfLength, 0);

  writeVertex(vertices, 7, originX, originY, cos, sin, halfLength, 0);
  writeVertex(vertices, 8, originX, originY, cos, sin, halfLength + nose, halfWidth);
  writeVertex(vertices, 9, originX, originY, cos, sin, halfLength, halfWidth);

  for (let part = 0; part < TANK_PART_COUNT; part += 1) {
    const start = TANK_PART_START[part];
    const size = TANK_PART_SIZE[part];
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < size; index += 1) {
      const x = vertices[(start + index) * 2];
      const y = vertices[(start + index) * 2 + 1];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
    partBounds[part * 4] = minX;
    partBounds[part * 4 + 1] = minY;
    partBounds[part * 4 + 2] = maxX;
    partBounds[part * 4 + 3] = maxY;
  }
}

function writeVertex(
  vertices: Float64Array,
  index: number,
  originX: number,
  originY: number,
  cos: number,
  sin: number,
  localX: number,
  localY: number
): void {
  vertices[index * 2] = originX + cos * localX - sin * localY;
  vertices[index * 2 + 1] = originY + sin * localX + cos * localY;
}

/**
 * Convex parts separated by more than the guard band cannot intersect, so the
 * separating-axis test would report no collision for them.
 */
function partBoundsSeparated(
  boundsA: Float64Array,
  partA: number,
  boundsB: Float64Array,
  partB: number
): boolean {
  const a = partA * 4;
  const b = partB * 4;
  return boundsA[a + 2] + BOUNDS_REJECT_GUARD < boundsB[b] ||
    boundsB[b + 2] + BOUNDS_REJECT_GUARD < boundsA[a] ||
    boundsA[a + 3] + BOUNDS_REJECT_GUARD < boundsB[b + 1] ||
    boundsB[b + 3] + BOUNDS_REJECT_GUARD < boundsA[a + 1];
}

function satPartCollision(
  verticesA: Float64Array,
  startA: number,
  countA: number,
  verticesB: Float64Array,
  startB: number,
  countB: number
): boolean {
  let bestAxisX = 0;
  let bestAxisY = 0;
  let hasBestAxis = false;
  let bestOverlap = Number.POSITIVE_INFINITY;

  let centerSumAX = 0;
  let centerSumAY = 0;
  for (let index = 0; index < countA; index += 1) {
    centerSumAX += verticesA[(startA + index) * 2];
    centerSumAY += verticesA[(startA + index) * 2 + 1];
  }
  let centerSumBX = 0;
  let centerSumBY = 0;
  for (let index = 0; index < countB; index += 1) {
    centerSumBX += verticesB[(startB + index) * 2];
    centerSumBY += verticesB[(startB + index) * 2 + 1];
  }
  const centerDeltaX = centerSumBX / countB - centerSumAX / countA;
  const centerDeltaY = centerSumBY / countB - centerSumAY / countA;

  const axisCount = countA + countB;
  for (let axisIndex = 0; axisIndex < axisCount; axisIndex += 1) {
    const fromA = axisIndex < countA;
    const source = fromA ? verticesA : verticesB;
    const sourceStart = fromA ? startA : startB;
    const sourceCount = fromA ? countA : countB;
    const edgeIndex = fromA ? axisIndex : axisIndex - countA;
    const edgeStart = (sourceStart + edgeIndex) * 2;
    const edgeEnd = (sourceStart + ((edgeIndex + 1) % sourceCount)) * 2;
    const edgeX = source[edgeEnd] - source[edgeStart];
    const edgeY = source[edgeEnd + 1] - source[edgeStart + 1];
    const edgeLength = Math.hypot(edgeX, edgeY) || 1;
    const axisX = edgeY / edgeLength;
    const axisY = -edgeX / edgeLength;

    let minA = Number.POSITIVE_INFINITY;
    let maxA = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < countA; index += 1) {
      const value = verticesA[(startA + index) * 2] * axisX +
        verticesA[(startA + index) * 2 + 1] * axisY;
      minA = Math.min(minA, value);
      maxA = Math.max(maxA, value);
    }
    let minB = Number.POSITIVE_INFINITY;
    let maxB = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < countB; index += 1) {
      const value = verticesB[(startB + index) * 2] * axisX +
        verticesB[(startB + index) * 2 + 1] * axisY;
      minB = Math.min(minB, value);
      maxB = Math.max(maxB, value);
    }

    const overlap = Math.min(maxA, maxB) - Math.max(minA, minB);
    if (overlap <= 0) {
      return false;
    }

    if (overlap < bestOverlap) {
      const direction = centerDeltaX * axisX + centerDeltaY * axisY >= 0 ? 1 : -1;
      bestAxisX = axisX * direction;
      bestAxisY = axisY * direction;
      hasBestAxis = true;
      bestOverlap = overlap;
    }
  }

  if (!hasBestAxis) {
    return false;
  }

  hitNormalX = bestAxisX;
  hitNormalY = bestAxisY;
  hitPenetration = bestOverlap;
  return true;
}

function tankBallCollision(
  tank: Tank,
  ballX: number,
  ballY: number,
  ballRadius: number
): boolean {
  writeTankGeometry(tank, tankVerticesA, tankPartBoundsA);

  let found = false;
  let bestNormalX = 0;
  let bestNormalY = 0;
  let bestPenetration = 0;

  for (let part = 0; part < TANK_PART_COUNT; part += 1) {
    if (circleBoundsSeparated(tankPartBoundsA, part, ballX, ballY, ballRadius)) {
      continue;
    }
    if (!circlePartCollision(
      tankVerticesA,
      TANK_PART_START[part],
      TANK_PART_SIZE[part],
      ballX,
      ballY,
      ballRadius
    )) {
      continue;
    }
    if (!found || hitPenetration > bestPenetration) {
      found = true;
      bestNormalX = hitNormalX;
      bestNormalY = hitNormalY;
      bestPenetration = hitPenetration;
    }
  }

  if (!found) {
    return false;
  }

  hitNormalX = bestNormalX;
  hitNormalY = bestNormalY;
  hitPenetration = bestPenetration;
  return true;
}

/**
 * A circle whose expanded box clears the part's box is farther than `radius`
 * from every edge and is outside the part, which is exactly the case the full
 * circle/polygon test rejects.
 */
function circleBoundsSeparated(
  partBounds: Float64Array,
  part: number,
  centerX: number,
  centerY: number,
  radius: number
): boolean {
  const base = part * 4;
  return centerX - radius > partBounds[base + 2] + BOUNDS_REJECT_GUARD ||
    centerX + radius < partBounds[base] - BOUNDS_REJECT_GUARD ||
    centerY - radius > partBounds[base + 3] + BOUNDS_REJECT_GUARD ||
    centerY + radius < partBounds[base + 1] - BOUNDS_REJECT_GUARD;
}

function circlePartCollision(
  vertices: Float64Array,
  start: number,
  count: number,
  centerX: number,
  centerY: number,
  radius: number
): boolean {
  let closestPointX = vertices[start * 2];
  let closestPointY = vertices[start * 2 + 1];
  let closestDistance = Number.POSITIVE_INFINITY;
  let closestEdgeNormalX = 1;
  let closestEdgeNormalY = 0;
  let closestInsideDistance = Number.POSITIVE_INFINITY;
  let closestInsideNormalX = 1;
  let closestInsideNormalY = 0;
  let inside = true;

  for (let index = 0; index < count; index += 1) {
    const edgeStart = (start + index) * 2;
    const edgeEnd = (start + ((index + 1) % count)) * 2;
    const startX = vertices[edgeStart];
    const startY = vertices[edgeStart + 1];
    const edgeX = vertices[edgeEnd] - startX;
    const edgeY = vertices[edgeEnd + 1] - startY;
    const edgeLength = Math.hypot(edgeX, edgeY) || 1;
    const outwardX = edgeY / edgeLength;
    const outwardY = -edgeX / edgeLength;
    const signedDistance = (centerX - startX) * outwardX + (centerY - startY) * outwardY;

    if (signedDistance > 0) {
      inside = false;
    }

    const insideDistance = Math.abs(signedDistance);
    if (insideDistance < closestInsideDistance) {
      closestInsideDistance = insideDistance;
      closestInsideNormalX = outwardX;
      closestInsideNormalY = outwardY;
    }

    const lengthSquared = edgeX * edgeX + edgeY * edgeY || 1;
    const t = clamp(((centerX - startX) * edgeX + (centerY - startY) * edgeY) / lengthSquared, 0, 1);
    const pointX = startX + edgeX * t;
    const pointY = startY + edgeY * t;
    const pointDistance = Math.hypot(centerX - pointX, centerY - pointY);
    if (pointDistance < closestDistance) {
      closestDistance = pointDistance;
      closestPointX = pointX;
      closestPointY = pointY;
      closestEdgeNormalX = outwardX;
      closestEdgeNormalY = outwardY;
    }
  }

  if (inside) {
    hitNormalX = closestInsideNormalX;
    hitNormalY = closestInsideNormalY;
    hitPenetration = radius + closestInsideDistance;
    return true;
  }

  if (closestDistance >= radius) {
    return false;
  }

  if (closestDistance <= EPSILON) {
    hitNormalX = closestEdgeNormalX;
    hitNormalY = closestEdgeNormalY;
    hitPenetration = radius;
    return true;
  }

  hitNormalX = (centerX - closestPointX) / closestDistance;
  hitNormalY = (centerY - closestPointY) / closestDistance;
  hitPenetration = radius - closestDistance;
  return true;
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
