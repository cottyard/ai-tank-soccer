import { describe, expect, it } from 'vitest';
import { FIELD, TEAM_SIZE, createInitialState } from '../src/game/model';
import { stepGame } from '../src/game/simulation';
import { idleCommands } from '../src/game/strategy';

const EXPECTED_NOSE_LENGTH = FIELD.tankWidth / 2;

describe('tank soccer physics', () => {
  it('uses a standard rectangular field, one tank per team, and a large ball', () => {
    expect(FIELD.length / FIELD.width).toBeCloseTo(105 / 68, 3);
    expect(FIELD.ballRadius * 2).toBeCloseTo(FIELD.width / 10, 6);
    expect(TEAM_SIZE).toBe(1);
    expect(createInitialState().tanks).toHaveLength(TEAM_SIZE * 2);
  });

  it('sets each square tank body side to 1.5 ball diameters with a V nose in front', () => {
    const tank = createInitialState().tanks[0];
    const expectedTankSide = FIELD.ballRadius * 2 * 1.5;

    expect(FIELD.tankLength).toBeCloseTo(expectedTankSide, 6);
    expect(FIELD.tankWidth).toBeCloseTo(expectedTankSide, 6);
    expect(tank.length).toBeCloseTo(expectedTankSide, 6);
    expect(tank.width).toBeCloseTo(expectedTankSide, 6);
    expect(tank.radius).toBeGreaterThan(expectedTankSide / 2);
  });

  it('reflects the ball from non-goal walls without leaving the field', () => {
    const state = createInitialState();
    state.ball.position.x = FIELD.length / 2;
    state.ball.position.y = FIELD.ballRadius + 1;
    state.ball.velocity.y = -420;

    stepGame(state, idleCommands(), 1 / 30);

    expect(state.ball.position.y).toBeGreaterThanOrEqual(FIELD.ballRadius);
    expect(state.ball.velocity.y).toBeGreaterThan(0);
  });

  it('applies strong rolling friction to the ball', () => {
    const state = createInitialState();
    state.ball.velocity.x = 900;

    for (let i = 0; i < 30; i += 1) {
      stepGame(state, idleCommands(), 1 / 30);
    }

    expect(Math.abs(state.ball.velocity.x)).toBeLessThan(480);
  });

  it('moves and rotates tanks through independent left and right track commands', () => {
    const forward = createInitialState();
    const turn = createInitialState();
    const tankId = forward.tanks[0].id;

    stepGame(forward, { [tankId]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    stepGame(turn, { [tankId]: { leftTrack: -1, rightTrack: 1 } }, 1 / 30);

    expect(forward.tanks[0].position.x).toBeGreaterThan(turn.tanks[0].position.x);
    expect(Math.abs(turn.tanks[0].angle)).toBeGreaterThan(Math.abs(forward.tanks[0].angle));
  });

  it('reaches maximum forward speed within three frames after both tracks start', () => {
    const state = createInitialState();
    const tank = state.tanks[0];

    for (let i = 0; i < 3; i += 1) {
      stepGame(state, { [tank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    expect(Math.hypot(tank.velocity.x, tank.velocity.y)).toBeGreaterThanOrEqual(
      tank.maxTrackSpeed * 0.95
    );
  });

  it('turns clockwise when only the left track moves forward', () => {
    const state = createInitialState();
    const tank = state.tanks[0];

    for (let i = 0; i < 3; i += 1) {
      stepGame(state, { [tank.id]: { leftTrack: 1, rightTrack: 0 } }, 1 / 30);
    }

    expect(tank.angularVelocity).toBeGreaterThan(0);
    expect(tank.angle).toBeGreaterThan(0);
  });

  it('follows a tank-width turn radius when one track moves and the other is stopped', () => {
    const state = createInitialState();
    const tank = state.tanks[0];

    for (let i = 0; i < 6; i += 1) {
      stepGame(state, { [tank.id]: { leftTrack: 1, rightTrack: 0 } }, 1 / 30);
    }

    const speed = Math.hypot(tank.velocity.x, tank.velocity.y);
    const turnRadius = speed / Math.abs(tank.angularVelocity);

    expect(turnRadius).toBeGreaterThanOrEqual(tank.width * 0.9);
    expect(turnRadius).toBeLessThanOrEqual(tank.width * 1.1);
  });

  it('brakes idle tanks to a near-immediate stop', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.position = { x: FIELD.length / 2, y: FIELD.width / 2 - 160 };
    tank.velocity = { x: tank.maxTrackSpeed, y: 0 };
    tank.angularVelocity = tank.maxTrackSpeed * 2 / tank.trackWidth;
    state.ball.position = { x: FIELD.length / 2, y: FIELD.width / 2 + 120 };
    state.tanks[1].position = { x: FIELD.length - 160, y: FIELD.width - 120 };
    const startX = tank.position.x;

    for (let i = 0; i < 3; i += 1) {
      stepGame(state, idleCommands(), 1 / 30);
    }

    expect(Math.hypot(tank.velocity.x, tank.velocity.y)).toBeLessThanOrEqual(4);
    expect(Math.abs(tank.angularVelocity)).toBeLessThanOrEqual(0.1);
    expect(tank.position.x - startX).toBeLessThanOrEqual(12);
  });

  it('brakes residual rotation quickly when tracks are active but not reinforcing the turn', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.position = { x: FIELD.length / 2, y: FIELD.width / 2 - 160 };
    tank.angularVelocity = tank.maxTrackSpeed * 2 / tank.trackWidth;
    state.ball.position = { x: FIELD.length / 2, y: FIELD.width / 2 + 120 };

    for (let i = 0; i < 3; i += 1) {
      stepGame(state, { [tank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    expect(Math.abs(tank.angularVelocity)).toBeLessThanOrEqual(0.1);
  });

  it('spends tank stamina per moving track and restores it while resting', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    const start = tank.stamina;

    for (let i = 0; i < 60; i += 1) {
      stepGame(state, { [tank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    const afterDriving = tank.stamina;
    expect(afterDriving).toBeLessThan(start);
    expect(afterDriving).toBeGreaterThanOrEqual(0);

    for (let i = 0; i < 120; i += 1) {
      stepGame(state, idleCommands(), 1 / 30);
    }

    expect(tank.stamina).toBeGreaterThan(afterDriving);
    expect(tank.stamina).toBeLessThanOrEqual(tank.maxStamina);
  });

  it('spends stamina twice as fast when both tracks run instead of one', () => {
    const oneTrack = createInitialState();
    const twoTracks = createInitialState();
    const oneTank = oneTrack.tanks[0];
    const twoTank = twoTracks.tanks[0];

    for (let i = 0; i < 30; i += 1) {
      stepGame(oneTrack, { [oneTank.id]: { leftTrack: 1, rightTrack: 0 } }, 1 / 30);
      stepGame(twoTracks, { [twoTank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    const oneTrackSpent = oneTank.maxStamina - oneTank.stamina;
    const twoTrackSpent = twoTank.maxStamina - twoTank.stamina;

    expect(oneTrackSpent).toBeCloseTo(FIELD.tankStaminaDrainPerTrackSecond, 6);
    expect(twoTrackSpent).toBeCloseTo(oneTrackSpent * 2, 6);
  });

  it('scales track output below half stamina and never drains linearly to zero', () => {
    const halfPower = createInitialState();
    const fullPower = createInitialState();
    const depleted = createInitialState();
    const halfTank = halfPower.tanks[0];
    const fullTank = fullPower.tanks[0];
    const depletedTank = depleted.tanks[0];

    halfTank.stamina = halfTank.maxStamina * 0.25;
    halfTank.maxTrackSpeed = 200;
    fullTank.stamina = fullTank.maxStamina * 0.5;
    depletedTank.stamina = 0;

    stepGame(halfPower, { [halfTank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);

    expect(Math.hypot(halfTank.velocity.x, halfTank.velocity.y)).toBeCloseTo(
      halfTank.maxTrackSpeed * 0.5,
      6
    );

    for (let i = 0; i < 3; i += 1) {
      stepGame(fullPower, { [fullTank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
      stepGame(depleted, { [depletedTank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    expect(Math.hypot(fullTank.velocity.x, fullTank.velocity.y)).toBeGreaterThanOrEqual(
      fullTank.maxTrackSpeed * 0.95
    );
    expect(Math.hypot(depletedTank.velocity.x, depletedTank.velocity.y)).toBe(0);

    for (let i = 0; i < 30 * 30; i += 1) {
      stepGame(halfPower, { [halfTank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    expect(halfTank.stamina).toBeGreaterThan(0);
  });

  it('supports about a half-time active duty cycle for two running tracks', () => {
    const state = createInitialState();
    const tank = state.tanks[0];

    for (let i = 0; i < 150; i += 1) {
      stepGame(state, { [tank.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    expect(tank.stamina).toBeGreaterThan(0);
    expect(tank.stamina).toBeLessThan(25);

    for (let i = 0; i < 150; i += 1) {
      stepGame(state, idleCommands(), 1 / 30);
    }

    expect(tank.stamina).toBeGreaterThanOrEqual(tank.maxStamina - 1);
  });

  it('lets a powered tank push a resting tank through collision impulses', () => {
    const state = createInitialState();
    const mover = state.tanks[0];
    const pushed = state.tanks[1];
    mover.position = { x: 260, y: FIELD.width / 2 };
    mover.angle = 0;
    pushed.position = {
      x: mover.position.x + FIELD.tankLength + EXPECTED_NOSE_LENGTH * 2 - 8,
      y: FIELD.width / 2
    };
    pushed.angle = Math.PI;
    const before = pushed.position.x;

    for (let i = 0; i < 36; i += 1) {
      stepGame(state, { [mover.id]: { leftTrack: 1, rightTrack: 1 } }, 1 / 30);
    }

    expect(pushed.position.x).toBeGreaterThan(before + 10);
  });

  it('collides with the V nose triangles beyond the original square front', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.position = { x: 300, y: 300 };
    tank.angle = 0;
    tank.velocity = { x: 0, y: 0 };

    const localBall = {
      x: FIELD.tankLength / 2 + EXPECTED_NOSE_LENGTH - 8,
      y: -FIELD.tankWidth / 2 + 8
    };
    state.ball.position = localPointToWorld(tank.position, tank.angle, localBall);
    state.ball.velocity = { x: 0, y: 0 };
    state.tanks[1].position = { x: FIELD.length - 140, y: FIELD.width - 120 };
    const before = { ...state.ball.position };

    stepGame(state, idleCommands(), 1 / 30);

    expect(distance(state.ball.position, before)).toBeGreaterThan(2);
    expect(distanceFromTankBody(tank, state.ball.position)).toBeGreaterThanOrEqual(
      state.ball.radius - 0.01
    );
  });

  it('leaves the open center of the V nose as a real concave mouth', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.position = { x: 300, y: 300 };
    tank.angle = 0;
    tank.velocity = { x: 0, y: 0 };

    const localBall = {
      x: FIELD.tankLength / 2 + EXPECTED_NOSE_LENGTH,
      y: 0
    };
    state.ball.position = localPointToWorld(tank.position, tank.angle, localBall);
    state.ball.velocity = { x: 0, y: 0 };
    state.tanks[1].position = { x: FIELD.length - 140, y: FIELD.width - 120 };
    const before = { ...state.ball.position };

    stepGame(state, idleCommands(), 1 / 30);

    expect(distance(state.ball.position, before)).toBeLessThan(0.01);
  });

  it('keeps rotated V tank vertices inside the walls', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.position = { x: FIELD.tankRadius + 1, y: FIELD.tankRadius + 1 };
    tank.angle = Math.PI / 4;

    stepGame(state, idleCommands(), 1 / 30);

    for (const vertex of tankBodyVertices(tank)) {
      expect(vertex.x).toBeGreaterThanOrEqual(-0.01);
      expect(vertex.x).toBeLessThanOrEqual(FIELD.length + 0.01);
      expect(vertex.y).toBeGreaterThanOrEqual(-0.01);
      expect(vertex.y).toBeLessThanOrEqual(FIELD.width + 0.01);
    }
  });

  it('separates tank V nose overlaps using rendered V geometry', () => {
    const state = createInitialState();
    const a = state.tanks[0];
    const b = state.tanks[1];
    a.position = { x: 300, y: 300 };
    a.angle = 0;
    a.velocity = { x: 0, y: 0 };
    b.position = { x: 300 + FIELD.tankLength + EXPECTED_NOSE_LENGTH * 2 - 20, y: 300 };
    b.angle = Math.PI;
    b.velocity = { x: 0, y: 0 };

    stepGame(state, idleCommands(), 1 / 30);

    expect(tankBodiesOverlap(a, b)).toBe(false);
  });

  it('resolves corner squeezing without leaving the ball overlapped with either tank', () => {
    const state = createInitialState();
    state.ball.position = {
      x: FIELD.length - FIELD.ballRadius,
      y: FIELD.ballRadius
    };
    state.ball.velocity = { x: 0, y: 0 };

    state.tanks[0].position = {
      x: FIELD.length - FIELD.ballRadius - FIELD.tankLength - 12,
      y: FIELD.ballRadius + 12
    };
    state.tanks[0].angle = 0;
    state.tanks[1].position = {
      x: FIELD.length - FIELD.ballRadius - FIELD.tankLength - 4,
      y: FIELD.ballRadius + FIELD.tankWidth + 14
    };
    state.tanks[1].angle = -Math.PI / 3;

    let worstSeparation = Number.POSITIVE_INFINITY;
    let worstFrame = -1;

    for (let i = 0; i < 90; i += 1) {
      stepGame(state, {
        [state.tanks[0].id]: { leftTrack: 1, rightTrack: 1 },
        [state.tanks[1].id]: { leftTrack: 1, rightTrack: 1 }
      }, 1 / 30);

      for (const tank of state.tanks) {
        const separation = distanceFromTankBody(tank, state.ball.position) - state.ball.radius;
        if (separation < worstSeparation) {
          worstSeparation = separation;
          worstFrame = state.frame;
        }
      }
    }

    for (const tank of state.tanks) {
      const separation = distanceFromTankBody(tank, state.ball.position);
      expect(separation).toBeGreaterThanOrEqual(state.ball.radius - 0.01);
    }
    expect(worstSeparation, `worst frame ${worstFrame}`).toBeGreaterThanOrEqual(-0.01);
  });

  it('counts a goal only through the central goal mouth and resets kickoff', () => {
    const state = createInitialState();
    state.ball.position.x = FIELD.ballRadius + 2;
    state.ball.position.y = FIELD.width / 2;
    state.ball.velocity.x = -600;

    stepGame(state, idleCommands(), 1 / 30);

    expect(state.score.blue).toBe(1);
    expect(state.ball.position.x).toBeCloseTo(FIELD.length / 2, 6);
    expect(state.ball.position.y).toBeCloseTo(FIELD.width / 2, 6);
  });
});

function localPointToWorld(center: { x: number; y: number }, angle: number, point: { x: number; y: number }) {
  return {
    x: center.x + Math.cos(angle) * point.x - Math.sin(angle) * point.y,
    y: center.y + Math.sin(angle) * point.x + Math.cos(angle) * point.y
  };
}

function worldToLocal(tank: { position: { x: number; y: number }; angle: number }, point: { x: number; y: number }) {
  const dx = point.x - tank.position.x;
  const dy = point.y - tank.position.y;
  return {
    x: Math.cos(tank.angle) * dx + Math.sin(tank.angle) * dy,
    y: -Math.sin(tank.angle) * dx + Math.cos(tank.angle) * dy
  };
}

function tankLocalParts() {
  const halfLength = FIELD.tankLength / 2;
  const halfWidth = FIELD.tankWidth / 2;
  const nose = EXPECTED_NOSE_LENGTH;
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

function tankBodyVertices(tank: { position: { x: number; y: number }; angle: number }) {
  const halfLength = FIELD.tankLength / 2;
  const halfWidth = FIELD.tankWidth / 2;
  const nose = EXPECTED_NOSE_LENGTH;
  return [
    localPointToWorld(tank.position, tank.angle, { x: -halfLength, y: -halfWidth }),
    localPointToWorld(tank.position, tank.angle, { x: halfLength, y: -halfWidth }),
    localPointToWorld(tank.position, tank.angle, { x: halfLength + nose, y: -halfWidth }),
    localPointToWorld(tank.position, tank.angle, { x: halfLength, y: 0 }),
    localPointToWorld(tank.position, tank.angle, { x: halfLength + nose, y: halfWidth }),
    localPointToWorld(tank.position, tank.angle, { x: halfLength, y: halfWidth }),
    localPointToWorld(tank.position, tank.angle, { x: -halfLength, y: halfWidth })
  ];
}

function tankWorldParts(tank: { position: { x: number; y: number }; angle: number }) {
  return tankLocalParts().map((part) =>
    part.map((point) => localPointToWorld(tank.position, tank.angle, point))
  );
}

function distanceFromTankBody(
  tank: { position: { x: number; y: number }; angle: number },
  point: { x: number; y: number }
) {
  const local = worldToLocal(tank, point);
  if (tankLocalParts().some((part) => pointInConvexPolygon(local, part))) {
    return 0;
  }

  let best = Number.POSITIVE_INFINITY;
  for (const part of tankLocalParts()) {
    for (let index = 0; index < part.length; index += 1) {
      const a = part[index];
      const b = part[(index + 1) % part.length];
      best = Math.min(best, distanceToSegment(local, a, b));
    }
  }
  return best;
}

function pointInConvexPolygon(point: { x: number; y: number }, polygon: Array<{ x: number; y: number }>) {
  return polygon.every((start, index) => {
    const end = polygon[(index + 1) % polygon.length];
    const edge = { x: end.x - start.x, y: end.y - start.y };
    const outward = { x: edge.y, y: -edge.x };
    return (point.x - start.x) * outward.x + (point.y - start.y) * outward.y <= 0.01;
  });
}

function distanceToSegment(
  point: { x: number; y: number },
  start: { x: number; y: number },
  end: { x: number; y: number }
) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared));
  const closest = { x: start.x + dx * t, y: start.y + dy * t };
  return distance(point, closest);
}

function tankBodiesOverlap(
  a: { position: { x: number; y: number }; angle: number },
  b: { position: { x: number; y: number }; angle: number }
) {
  for (const partA of tankWorldParts(a)) {
    for (const partB of tankWorldParts(b)) {
      if (convexPolygonsOverlap(partA, partB)) {
        return true;
      }
    }
  }
  return false;
}

function convexPolygonsOverlap(
  a: Array<{ x: number; y: number }>,
  b: Array<{ x: number; y: number }>
) {
  const axes = [...polygonAxes(a), ...polygonAxes(b)];
  for (const axis of axes) {
    const projectedA = projectPoints(a, axis);
    const projectedB = projectPoints(b, axis);
    if (projectedA.max <= projectedB.min + 0.01 || projectedB.max <= projectedA.min + 0.01) {
      return false;
    }
  }
  return true;
}

function polygonAxes(points: Array<{ x: number; y: number }>) {
  return points.map((start, index) => {
    const end = points[(index + 1) % points.length];
    const edge = { x: end.x - start.x, y: end.y - start.y };
    const length = Math.hypot(edge.x, edge.y) || 1;
    return { x: edge.y / length, y: -edge.x / length };
  });
}

function projectPoints(points: Array<{ x: number; y: number }>, axis: { x: number; y: number }) {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    const value = point.x * axis.x + point.y * axis.y;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return { min, max };
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
