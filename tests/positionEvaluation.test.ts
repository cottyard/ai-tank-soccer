import { describe, expect, it } from 'vitest';
import { evaluatePosition, evaluatePositionDelta } from '../src/ai/positionEvaluation';
import { FIELD, createInitialState, type GameState } from '../src/game/model';

describe('position evaluation', () => {
  it('values a controlled direct finish above neutral kickoff play', () => {
    const neutral = createInitialState();
    const finish = createDirectFinishState();

    expect(evaluatePosition(finish, 'red').total).toBeGreaterThan(
      evaluatePosition(neutral, 'red').total + 1.1
    );
  });

  it('rewards being behind the ball with a clean shot line', () => {
    const good = createDirectFinishState();
    const bad = createDirectFinishState();
    const badRed = bad.tanks.find((tank) => tank.team === 'red');
    if (!badRed) {
      throw new Error('missing red tank');
    }
    badRed.position.x = bad.ball.position.x + FIELD.ballRadius + FIELD.tankLength;
    badRed.angle = Math.PI;

    expect(evaluatePosition(good, 'red').breakdown.possession).toBeGreaterThan(
      evaluatePosition(bad, 'red').breakdown.possession + 0.35
    );
  });

  it('values an on-target shot near the goal above a stalled ball in the same lane', () => {
    const stalled = createNearGoalLaneState();
    const shot = createNearGoalLaneState();
    shot.ball.velocity = { x: 285, y: 0 };

    const stalledScore = evaluatePosition(stalled, 'red');
    const shotScore = evaluatePosition(shot, 'red');

    expect(shotScore.breakdown.finishThreat).toBeGreaterThan(0.75);
    expect(shotScore.total).toBeGreaterThan(stalledScore.total + 0.75);
  });

  it('can ablate one term without changing the reported breakdown', () => {
    const shot = createNearGoalLaneState();
    shot.ball.velocity = { x: 285, y: 0 };

    const baseline = evaluatePosition(shot, 'red');
    const ablated = evaluatePosition(shot, 'red', { finishThreat: 0 });

    expect(ablated.breakdown).toEqual(baseline.breakdown);
    expect(baseline.total - ablated.total).toBeCloseTo(
      baseline.breakdown.finishThreat * 1.05,
      10
    );
  });

  it('applies the same term ablation to delta weights', () => {
    const stalled = createNearGoalLaneState();
    const shot = createNearGoalLaneState();
    shot.ball.velocity = { x: 285, y: 0 };

    const baseline = evaluatePositionDelta(shot, stalled, 'red');
    const ablated = evaluatePositionDelta(shot, stalled, 'red', { finishThreat: 0 });

    expect(ablated.breakdown).toEqual(baseline.breakdown);
    expect(baseline.total - ablated.total).toBeCloseTo(
      baseline.breakdown.finishThreat * 1.25,
      10
    );
  });

  it('penalizes a dangerous ball in the own goal lane', () => {
    const danger = createInitialState();
    danger.ball.position = { x: 150, y: FIELD.width / 2 };
    danger.ball.velocity = { x: -260, y: 0 };

    const cleared = createInitialState();
    cleared.ball.position = { x: FIELD.length / 2 + 120, y: FIELD.width / 2 + 90 };
    cleared.ball.velocity = { x: 120, y: 20 };

    expect(evaluatePosition(danger, 'red').breakdown.ownDanger).toBeGreaterThan(0.55);
    expect(evaluatePosition(cleared, 'red').total).toBeGreaterThan(
      evaluatePosition(danger, 'red').total + 1.0
    );
  });

  it('rewards moving an attacking corner ball away from the wall and toward the center', () => {
    const initial = createAttackCornerState();
    const escaped = createAttackCornerState();
    escaped.ball.position.y = FIELD.width / 2 + 70;
    escaped.ball.position.x -= 30;
    escaped.ball.velocity = { x: -20, y: 120 };
    const red = escaped.tanks.find((tank) => tank.team === 'red');
    if (red) {
      red.position.x -= 30;
      red.position.y += (FIELD.width / 2 + 70) - initial.ball.position.y;
    }

    expect(evaluatePositionDelta(escaped, initial, 'red').total).toBeGreaterThan(0.45);
    expect(evaluatePositionDelta(escaped, initial, 'red').breakdown.cornerEscape).toBeGreaterThan(0.2);
  });

  it('rewards contesting a defended own-corner ball with the nose toward the ball', () => {
    const passive = createOwnCornerContestState();
    const active = createOwnCornerContestState();
    const activeRed = active.tanks.find((tank) => tank.team === 'red');
    if (!activeRed) {
      throw new Error('missing red tank');
    }
    activeRed.position = { x: 122, y: FIELD.ballRadius + 92 };
    activeRed.angle = -Math.PI / 2;

    expect(evaluatePosition(active, 'red').breakdown.contest).toBeGreaterThan(
      evaluatePosition(passive, 'red').breakdown.contest + 0.25
    );
    expect(evaluatePosition(active, 'red').total).toBeGreaterThan(evaluatePosition(passive, 'red').total);
  });
});

function createDirectFinishState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.length - 185, y: FIELD.width / 2 };
  state.ball.velocity = { x: 80, y: 0 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  red.position = {
    x: state.ball.position.x - FIELD.ballRadius - FIELD.tankLength - 2,
    y: state.ball.position.y
  };
  red.angle = 0;
  red.velocity = { x: 0, y: 0 };
  red.angularVelocity = 0;

  blue.position = { x: FIELD.length - 120, y: FIELD.width / 2 + 230 };
  blue.angle = Math.PI;

  return state;
}

function createNearGoalLaneState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.length - 108, y: FIELD.width / 2 };
  state.ball.velocity = { x: 0, y: 0 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  red.position = {
    x: state.ball.position.x - FIELD.ballRadius - FIELD.tankLength - 6,
    y: state.ball.position.y
  };
  red.angle = 0;
  blue.position = { x: FIELD.length - 130, y: FIELD.width / 2 + 210 };

  return state;
}

function createAttackCornerState(): GameState {
  const state = createInitialState();
  state.ball.position = {
    x: FIELD.length - FIELD.ballRadius - 24,
    y: FIELD.ballRadius + 18
  };
  state.ball.velocity = { x: 0, y: 0 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  if (!red) {
    throw new Error('missing red tank');
  }
  red.position = { x: state.ball.position.x - 150, y: state.ball.position.y + 88 };
  red.angle = 0;

  return state;
}

function createOwnCornerContestState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.ballRadius + 20, y: FIELD.ballRadius + 20 };
  state.ball.velocity = { x: 0, y: 0 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  red.position = { x: 160, y: FIELD.ballRadius + 160 };
  red.angle = Math.PI / 2;
  blue.position = { x: 120, y: FIELD.ballRadius + 75 };
  blue.angle = Math.PI;

  return state;
}
