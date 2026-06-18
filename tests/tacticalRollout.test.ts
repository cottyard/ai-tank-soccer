import { describe, expect, it } from 'vitest';
import { FIELD, createInitialState, type GameState } from '../src/game/model';
import { chooseTacticalAction } from '../src/ai/tacticalRollout';

describe('tactical rollout action selector', () => {
  it('overrides a bad policy action when a direct finish is available', () => {
    const state = createDirectFinishState();

    const choice = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      rolloutFrames: 18
    });

    expect(choice.actionIndex).toBe(8);
    expect(choice.score).toBeGreaterThan(choice.policyScore);
    expect(choice.actionScores).toHaveLength(9);
    expect(choice.actionScores[2]).toBeCloseTo(choice.policyScore, 9);
    expect(Math.max(...choice.actionScores)).toBeCloseTo(choice.score, 9);
  });

  it('keeps the policy action when every rollout has the same tactical value', () => {
    const state = createInitialState();
    const red = state.tanks.find((tank) => tank.team === 'red');
    if (!red) {
      throw new Error('missing red tank');
    }
    red.stamina = 0;

    const choice = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 4,
      rolloutFrames: 1
    });

    expect(choice.actionIndex).toBe(4);
  });

  it('uses a longer horizon to release a slow pinned attacking-corner ball', () => {
    const state = createPinnedAttackCornerState();

    const shortChoice = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 5,
      rolloutFrames: 18
    });
    const defaultChoice = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 5
    });

    expect(defaultChoice.actionIndex).not.toBe(4);
    expect(defaultChoice.score).toBeGreaterThan(shortChoice.score + 0.5);
  });

  it('uses a two-step sequence to recover a slow attacking stall', () => {
    const state = createSlowAttackStallState();

    const shortChoice = chooseTacticalAction({
      state,
      team: 'blue',
      policyActionIndex: 8,
      rolloutFrames: 18
    });
    const defaultChoice = chooseTacticalAction({
      state,
      team: 'blue',
      policyActionIndex: 8
    });

    expect(shortChoice.actionIndex).toBe(4);
    expect(defaultChoice.actionIndex).not.toBe(4);
    expect(defaultChoice.score).toBeGreaterThan(shortChoice.score + 0.15);
  });

  it('does not use the two-step attacking stall sequence at low stamina', () => {
    const state = createSlowAttackStallState();
    const blue = state.tanks.find((tank) => tank.team === 'blue');
    if (!blue) {
      throw new Error('missing blue tank');
    }
    blue.stamina = blue.maxStamina * 0.2;

    const shortChoice = chooseTacticalAction({
      state,
      team: 'blue',
      policyActionIndex: 8,
      rolloutFrames: 18
    });
    const defaultChoice = chooseTacticalAction({
      state,
      team: 'blue',
      policyActionIndex: 8
    });

    expect(defaultChoice.actionIndex).toBe(shortChoice.actionIndex);
    expect(defaultChoice.score).toBeCloseTo(shortChoice.score, 9);
  });

  it('uses a short two-step sequence for a central finish stall', () => {
    const state = createCentralFinishStallState();

    const shortChoice = chooseTacticalAction({
      state,
      team: 'blue',
      policyActionIndex: 8,
      rolloutFrames: 18
    });
    const defaultChoice = chooseTacticalAction({
      state,
      team: 'blue',
      policyActionIndex: 8
    });

    expect(shortChoice.actionIndex).toBe(8);
    expect(defaultChoice.actionIndex).not.toBe(4);
    expect(defaultChoice.score).toBeGreaterThan(shortChoice.score + 9);
  });

  it('uses a longer defensive horizon for a fast own-goal threat', () => {
    const state = createFastOwnGoalThreatState();

    const shortChoice = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 7,
      rolloutFrames: 18
    });
    const defaultChoice = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 7
    });

    expect(shortChoice.actionIndex).toBe(0);
    expect(defaultChoice.actionIndex).not.toBe(0);
    expect(defaultChoice.score).toBeGreaterThan(defaultChoice.policyScore + 0.15);
  });
});

function createDirectFinishState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.length - 185, y: FIELD.width / 2 };
  state.ball.velocity = { x: 0, y: 0 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  red.position = {
    x: state.ball.position.x - FIELD.ballRadius - FIELD.tankLength - 2,
    y: state.ball.position.y
  };
  red.velocity = { x: 0, y: 0 };
  red.angle = 0;
  red.angularVelocity = 0;
  red.stamina = red.maxStamina;

  blue.position = { x: FIELD.length - 130, y: FIELD.width / 2 + 210 };
  blue.velocity = { x: 0, y: 0 };
  blue.angle = Math.PI;
  blue.angularVelocity = 0;

  return state;
}

function createPinnedAttackCornerState(): GameState {
  const state = createInitialState();
  state.ball.position = {
    x: 940,
    y: 560
  };
  state.ball.velocity = { x: 0, y: 0 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  red.position = { x: 820, y: 540 };
  red.velocity = { x: 0, y: 0 };
  red.angle = -1;
  red.angularVelocity = 0;
  red.stamina = red.maxStamina;

  blue.position = { x: 970, y: 470 };
  blue.velocity = { x: 0, y: 0 };
  blue.angle = Math.PI / 2;
  blue.angularVelocity = 0;

  return state;
}

function createSlowAttackStallState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: 108, y: 322 };
  state.ball.velocity = { x: -0.2, y: -0.1 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  blue.position = { x: 215, y: 288 };
  blue.velocity = { x: 0, y: 0 };
  blue.angle = 2.72;
  blue.angularVelocity = 0;
  blue.stamina = blue.maxStamina;

  red.position = { x: 68, y: 412 };
  red.velocity = { x: 0, y: 0 };
  red.angle = -1.15;
  red.angularVelocity = 0;

  return state;
}

function createCentralFinishStallState(): GameState {
  const state = createInitialState();
  state.frame = 420;
  state.time = 13.999999999999961;
  state.ball.position = { x: 114.15339122281685, y: 332.02971868102304 };
  state.ball.velocity = { x: -0.9804244313712193, y: -2.5387959414967485 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  blue.position = { x: 258.1043086835261, y: 371.09308225509864 };
  blue.velocity = { x: 0, y: 0 };
  blue.angle = -2.8245060058127316;
  blue.angularVelocity = 0;
  blue.stamina = blue.maxStamina;

  red.position = { x: 47.992568089235404, y: 428.320749707148 };
  red.velocity = { x: 0, y: 0 };
  red.angle = -1.8044663074148523;
  red.angularVelocity = 0;

  return state;
}

function createFastOwnGoalThreatState(): GameState {
  const state = createInitialState();
  state.frame = 78;
  state.time = 2.6;
  state.ball.position = { x: 227.122, y: 369.038 };
  state.ball.velocity = { x: -243.512, y: -0.436 };

  const red = state.tanks.find((tank) => tank.team === 'red');
  const blue = state.tanks.find((tank) => tank.team === 'blue');
  if (!red || !blue) {
    throw new Error('missing tanks');
  }

  red.position = { x: 201.689, y: 284.038 };
  red.velocity = { x: -245, y: 0 };
  red.angle = 0;
  red.angularVelocity = 0;
  red.stamina = red.maxStamina * 0.52;

  blue.position = { x: 330, y: 339 };
  blue.velocity = { x: 0, y: 0 };
  blue.angle = Math.PI;
  blue.angularVelocity = 0;

  return state;
}
