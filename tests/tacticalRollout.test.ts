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
