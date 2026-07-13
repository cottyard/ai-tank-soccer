import { describe, expect, it } from 'vitest';
import { FIELD, createInitialState, type GameState } from '../src/game/model';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import {
  inspectFixedActionOutcome,
  inspectPolicyContinuation,
  parseInspectRuntimeMatchArgs
} from '../scripts/inspect-runtime-match';

describe('runtime inspect probe', () => {
  it('parses rollout breakdown diagnostics', () => {
    expect(parseInspectRuntimeMatchArgs([
      '--seed',
      '71',
      '--match',
      '3',
      '--from',
      '528',
      '--rollout-frames',
      '18',
      '36',
      '--continuation-frames',
      '36',
      '72',
      '--rollout-breakdown'
    ])).toMatchObject({
      seed: 71,
      match: 3,
      from: 528,
      rolloutFrames: [18, 36],
      continuationFrames: [36, 72],
      rolloutBreakdown: true
    });
  });

  it('reports position breakdowns for a fixed rollout action', () => {
    const state = createDirectFinishState();

    const result = inspectFixedActionOutcome(state, 'red', 8, 18);
    const delta = result.delta as Record<string, number>;
    const after = result.after as Record<string, number>;

    expect(result).toMatchObject({
      actionIndex: 8,
      frames: 18
    });
    expect(Number.isFinite(result.score as number)).toBe(true);
    expect(Number.isFinite(delta.finishThreat)).toBe(true);
    expect(Number.isFinite(after.shotLane)).toBe(true);
  });

  it('reports runtime policy continuation outcomes from an inspected state', () => {
    const state = createDirectFinishState();

    const result = inspectPolicyContinuation(state, 'red', defaultNeuralWeights(), 12);

    expect(result).toMatchObject({
      frames: 12
    });
    expect(Number.isFinite(result.goalsForDelta as number)).toBe(true);
    expect(Number.isFinite(result.finalAttackX as number)).toBe(true);
    expect(Number.isFinite(result.finalLane as number)).toBe(true);
  });
});

function createDirectFinishState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.length - 185, y: FIELD.width / 2 + 12 };
  state.ball.velocity = { x: 40, y: 0 };

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
  red.stamina = red.maxStamina;

  blue.position = { x: FIELD.length - 120, y: FIELD.width / 2 + 210 };
  blue.angle = Math.PI;

  return state;
}
