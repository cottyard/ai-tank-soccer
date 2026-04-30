import { describe, expect, it } from 'vitest';
import { FIELD, createInitialState } from '../src/game/model';
import { simulateMatch } from '../src/game/match';
import type { Strategy } from '../src/game/strategy';

const idle: Strategy = {
  name: 'idle',
  decide() {
    return {};
  }
};

describe('match runner', () => {
  it('runs deterministic fixed-step matches and keeps entities inside the walled field', () => {
    const result = simulateMatch({
      red: idle,
      blue: idle,
      frames: 90,
      initialState: createInitialState()
    });

    expect(result.state.frame).toBe(90);
    expect(result.state.ball.position.x).toBeGreaterThanOrEqual(FIELD.ballRadius);
    expect(result.state.ball.position.x).toBeLessThanOrEqual(FIELD.length - FIELD.ballRadius);
    expect(result.state.ball.position.y).toBeGreaterThanOrEqual(FIELD.ballRadius);
    expect(result.state.ball.position.y).toBeLessThanOrEqual(FIELD.width - FIELD.ballRadius);
    expect(result.redDecisions).toBe(15);
    expect(result.blueDecisions).toBe(15);
  });
});
