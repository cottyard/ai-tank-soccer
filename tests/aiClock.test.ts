import { describe, expect, it } from 'vitest';
import { createInitialState } from '../src/game/model';
import { AiClock, type Strategy } from '../src/game/strategy';

describe('ai decision cadence', () => {
  it('runs strategy decisions at 5Hz while physics frames run at 30Hz', () => {
    const calls: number[] = [];
    const strategy: Strategy = {
      name: 'counter',
      decide(state) {
        calls.push(state.frame);
        return {};
      }
    };
    const clock = new AiClock(strategy, strategy);
    const state = createInitialState();

    for (let frame = 0; frame < 30; frame += 1) {
      state.frame = frame;
      clock.update(state);
    }

    expect(calls).toEqual([0, 0, 6, 6, 12, 12, 18, 18, 24, 24]);
  });
});
