import { describe, expect, it } from 'vitest';
import type { Strategy } from '../src/game/strategy';
import { generateValueSamples } from '../src/ai/valueTraining';

const idle: Strategy = {
  name: 'idle',
  decide(state, team) {
    const tank = state.tanks.find(
      (candidate) => candidate.team === team && candidate.index === 0
    );
    return tank
      ? { [tank.id]: { leftTrack: 0, rightTrack: 0 } }
      : {};
  }
};

describe('value training data', () => {
  it('adds weighted fork labels without changing the source-match samples', () => {
    const common = {
      candidate: idle,
      opponent: idle,
      seed: 701,
      scenario: 2,
      frames: 12,
      sampleEvery: 6,
      decayFrames: 150
    };
    const sourceOnly = generateValueSamples(common);
    const withFork = generateValueSamples({
      ...common,
      forksPerMatch: 1,
      forkRolloutFrames: 3,
      forkPlayoutFrames: 6,
      forkWeight: 7,
      forkCandidate: idle,
      forkOpponent: idle,
      forkSeed: 99
    });

    expect(sourceOnly).toHaveLength(4);
    expect(withFork.filter((sample) => sample.weight === undefined)).toEqual(sourceOnly);
    expect(withFork.filter((sample) => sample.weight === 7)).toHaveLength(1);
  });

  it('generates deterministic fork states and labels', () => {
    const options = {
      candidate: idle,
      opponent: idle,
      seed: 809,
      scenario: 4,
      frames: 18,
      sampleEvery: 6,
      decayFrames: 150,
      forksPerMatch: 2,
      forkRolloutFrames: 18,
      forkPlayoutFrames: 12,
      forkWeight: 5,
      forkCandidate: idle,
      forkOpponent: idle,
      forkSeed: 1234
    };

    const first = generateValueSamples(options).filter((sample) => sample.weight !== undefined);
    const second = generateValueSamples(options).filter((sample) => sample.weight !== undefined);

    expect(first).toHaveLength(2);
    expect(second).toEqual(first);
    expect(first.every((sample) => sample.inputs.length === 36)).toBe(true);
  });
});
