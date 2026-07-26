import { describe, expect, it } from 'vitest';
import {
  benchmarkSeeds,
  comparePairedBenchmarks,
  runBenchmarkScenario,
  summarizeBenchmark,
  type BenchmarkScenarioOutcome
} from '../src/ai/policyBenchmark';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { parsePolicySpec } from '../scripts/benchmark-runtime';

function scenarioOutcome(seed: number, winProxy: number): BenchmarkScenarioOutcome {
  return {
    seed,
    scenario: 0,
    winProxy,
    goalsFor: 0,
    goalsAgainst: 0,
    ballProgress: 0,
    matches: []
  };
}

describe('policy benchmark', () => {
  it('generates distinct seeds outside the legacy gate and league ranges', () => {
    const seeds = benchmarkSeeds(500, 1);
    expect(seeds).toHaveLength(500);
    expect(new Set(seeds).size).toBe(500);
    for (const seed of seeds) {
      expect(seed).toBeGreaterThan(1_000_000);
    }
    // Deterministic for a given salt, and different across salts.
    expect(benchmarkSeeds(8, 1)).toEqual(seeds.slice(0, 8));
    expect(benchmarkSeeds(8, 2)).not.toEqual(benchmarkSeeds(8, 1));
  });

  it('scores identical strategies as an exact draw on paired starts', () => {
    // Self-play on a swapped-side pair must cancel exactly; anything else means
    // the harness itself favours one colour and would fake improvements.
    const weights = defaultNeuralWeights();
    const left = createNeuralStrategy({ weights, name: 'mirror-a', tacticalRollout: false });
    const right = createNeuralStrategy({ weights, name: 'mirror-b', tacticalRollout: false });

    for (const seed of benchmarkSeeds(4, 7)) {
      const outcome = runBenchmarkScenario(left, right, seed, 0, 240);
      expect(outcome.winProxy).toBe(0.5);
      expect(outcome.goalsFor).toBe(outcome.goalsAgainst);
    }
  });

  it('reports a symmetric result regardless of which side the candidate takes', () => {
    const weights = defaultNeuralWeights();
    const neural = createNeuralStrategy({ weights, name: 'neural', tacticalRollout: false });
    const outcome = runBenchmarkScenario(neural, traditionalStrategy, benchmarkSeeds(1, 3)[0], 0, 240);
    expect(outcome.matches).toHaveLength(2);
    expect(outcome.matches[0].candidateTeam).toBe('red');
    expect(outcome.matches[1].candidateTeam).toBe('blue');
    expect(outcome.winProxy).toBeGreaterThanOrEqual(0);
    expect(outcome.winProxy).toBeLessThanOrEqual(1);
  });

  it('summarizes a win rate with a confidence interval that shrinks as samples grow', () => {
    const small = summarizeBenchmark([
      scenarioOutcome(1, 1),
      scenarioOutcome(2, 0),
      scenarioOutcome(3, 1),
      scenarioOutcome(4, 0)
    ]);
    const large = summarizeBenchmark(
      Array.from({ length: 400 }, (_unused, index) => scenarioOutcome(index, index % 2))
    );

    expect(small.winRate).toBeCloseTo(0.5, 10);
    expect(large.winRate).toBeCloseTo(0.5, 10);
    expect(large.standardError).toBeLessThan(small.standardError);
    expect(large.ci95High - large.ci95Low).toBeLessThan(small.ci95High - small.ci95Low);
  });

  it('reports zero error for a constant result', () => {
    const summary = summarizeBenchmark(
      Array.from({ length: 32 }, (_unused, index) => scenarioOutcome(index, 0.5))
    );
    expect(summary.winRate).toBe(0.5);
    expect(summary.standardError).toBe(0);
    expect(summary.ci95Low).toBe(0.5);
    expect(summary.ci95High).toBe(0.5);
  });

  it('detects a consistent paired improvement but not a noisy one', () => {
    const baseline = Array.from({ length: 60 }, (_unused, index) => scenarioOutcome(index, 0.5));
    const consistent = Array.from({ length: 60 }, (_unused, index) => scenarioOutcome(index, 1));
    const noisy = Array.from({ length: 60 }, (_unused, index) =>
      scenarioOutcome(index, index % 2 === 0 ? 1 : 0)
    );

    const real = comparePairedBenchmarks(consistent, baseline);
    expect(real.meanDifference).toBeCloseTo(0.5, 10);
    expect(real.significant).toBe(true);
    expect(real.better).toBe(60);

    const noise = comparePairedBenchmarks(noisy, baseline);
    expect(noise.meanDifference).toBeCloseTo(0, 10);
    expect(noise.significant).toBe(false);
  });

  it('rejects mismatched scenario sets so comparisons stay paired', () => {
    expect(() => comparePairedBenchmarks(
      [scenarioOutcome(1, 1)],
      [scenarioOutcome(1, 1), scenarioOutcome(2, 1)]
    )).toThrow(/matching scenario counts/);

    expect(() => comparePairedBenchmarks(
      [scenarioOutcome(1, 1)],
      [scenarioOutcome(2, 1)]
    )).toThrow(/identical scenario ordering/);
  });
});

describe('benchmark policy specs', () => {
  it('parses a bare opponent kind', () => {
    expect(parsePolicySpec('accepted-runtime')).toEqual({
      id: 'accepted-runtime',
      kind: 'accepted-runtime'
    });
  });

  it('parses search-shape overrides', () => {
    expect(parsePolicySpec('accepted-runtime@frames=36+margin=0.05')).toEqual({
      id: 'accepted-runtime@frames=36+margin=0.05',
      kind: 'accepted-runtime',
      tuning: { defaultFrames: 36, improvementMargin: 0.05 }
    });
  });

  it('parses the trigger override used to test where search is allowed to run', () => {
    expect(parsePolicySpec('accepted-runtime@force=1')).toEqual({
      id: 'accepted-runtime@force=1',
      kind: 'accepted-runtime',
      tuning: { forceTrigger: true }
    });
  });

  it('rejects unknown kinds and tuning keys', () => {
    expect(() => parsePolicySpec('nonsense')).toThrow(/Unknown policy kind/);
    expect(() => parsePolicySpec('accepted-runtime@depth=4')).toThrow(/Unknown tuning key/);
    expect(() => parsePolicySpec('accepted-runtime@frames=abc')).toThrow(/Invalid tuning value/);
  });
});
