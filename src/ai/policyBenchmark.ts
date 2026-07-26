import { simulateMatch } from '../game/match';
import type { Team } from '../game/model';
import type { Strategy } from '../game/strategy';
import { createSeededInitialState } from './policyGate';

/**
 * Large-sample paired evaluation.
 *
 * The legacy standard/holdout gates are 20 matches each, which makes their win
 * proxy move in 0.025 steps and gives no way to tell a real gain from sampling
 * noise. This module measures the same runtime over an arbitrary number of
 * independent start states and reports an explicit confidence interval, so a
 * change has to clear its own error bar before it counts as progress.
 *
 * Every scenario is a single physical start played twice with the sides
 * swapped, which is the pairing the opponent league already established. The
 * scenario mean is the unit of observation, so side bias cancels instead of
 * inflating the sample.
 */

export type BenchmarkMatchOutcome = {
  candidateTeam: Team;
  goalsFor: number;
  goalsAgainst: number;
  winProxy: number;
};

export type BenchmarkScenarioOutcome = {
  seed: number;
  scenario: number;
  winProxy: number;
  goalsFor: number;
  goalsAgainst: number;
  ballProgress: number;
  matches: BenchmarkMatchOutcome[];
};

export type BenchmarkSummary = {
  scenarios: number;
  matches: number;
  winRate: number;
  standardError: number;
  ci95Low: number;
  ci95High: number;
  goalsFor: number;
  goalsAgainst: number;
  ballProgress: number;
};

export type PairedComparison = {
  scenarios: number;
  meanDifference: number;
  standardError: number;
  ci95Low: number;
  ci95High: number;
  better: number;
  worse: number;
  tied: number;
  significant: boolean;
};

const CI95_Z = 1.959964;

/**
 * Deterministic benchmark seeds.
 *
 * These are drawn from a high range so they cannot collide with the legacy
 * gate seeds (19..149) or the league seeds (163..397); the benchmark is meant
 * to measure generalisation, not to re-examine the seeds already tuned against.
 */
export function benchmarkSeeds(count: number, salt = 0): number[] {
  const seeds: number[] = [];
  const used = new Set<number>();
  let state = (0x9e3779b9 ^ Math.imul(salt + 1, 2654435761)) >>> 0;
  while (seeds.length < count) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const seed = 1_000_003 + (state % 8_999_993);
    if (!used.has(seed)) {
      used.add(seed);
      seeds.push(seed);
    }
  }
  return seeds;
}

export function runBenchmarkScenario(
  candidate: Strategy,
  opponent: Strategy,
  seed: number,
  scenario: number,
  frames: number
): BenchmarkScenarioOutcome {
  const initialState = createSeededInitialState(seed, scenario, 'red');
  const matches: BenchmarkMatchOutcome[] = [];

  for (const candidateTeam of ['red', 'blue'] as const) {
    const result = simulateMatch({
      red: candidateTeam === 'red' ? candidate : opponent,
      blue: candidateTeam === 'blue' ? candidate : opponent,
      frames,
      initialState
    }).state;
    const goalsFor = candidateTeam === 'red' ? result.score.red : result.score.blue;
    const goalsAgainst = candidateTeam === 'red' ? result.score.blue : result.score.red;
    matches.push({
      candidateTeam,
      goalsFor,
      goalsAgainst,
      winProxy: goalsFor > goalsAgainst ? 1 : goalsFor === goalsAgainst ? 0.5 : 0
    });
  }

  const goalsFor = matches.reduce((sum, match) => sum + match.goalsFor, 0);
  const goalsAgainst = matches.reduce((sum, match) => sum + match.goalsAgainst, 0);
  return {
    seed,
    scenario,
    winProxy: matches.reduce((sum, match) => sum + match.winProxy, 0) / matches.length,
    goalsFor,
    goalsAgainst,
    ballProgress: 0,
    matches
  };
}

export function summarizeBenchmark(
  outcomes: readonly BenchmarkScenarioOutcome[]
): BenchmarkSummary {
  const scenarios = outcomes.length;
  const values = outcomes.map((outcome) => outcome.winProxy);
  const winRate = mean(values);
  const standardError = standardErrorOfMean(values);

  return {
    scenarios,
    matches: outcomes.reduce((sum, outcome) => sum + outcome.matches.length, 0),
    winRate,
    standardError,
    ci95Low: winRate - CI95_Z * standardError,
    ci95High: winRate + CI95_Z * standardError,
    goalsFor: outcomes.reduce((sum, outcome) => sum + outcome.goalsFor, 0),
    goalsAgainst: outcomes.reduce((sum, outcome) => sum + outcome.goalsAgainst, 0),
    ballProgress: mean(outcomes.map((outcome) => outcome.ballProgress))
  };
}

/**
 * Paired difference between two policies measured on identical start states.
 * Pairing removes start-state variance, which is by far the largest noise term,
 * so this detects far smaller true effects than comparing two independent runs.
 */
export function comparePairedBenchmarks(
  candidate: readonly BenchmarkScenarioOutcome[],
  baseline: readonly BenchmarkScenarioOutcome[]
): PairedComparison {
  if (candidate.length !== baseline.length) {
    throw new Error('Paired benchmark comparison requires matching scenario counts');
  }

  const differences: number[] = [];
  let better = 0;
  let worse = 0;
  let tied = 0;
  for (let index = 0; index < candidate.length; index += 1) {
    const left = candidate[index];
    const right = baseline[index];
    if (left.seed !== right.seed || left.scenario !== right.scenario) {
      throw new Error('Paired benchmark comparison requires identical scenario ordering');
    }
    const difference = left.winProxy - right.winProxy;
    differences.push(difference);
    if (difference > 0) {
      better += 1;
    } else if (difference < 0) {
      worse += 1;
    } else {
      tied += 1;
    }
  }

  const meanDifference = mean(differences);
  const standardError = standardErrorOfMean(differences);
  const ci95Low = meanDifference - CI95_Z * standardError;
  const ci95High = meanDifference + CI95_Z * standardError;

  return {
    scenarios: differences.length,
    meanDifference,
    standardError,
    ci95Low,
    ci95High,
    better,
    worse,
    tied,
    significant: ci95Low > 0 || ci95High < 0
  };
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardErrorOfMean(values: readonly number[]): number {
  if (values.length < 2) {
    return 0;
  }
  const average = mean(values);
  const variance = values.reduce(
    (sum, value) => sum + (value - average) ** 2,
    0
  ) / (values.length - 1);
  return Math.sqrt(variance / values.length);
}
