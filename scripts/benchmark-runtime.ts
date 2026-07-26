import { readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { Worker } from 'node:worker_threads';
import {
  benchmarkSeeds,
  comparePairedBenchmarks,
  runBenchmarkScenario,
  summarizeBenchmark,
  type BenchmarkScenarioOutcome,
  type BenchmarkSummary,
  type PairedComparison
} from '../src/ai/policyBenchmark';
import {
  createRuntimeOpponentStrategy,
  type RuntimeOpponentKind
} from '../src/ai/runtimeOpponentLeague';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import type { TacticalRolloutTuning } from '../src/ai/tacticalRollout';
import type { NeuralWeights } from '../src/ai/neuralWeights';
import type { Strategy } from '../src/game/strategy';
import { loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
  exitCode?: number;
};

/**
 * Parallel large-sample runtime benchmark.
 *
 * Scenarios are independent, so they shard cleanly across worker threads. This
 * is what makes a few hundred start states cheap enough to use as the routine
 * evaluation instead of the 20-match legacy gate.
 */

/**
 * A policy under test: a league opponent kind, optionally with search-shape
 * overrides written as `kind@frames=36,margin=0.05`.
 */
export type BenchmarkPolicySpec = {
  id: string;
  kind: RuntimeOpponentKind;
  tuning?: TacticalRolloutTuning;
};

export type BenchmarkCliOptions = {
  weightsPath: string;
  policies: BenchmarkPolicySpec[];
  opponent: BenchmarkPolicySpec;
  scenarios: number;
  frames: number;
  workers: number;
  salt: number;
  outputPath?: string;
};

type ScenarioSpec = {
  index: number;
  seed: number;
  scenario: number;
};

type WorkerRequest = {
  weights: number[];
  policies: BenchmarkPolicySpec[];
  opponent: BenchmarkPolicySpec;
  frames: number;
  specs: ScenarioSpec[];
};

type WorkerResponse = {
  index: number;
  outcomes: BenchmarkScenarioOutcome[];
};

export function parseBenchmarkArgs(argv: readonly string[]): BenchmarkCliOptions {
  const policies = listArg(argv, '--policies');
  return {
    weightsPath: stringArg(argv, '--weights') ?? 'public/models/neural-best.json',
    policies: (policies.length > 0 ? policies : ['accepted-runtime']).map(parsePolicySpec),
    opponent: parsePolicySpec(stringArg(argv, '--opponent') ?? 'traditional'),
    scenarios: positiveIntegerArg(argv, '--scenarios', 200),
    frames: positiveIntegerArg(argv, '--frames', 600),
    workers: positiveIntegerArg(argv, '--workers', Math.max(1, cpus().length - 2)),
    salt: positiveIntegerArg(argv, '--salt', 1),
    outputPath: stringArg(argv, '--output')
  };
}

export function buildScenarioSpecs(scenarios: number, salt: number): ScenarioSpec[] {
  return benchmarkSeeds(scenarios, salt).map((seed, index) => ({
    index,
    seed,
    scenario: 0
  }));
}

export function createBenchmarkStrategy(
  spec: BenchmarkPolicySpec,
  weights: NeuralWeights,
  name: string
): Strategy {
  if (!spec.tuning || spec.kind === 'traditional') {
    return createRuntimeOpponentStrategy(spec.kind, weights, name);
  }
  return createNeuralStrategy({
    weights,
    name,
    tacticalRollout: spec.kind === 'accepted-runtime',
    tacticalTuning: spec.tuning
  });
}

export function runBenchmarkShard(request: WorkerRequest): WorkerResponse[] {
  const opponent = createBenchmarkStrategy(request.opponent, request.weights, 'benchmark-opponent');
  const policies = request.policies.map((spec, index) =>
    createBenchmarkStrategy(spec, request.weights, `benchmark-policy-${index}`)
  );

  return request.specs.map((spec) => ({
    index: spec.index,
    outcomes: policies.map((policy) =>
      runBenchmarkScenario(policy, opponent, spec.seed, spec.scenario, request.frames)
    )
  }));
}

async function runParallel(
  options: BenchmarkCliOptions,
  weights: NeuralWeights
): Promise<BenchmarkScenarioOutcome[][]> {
  const specs = buildScenarioSpecs(options.scenarios, options.salt);
  const workerCount = Math.max(1, Math.min(options.workers, specs.length));
  const shards: ScenarioSpec[][] = Array.from({ length: workerCount }, () => []);
  specs.forEach((spec, index) => shards[index % workerCount].push(spec));

  const responses = await Promise.all(
    shards.map((shard) => runShardInWorker({
      weights: [...weights],
      policies: options.policies,
      opponent: options.opponent,
      frames: options.frames,
      specs: shard
    }))
  );

  const ordered: BenchmarkScenarioOutcome[][] = options.policies.map(() => []);
  const flat = responses.flat().sort((left, right) => left.index - right.index);
  for (const response of flat) {
    response.outcomes.forEach((outcome, policyIndex) => {
      ordered[policyIndex].push(outcome);
    });
  }
  return ordered;
}

function runShardInWorker(request: WorkerRequest): Promise<WorkerResponse[]> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./benchmark-worker.mjs', import.meta.url), {
      workerData: request
    });
    worker.on('message', (value: never) => {
      resolve(value as WorkerResponse[]);
      void worker.terminate();
    });
    worker.on('error', (error: Error) => reject(error));
  });
}

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<void> {
  const options = parseBenchmarkArgs(argv);
  const weights = loadWeightsPayload(readFileSync(options.weightsPath, 'utf8'));

  const started = Date.now();
  const results = await runParallel(options, weights);
  const elapsed = Date.now() - started;

  const summaries = results.map(summarizeBenchmark);
  console.log(
    `runtime-benchmark opponent=${options.opponent.id} scenarios=${options.scenarios} ` +
    `frames=${options.frames} matches=${summaries[0]?.matches ?? 0} workers=${options.workers} ms=${elapsed}`
  );
  options.policies.forEach((policy, index) => {
    console.log(`  ${policy.id.padEnd(34)} ${formatSummary(summaries[index])}`);
  });

  const comparisons: Record<string, PairedComparison> = {};
  for (let index = 1; index < options.policies.length; index += 1) {
    const comparison = comparePairedBenchmarks(results[index], results[0]);
    comparisons[`${options.policies[index].id}_vs_${options.policies[0].id}`] = comparison;
    console.log(
      `  paired ${options.policies[index].id} - ${options.policies[0].id}: ${formatComparison(comparison)}`
    );
  }

  if (options.outputPath) {
    writeFileSync(
      options.outputPath,
      JSON.stringify({ options, summaries, comparisons, results }, null, 2),
      'utf8'
    );
  }
}

export function formatSummary(summary: BenchmarkSummary): string {
  return [
    `win=${summary.winRate.toFixed(4)}`,
    `+-${(CI_HALF_WIDTH(summary)).toFixed(4)}`,
    `ci=[${summary.ci95Low.toFixed(4)},${summary.ci95High.toFixed(4)}]`,
    `goals=${summary.goalsFor}-${summary.goalsAgainst}`
  ].join(' ');
}

function CI_HALF_WIDTH(summary: BenchmarkSummary): number {
  return (summary.ci95High - summary.ci95Low) / 2;
}

export function formatComparison(comparison: PairedComparison): string {
  return [
    `delta=${comparison.meanDifference >= 0 ? '+' : ''}${comparison.meanDifference.toFixed(4)}`,
    `ci=[${comparison.ci95Low.toFixed(4)},${comparison.ci95High.toFixed(4)}]`,
    `better/worse/tied=${comparison.better}/${comparison.worse}/${comparison.tied}`,
    comparison.significant ? 'SIGNIFICANT' : 'not-significant'
  ].join(' ');
}

export function parsePolicySpec(value: string): BenchmarkPolicySpec {
  const [kindPart, tuningPart] = value.split('@');
  const kind = parseKind(kindPart);
  if (!tuningPart) {
    return { id: value, kind };
  }

  // `+` separates tuning pairs because `,` already separates policies.
  const tuning: TacticalRolloutTuning = {};
  for (const entry of tuningPart.split('+')) {
    const [key, raw] = entry.split('=');
    const parsed = Number(raw);
    if (!Number.isFinite(parsed)) {
      throw new Error(`Invalid tuning value in policy spec: ${value}`);
    }
    if (key === 'frames') {
      tuning.defaultFrames = parsed;
    } else if (key === 'margin') {
      tuning.improvementMargin = parsed;
    } else if (key === 'force') {
      tuning.forceTrigger = parsed !== 0;
    } else if (key === 'opp') {
      tuning.opponentModel = parsed !== 0 ? 'policy' : 'stop';
    } else {
      throw new Error(`Unknown tuning key "${key}" in policy spec: ${value}`);
    }
  }
  return { id: value, kind, tuning };
}

function parseKind(value: string): RuntimeOpponentKind {
  if (value === 'traditional' || value === 'accepted-no-rollout' || value === 'accepted-runtime') {
    return value;
  }
  throw new Error(`Unknown policy kind: ${value}`);
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value && !value.startsWith('--') ? value : undefined;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  const value = Number(stringArg(argv, name));
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
}

function listArg(argv: readonly string[], name: string): string[] {
  const index = argv.indexOf(name);
  if (index < 0) {
    return [];
  }
  const values: string[] = [];
  for (let cursor = index + 1; cursor < argv.length; cursor += 1) {
    const value = argv[cursor];
    if (value.startsWith('--')) {
      break;
    }
    values.push(...value.split(',').map((part) => part.trim()).filter(Boolean));
  }
  return values;
}

if (
  process.argv[1]?.replace(/\\/g, '/').endsWith('/benchmark-runtime.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/benchmark-runtime.js')
) {
  void main();
}
