import { readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { Worker } from 'node:worker_threads';
import { benchmarkSeeds } from '../src/ai/policyBenchmark';
import {
  createRuntimeOpponentStrategy,
  type RuntimeOpponentKind
} from '../src/ai/runtimeOpponentLeague';
import { createExploringStrategy, generateValueSamples } from '../src/ai/valueTraining';
import {
  VALUE_WEIGHT_COUNT,
  createValueAdamMoments,
  createValueWeights,
  trainValueBatch,
  valueLossGradients,
  type ValueSample
} from '../src/ai/valueNetwork';
import { loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
  exitCode?: number;
};

/**
 * Trains the state value network on Monte-Carlo next-goal labels.
 *
 * Data generation dominates the cost and shards cleanly across worker threads.
 * Matches are drawn against a mix of opponents with some action noise, so the
 * network sees more than the narrow on-policy ridge the accepted runtime walks.
 */

export type ValueTrainingCliOptions = {
  weightsPath: string;
  outputPath: string;
  matches: number;
  frames: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  l2: number;
  explorationRate: number;
  decayFrames: number;
  holdoutFraction: number;
  workers: number;
  seed: number;
};

type GenerationRequest = {
  weights: number[];
  seeds: number[];
  frames: number;
  explorationRate: number;
  decayFrames: number;
  seed: number;
};

const OPPONENT_MIX: RuntimeOpponentKind[] = [
  'traditional',
  'accepted-runtime',
  'accepted-no-rollout'
];

export function parseValueTrainingArgs(argv: readonly string[]): ValueTrainingCliOptions {
  return {
    weightsPath: stringArg(argv, '--weights') ?? 'public/models/neural-best.json',
    outputPath: stringArg(argv, '--output') ?? 'training-runs/value-network.json',
    matches: positiveIntegerArg(argv, '--matches', 600),
    frames: positiveIntegerArg(argv, '--frames', 600),
    epochs: positiveIntegerArg(argv, '--epochs', 12),
    batchSize: positiveIntegerArg(argv, '--batch-size', 256),
    learningRate: numberArg(argv, '--learning-rate', 0.003),
    l2: numberArg(argv, '--l2', 1e-6),
    explorationRate: numberArg(argv, '--exploration', 0.15),
    decayFrames: numberArg(argv, '--decay-frames', 150),
    holdoutFraction: numberArg(argv, '--holdout', 0.15),
    workers: positiveIntegerArg(argv, '--workers', Math.max(1, cpus().length - 2)),
    seed: positiveIntegerArg(argv, '--seed', 20260726)
  };
}

export function generateSampleShard(request: GenerationRequest): ValueSample[] {
  const samples: ValueSample[] = [];

  request.seeds.forEach((seed, index) => {
    const opponentKind = OPPONENT_MIX[index % OPPONENT_MIX.length];
    const candidate = createRuntimeOpponentStrategy('accepted-runtime', request.weights, 'value-candidate');
    const opponent = createRuntimeOpponentStrategy(opponentKind, request.weights, 'value-opponent');
    // Explore on both sides so the noise does not bias which colour looks good.
    const exploringCandidate = createExploringStrategy(
      candidate,
      request.explorationRate,
      seed ^ request.seed
    );
    const exploringOpponent = createExploringStrategy(
      opponent,
      request.explorationRate,
      (seed ^ request.seed) + 7919
    );

    samples.push(...generateValueSamples({
      candidate: exploringCandidate,
      opponent: exploringOpponent,
      seed,
      scenario: 0,
      frames: request.frames,
      decayFrames: request.decayFrames
    }));
  });

  return samples;
}

async function generateSamples(
  options: ValueTrainingCliOptions,
  weights: readonly number[]
): Promise<ValueSample[]> {
  const seeds = benchmarkSeeds(options.matches, options.seed % 1000);
  const workerCount = Math.max(1, Math.min(options.workers, seeds.length));
  const shards: number[][] = Array.from({ length: workerCount }, () => []);
  seeds.forEach((seed, index) => shards[index % workerCount].push(seed));

  const results = await Promise.all(shards.map((shard) => runGenerationWorker({
    weights: [...weights],
    seeds: shard,
    frames: options.frames,
    explorationRate: options.explorationRate,
    decayFrames: options.decayFrames,
    seed: options.seed
  })));

  return results.flat();
}

function runGenerationWorker(request: GenerationRequest): Promise<ValueSample[]> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./value-training-worker.mjs', import.meta.url), {
      workerData: request
    });
    worker.on('message', (value: never) => {
      resolve(value as ValueSample[]);
      void worker.terminate();
    });
    worker.on('error', (error: Error) => reject(error));
  });
}

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<void> {
  const options = parseValueTrainingArgs(argv);
  const policyWeights = loadWeightsPayload(readFileSync(options.weightsPath, 'utf8'));

  const generationStarted = Date.now();
  const samples = await generateSamples(options, policyWeights);
  const generationMs = Date.now() - generationStarted;

  const shuffled = shuffle(samples, options.seed);
  const holdoutSize = Math.floor(shuffled.length * options.holdoutFraction);
  const holdout = shuffled.slice(0, holdoutSize);
  const training = shuffled.slice(holdoutSize);

  console.log(
    `value-training samples=${samples.length} training=${training.length} holdout=${holdout.length} ` +
    `nonzero=${(samples.filter((sample) => sample.target !== 0).length / Math.max(1, samples.length)).toFixed(3)} ` +
    `generationMs=${generationMs}`
  );

  let weights = createValueWeights(options.seed);
  const moments = createValueAdamMoments();
  let bestWeights = weights;
  let bestHoldoutLoss = Number.POSITIVE_INFINITY;

  for (let epoch = 0; epoch < options.epochs; epoch += 1) {
    const epochSamples = shuffle(training, options.seed + epoch + 1);
    let trainingLoss = 0;
    let batches = 0;

    for (let start = 0; start < epochSamples.length; start += options.batchSize) {
      const batch = epochSamples.slice(start, start + options.batchSize);
      const result = trainValueBatch(batch, weights, {
        learningRate: options.learningRate,
        l2: options.l2,
        moments
      });
      weights = result.weights;
      trainingLoss += result.loss;
      batches += 1;
    }

    const holdoutLoss = holdout.length > 0
      ? valueLossGradients(holdout, weights).loss
      : trainingLoss / Math.max(1, batches);
    // Keep the best holdout epoch rather than the last, so a late overfitting
    // epoch cannot silently become the shipped model.
    if (holdoutLoss < bestHoldoutLoss) {
      bestHoldoutLoss = holdoutLoss;
      bestWeights = weights;
    }

    console.log(
      `  epoch=${epoch + 1} trainingLoss=${(trainingLoss / Math.max(1, batches)).toFixed(6)} ` +
      `holdoutLoss=${holdoutLoss.toFixed(6)}${holdoutLoss === bestHoldoutLoss ? ' *' : ''}`
    );
  }

  const baselineLoss = holdout.length > 0 ? meanSquare(holdout) : 0;
  console.log(
    `value-training bestHoldoutLoss=${bestHoldoutLoss.toFixed(6)} ` +
    `predictAlwaysZeroLoss=${baselineLoss.toFixed(6)} ` +
    `explainedVariance=${(1 - bestHoldoutLoss / Math.max(1e-9, baselineLoss)).toFixed(4)}`
  );

  writeFileSync(options.outputPath, JSON.stringify({
    schemaVersion: 1,
    kind: 'value-network',
    inputCount: 36,
    hiddenLayerSizes: [32, 32],
    weightCount: VALUE_WEIGHT_COUNT,
    holdoutLoss: bestHoldoutLoss,
    baselineLoss,
    options,
    weights: bestWeights
  }, null, 2), 'utf8');
  console.log(`value-training wrote ${options.outputPath}`);
}

function meanSquare(samples: readonly ValueSample[]): number {
  if (samples.length === 0) {
    return 0;
  }
  return samples.reduce((sum, sample) => sum + sample.target * sample.target, 0) / samples.length;
}

function shuffle<T>(values: readonly T[], seed: number): T[] {
  const result = [...values];
  let state = seed >>> 0;
  const random = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };

  for (let index = result.length - 1; index > 0; index -= 1) {
    const swap = Math.floor(random() * (index + 1));
    const temp = result[index];
    result[index] = result[swap];
    result[swap] = temp;
  }
  return result;
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value && !value.startsWith('--') ? value : undefined;
}

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = Number(stringArg(argv, name));
  return Number.isFinite(value) ? value : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  const value = Number(stringArg(argv, name));
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
}

if (
  process.argv[1]?.replace(/\\/g, '/').endsWith('/train-value-network.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/train-value-network.js')
) {
  void main();
}
