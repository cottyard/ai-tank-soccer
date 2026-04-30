import { readFileSync, writeFileSync } from 'node:fs';
import { loadWeightsPayload, serializeWeightsPayload } from './coach-neural';
import { trainPolicyGradientSelfPlay } from '../src/ai/policyGradientTraining';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type PolicyGradientCliOptions = {
  input?: string;
  output?: string;
  metricsOutput?: string;
  seed: number;
  matches: number;
  frames: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  ppoClip: number;
  temperature: number;
  discount: number;
  startStateMode: 'open' | 'outcome-curriculum';
};

const DEFAULT_OPTIONS: PolicyGradientCliOptions = {
  seed: 1,
  matches: 32,
  frames: 30 * 30,
  epochs: 4,
  batchSize: 128,
  learningRate: 0.006,
  ppoClip: 0.2,
  temperature: 1.08,
  discount: 0.992,
  startStateMode: 'outcome-curriculum'
};

export function parsePolicyGradientArgs(argv: readonly string[]): PolicyGradientCliOptions {
  return {
    input: stringArg(argv, '--input'),
    output: stringArg(argv, '--output'),
    metricsOutput: stringArg(argv, '--metrics-output'),
    seed: numberArg(argv, '--seed', DEFAULT_OPTIONS.seed),
    matches: positiveIntegerArg(argv, '--matches', DEFAULT_OPTIONS.matches),
    frames: positiveIntegerArg(argv, '--frames', DEFAULT_OPTIONS.frames),
    epochs: nonNegativeIntegerArg(argv, '--epochs', DEFAULT_OPTIONS.epochs),
    batchSize: positiveIntegerArg(argv, '--batch-size', DEFAULT_OPTIONS.batchSize),
    learningRate: numberArg(argv, '--learning-rate', DEFAULT_OPTIONS.learningRate),
    ppoClip: Math.max(0, numberArg(argv, '--ppo-clip', DEFAULT_OPTIONS.ppoClip)),
    temperature: Math.max(0.05, numberArg(argv, '--temperature', DEFAULT_OPTIONS.temperature)),
    discount: clamp01(numberArg(argv, '--discount', DEFAULT_OPTIONS.discount)),
    startStateMode: startStateModeArg(argv, '--start-state-mode', DEFAULT_OPTIONS.startStateMode)
  };
}

export function runPolicyGradientCli(options: PolicyGradientCliOptions): ReturnType<typeof trainPolicyGradientSelfPlay> {
  const weights = options.input
    ? loadWeightsPayload(readFileSync(options.input, 'utf8'))
    : defaultNeuralWeights();
  const result = trainPolicyGradientSelfPlay({
    weights,
    matches: options.matches,
    frames: options.frames,
    epochs: options.epochs,
    batchSize: options.batchSize,
    learningRate: options.learningRate,
    ppoClip: options.ppoClip,
    temperature: options.temperature,
    discount: options.discount,
    startStateMode: options.startStateMode,
    seed: options.seed
  });

  if (options.output) {
    writeFileSync(options.output, serializeWeightsPayload(result.weights, {
      cycle: 0,
      bestCycle: 0,
      selectionScore: result.redGoals - result.blueGoals,
      seed: options.seed,
      replaySamples: 0,
      selfPlaySamples: result.samples,
      loss: result.loss
    }), 'utf8');
  }

  if (options.metricsOutput) {
    writeFileSync(options.metricsOutput, `${JSON.stringify({
      samples: result.samples,
      trainedSamples: result.trainedSamples,
      frames: result.frames,
      redGoals: result.redGoals,
      blueGoals: result.blueGoals,
      loss: result.loss
    }, null, 2)}\n`, 'utf8');
  }

  return result;
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const options = parsePolicyGradientArgs(argv);
    const result = runPolicyGradientCli(options);
    console.log(
      [
        `samples=${result.samples}`,
        `trainedSamples=${result.trainedSamples}`,
        `frames=${result.frames}`,
        `goals=${result.redGoals}-${result.blueGoals}`,
        `loss=${result.loss.toFixed(4)}`
      ].join(' ')
    );
    if (options.output) {
      console.log(`weightsOut=${options.output}`);
    }
    if (options.metricsOutput) {
      console.log(`metricsOut=${options.metricsOutput}`);
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = valueAfter(argv, name);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, Math.floor(numberArg(argv, name, fallback)));
}

function nonNegativeIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(0, Math.floor(numberArg(argv, name, fallback)));
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  return valueAfter(argv, name);
}

function valueAfter(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }
  return argv[index + 1];
}

function startStateModeArg(
  argv: readonly string[],
  name: string,
  fallback: 'open' | 'outcome-curriculum'
): 'open' | 'outcome-curriculum' {
  const value = valueAfter(argv, name);
  return value === 'open' || value === 'outcome-curriculum'
    ? value
    : fallback;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/train-policy-gradient.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/train-policy-gradient.js')) {
  main();
}
