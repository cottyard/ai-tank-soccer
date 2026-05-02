import { execFileSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadWeightsPayload, serializeWeightsPayload } from './coach-neural';
import {
  trainPolicyGradientSelfPlay,
  type PolicyGradientAdvantageBaseline,
  type PolicyGradientStartStateMode,
  type PolicyGradientTrainingResult
} from '../src/ai/policyGradientTraining';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { createInitialState } from '../src/game/model';

declare const process: {
  argv: string[];
  cwd(): string;
  exitCode?: number;
  platform: string;
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
  startStateMode: PolicyGradientStartStateMode;
  advantageBaseline: PolicyGradientAdvantageBaseline;
  actionMode: 'raw' | 'runtime';
  opponentMode: 'self' | 'traditional';
  native: boolean;
  nativeBin?: string;
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
  startStateMode: 'outcome-curriculum',
  advantageBaseline: 'global',
  actionMode: 'raw',
  opponentMode: 'self',
  native: false
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
    startStateMode: startStateModeArg(argv, '--start-state-mode', DEFAULT_OPTIONS.startStateMode),
    advantageBaseline: advantageBaselineArg(argv, '--advantage-baseline', DEFAULT_OPTIONS.advantageBaseline),
    actionMode: actionModeArg(argv, '--action-mode', DEFAULT_OPTIONS.actionMode),
    opponentMode: opponentModeArg(argv, '--opponent-mode', DEFAULT_OPTIONS.opponentMode),
    native: argv.includes('--native'),
    nativeBin: stringArg(argv, '--native-bin')
  };
}

export function runPolicyGradientCli(options: PolicyGradientCliOptions): PolicyGradientTrainingResult {
  if (options.native) {
    return runNativePolicyGradientCli(options);
  }

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
    advantageBaseline: options.advantageBaseline,
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

function runNativePolicyGradientCli(options: PolicyGradientCliOptions): PolicyGradientTrainingResult {
  const nativeBin = resolveNativeTrainer(options.nativeBin);
  const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-gradient-native-'));
  const weightsPath = options.input ?? join(workdir, 'weights.json');
  const outputPath = options.output ?? join(workdir, 'trained.json');
  const metricsPath = options.metricsOutput ?? join(workdir, 'metrics.json');

  if (!options.input) {
    writeFileSync(weightsPath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');
  }

  execFileSync(nativeBin, [
    '--mode',
    'policy-gradient',
    '--weights',
    weightsPath,
    '--output',
    outputPath,
    '--metrics-output',
    metricsPath,
    '--seed',
    String(options.seed),
    '--matches',
    String(options.matches),
    '--frames',
    String(options.frames),
    '--epochs',
    String(options.epochs),
    '--batch-size',
    String(options.batchSize),
    '--learning-rate',
    String(options.learningRate),
    '--ppo-clip',
    String(options.ppoClip),
    '--temperature',
    String(options.temperature),
    '--discount',
    String(options.discount),
    '--start-state-mode',
    options.startStateMode,
    '--advantage-baseline',
    options.advantageBaseline,
    '--action-mode',
    options.actionMode,
    '--opponent-mode',
    options.opponentMode
  ], { stdio: 'pipe' });

  const weights = loadWeightsPayload(readFileSync(outputPath, 'utf8'));
  const metrics = parseNativeMetrics(readFileSync(metricsPath, 'utf8'));
  const finalState = createInitialState();
  finalState.frame = metrics.frames;
  finalState.time = metrics.frames / 30;
  finalState.score = { red: metrics.redGoals, blue: metrics.blueGoals };
  finalState.ball.position = {
    x: metrics.finalBallX,
    y: metrics.finalBallY
  };

  return {
    weights,
    loss: metrics.loss,
    trainedSamples: metrics.trainedSamples,
    samples: metrics.samples,
    frames: metrics.frames,
    redGoals: metrics.redGoals,
    blueGoals: metrics.blueGoals,
    finalState
  };
}

function resolveNativeTrainer(nativeBin: string | undefined): string {
  if (nativeBin) {
    return nativeBin;
  }

  const defaultPath = join(process.cwd(), 'trainer-rust', 'target', 'release', process.platform === 'win32'
    ? 'soccer-policy-trainer.exe'
    : 'soccer-policy-trainer');
  if (existsSync(defaultPath)) {
    return defaultPath;
  }

  throw new Error('Native trainer not found. Build trainer-rust or pass --native-bin.');
}

function parseNativeMetrics(json: string): {
  samples: number;
  trainedSamples: number;
  frames: number;
  redGoals: number;
  blueGoals: number;
  loss: number;
  finalBallX: number;
  finalBallY: number;
} {
  const parsed = JSON.parse(json) as Record<string, unknown>;
  return {
    samples: finiteNumberField(parsed, 'samples'),
    trainedSamples: finiteNumberField(parsed, 'trainedSamples'),
    frames: finiteNumberField(parsed, 'frames'),
    redGoals: finiteNumberField(parsed, 'redGoals'),
    blueGoals: finiteNumberField(parsed, 'blueGoals'),
    loss: finiteNumberField(parsed, 'loss'),
    finalBallX: finiteNumberField(parsed, 'finalBallX'),
    finalBallY: finiteNumberField(parsed, 'finalBallY')
  };
}

function finiteNumberField(record: Record<string, unknown>, field: string): number {
  const value = record[field];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Native trainer metrics missing finite ${field}`);
  }
  return value;
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
  fallback: PolicyGradientStartStateMode
): PolicyGradientStartStateMode {
  const value = valueAfter(argv, name);
  return value === 'open' || value === 'outcome-curriculum' || value === 'mixed'
    ? value
    : fallback;
}

function advantageBaselineArg(
  argv: readonly string[],
  name: string,
  fallback: PolicyGradientAdvantageBaseline
): PolicyGradientAdvantageBaseline {
  const value = valueAfter(argv, name);
  return value === 'global' || value === 'start-team-time'
    ? value
    : fallback;
}

function actionModeArg(
  argv: readonly string[],
  name: string,
  fallback: 'raw' | 'runtime'
): 'raw' | 'runtime' {
  const value = valueAfter(argv, name);
  return value === 'raw' || value === 'runtime'
    ? value
    : fallback;
}

function opponentModeArg(
  argv: readonly string[],
  name: string,
  fallback: 'self' | 'traditional'
): 'self' | 'traditional' {
  const value = valueAfter(argv, name);
  return value === 'self' || value === 'traditional'
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
