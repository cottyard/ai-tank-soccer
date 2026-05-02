import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';
import { evaluateRuntimePolicy, type RuntimeEvaluationResult } from '../src/ai/policyGate';
import { defaultNeuralWeights, NEURAL_WEIGHT_COUNT, type NeuralWeights } from '../src/ai/neuralWeights';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type SearchScope = 'all' | 'output' | 'output-bias';

export type RuntimeHillClimbOptions = {
  input?: string;
  output?: string;
  weights?: NeuralWeights;
  seed: number;
  iterations: number;
  sigma: number;
  gateSeeds: number[];
  matches: number;
  frames: number;
  scope: SearchScope;
  evaluate?: (weights: NeuralWeights) => RuntimeEvaluationResult;
};

export type RuntimeHillClimbResult = {
  weights: number[];
  baselineScore: number;
  bestScore: number;
  accepted: boolean;
  iterations: number;
};

const DEFAULT_OPTIONS: RuntimeHillClimbOptions = {
  seed: 1,
  iterations: 32,
  sigma: 0.02,
  gateSeeds: [19, 31, 43],
  matches: 2,
  frames: 360,
  scope: 'output'
};

export function parseRuntimeHillClimbArgs(argv: readonly string[]): RuntimeHillClimbOptions {
  return {
    input: stringArg(argv, '--input'),
    output: stringArg(argv, '--output'),
    seed: numberArg(argv, '--seed', DEFAULT_OPTIONS.seed),
    iterations: nonNegativeIntegerArg(argv, '--iterations', DEFAULT_OPTIONS.iterations),
    sigma: Math.max(0, numberArg(argv, '--sigma', DEFAULT_OPTIONS.sigma)),
    gateSeeds: seedsArg(argv, '--seeds', DEFAULT_OPTIONS.gateSeeds),
    matches: positiveIntegerArg(argv, '--matches', DEFAULT_OPTIONS.matches),
    frames: positiveIntegerArg(argv, '--frames', DEFAULT_OPTIONS.frames),
    scope: scopeArg(argv, '--scope', DEFAULT_OPTIONS.scope)
  };
}

export function runRuntimeHillClimb(options: RuntimeHillClimbOptions): RuntimeHillClimbResult {
  const baseWeights = resolveWeights(options);
  validateWeights(baseWeights);
  const evaluate = options.evaluate ?? ((weights: NeuralWeights) => evaluateRuntimeSeeds(weights, options));
  const random = createSeededRandom(options.seed);
  let bestWeights = [...baseWeights];
  const baselineScore = evaluate(bestWeights).score;
  let bestScore = baselineScore;
  const mutableIndexes = scopeIndexes(options.scope);
  if (mutableIndexes.length <= 16 && options.sigma > 0) {
    for (const index of mutableIndexes) {
      for (const direction of [-1, 1]) {
        const candidate = [...bestWeights];
        candidate[index] = clamp(candidate[index] + direction * options.sigma, -4, 4);
        const score = evaluate(candidate).score;
        if (score > bestScore) {
          bestWeights = candidate;
          bestScore = score;
        }
      }
    }
  }

  for (let iteration = 0; iteration < options.iterations; iteration += 1) {
    const candidate = [...bestWeights];
    mutateCandidate(candidate, mutableIndexes, options.sigma, random);
    const score = evaluate(candidate).score;
    if (score > bestScore) {
      bestWeights = candidate;
      bestScore = score;
    }
  }

  const accepted = bestScore > baselineScore;
  if (accepted && options.output) {
    writeFileSync(options.output, `${JSON.stringify({
      weights: bestWeights,
      metadata: {
        trainer: 'runtime-hill-climb',
        accepted: true,
        cycle: options.iterations,
        bestCycle: options.iterations,
        selectionScore: bestScore,
        seed: options.seed,
        replaySamples: 0,
        selfPlaySamples: 0,
        loss: baselineScore - bestScore,
        gateSeeds: options.gateSeeds,
        matches: options.matches,
        frames: options.frames,
        scope: options.scope,
        sigma: options.sigma
      }
    }, null, 2)}\n`, 'utf8');
  }

  return {
    weights: bestWeights,
    baselineScore,
    bestScore,
    accepted,
    iterations: options.iterations
  };
}

function evaluateRuntimeSeeds(weights: NeuralWeights, options: RuntimeHillClimbOptions): RuntimeEvaluationResult {
  let score = 0;
  let goalDiff = 0;
  let ballProgress = 0;
  let goalsFor = 0;
  let goalsAgainst = 0;
  let winProxy = 0;

  for (const seed of options.gateSeeds) {
    const result = evaluateRuntimePolicy(weights, {
      seed,
      matches: options.matches,
      frames: options.frames
    });
    score += result.score;
    goalDiff += result.goalDiff;
    ballProgress += result.ballProgress;
    goalsFor += result.goalsFor;
    goalsAgainst += result.goalsAgainst;
    winProxy += result.winProxy;
  }

  const divisor = options.gateSeeds.length || 1;
  return {
    score: score / divisor,
    goalDiff: goalDiff / divisor,
    ballProgress: ballProgress / divisor,
    goalsFor,
    goalsAgainst,
    winProxy: winProxy / divisor
  };
}

function resolveWeights(options: RuntimeHillClimbOptions): number[] {
  if (options.weights) {
    return [...options.weights];
  }
  if (options.input && existsSync(options.input)) {
    return loadWeightsPayload(readFileSync(options.input, 'utf8'));
  }
  return defaultNeuralWeights();
}

function validateWeights(weights: readonly number[]): void {
  if (weights.length !== NEURAL_WEIGHT_COUNT) {
    throw new Error(`Expected ${NEURAL_WEIGHT_COUNT} weights, received ${weights.length}`);
  }
}

function mutateCandidate(
  candidate: number[],
  indexes: readonly number[],
  sigma: number,
  random: () => number
): void {
  if (indexes.length === 0 || sigma === 0) {
    return;
  }
  const edits = Math.max(1, Math.floor(indexes.length * 0.08));
  for (let edit = 0; edit < edits; edit += 1) {
    const index = indexes[Math.floor(random() * indexes.length)];
    candidate[index] = clamp(candidate[index] + gaussian(random) * sigma, -4, 4);
  }
}

function scopeIndexes(scope: SearchScope): number[] {
  const inputCount = 36;
  const hidden = 64;
  const output = 9;
  const outputOffset = hidden * (inputCount + 1) + hidden * (hidden + 1);
  if (scope === 'all') {
    return Array.from({ length: NEURAL_WEIGHT_COUNT }, (_, index) => index);
  }
  if (scope === 'output-bias') {
    return Array.from({ length: output }, (_, action) => outputOffset + action * (hidden + 1) + hidden);
  }
  return Array.from({ length: output * (hidden + 1) }, (_, index) => outputOffset + index);
}

function gaussian(random: () => number): number {
  const u1 = Math.max(1e-9, random());
  const u2 = Math.max(1e-9, random());
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(Math.PI * 2 * u2);
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
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

function seedsArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = valueAfter(argv, name);
  if (!value) {
    return [...fallback];
  }
  const parsed = value.split(',')
    .map((part) => Number(part.trim()))
    .filter((seed) => Number.isFinite(seed))
    .map((seed) => Math.floor(seed));
  return parsed.length > 0 ? parsed : [...fallback];
}

function scopeArg(argv: readonly string[], name: string, fallback: SearchScope): SearchScope {
  const value = valueAfter(argv, name);
  return value === 'all' || value === 'output' || value === 'output-bias'
    ? value
    : fallback;
}

function valueAfter(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }
  return argv[index + 1];
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runRuntimeHillClimb(parseRuntimeHillClimbArgs(argv));
    console.log([
      `baseline=${result.baselineScore.toFixed(3)}`,
      `best=${result.bestScore.toFixed(3)}`,
      `accepted=${result.accepted}`
    ].join(' '));
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/hill-climb-runtime.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/hill-climb-runtime.js')) {
  main();
}
