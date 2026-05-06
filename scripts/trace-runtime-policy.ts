import { readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { existsSync, mkdirSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';
import { compareRuntimeTraces, traceRuntimePolicy, type RuntimeTraceDelta, type RuntimeTraceSummary } from '../src/ai/policyGate';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type TraceRuntimePolicyOptions = {
  currentPath: string;
  candidatePath?: string;
  outputPath?: string;
  seeds: number[];
  matches: number;
  frames: number;
};

type NamedTrace = {
  name: string;
  path: string;
  trace: RuntimeTraceSummary;
};

const DEFAULT_STANDARD_SEEDS = [19, 31, 43, 57, 71];

export function parseTraceRuntimePolicyArgs(argv: readonly string[]): TraceRuntimePolicyOptions {
  return {
    currentPath: stringArg(argv, '--current') ?? 'public/models/neural-best.json',
    candidatePath: stringArg(argv, '--candidate'),
    outputPath: stringArg(argv, '--output'),
    seeds: seedListArg(argv, '--seeds', DEFAULT_STANDARD_SEEDS),
    matches: positiveIntegerArg(argv, '--matches', 4),
    frames: positiveIntegerArg(argv, '--frames', 600)
  };
}

export function runTraceRuntimePolicy(options: TraceRuntimePolicyOptions): NamedTrace[] {
  const traces = [
    traceNamedPolicy('current', options.currentPath, options)
  ];
  if (options.candidatePath) {
    traces.push(traceNamedPolicy('candidate', options.candidatePath, options));
  }
  if (options.outputPath) {
    ensureParentDirectory(options.outputPath);
    writeFileSync(options.outputPath, `${JSON.stringify({
      generatedAt: new Date().toISOString(),
      options,
      traces,
      delta: traces.length > 1 ? compareRuntimeTraces(traces[1].trace, traces[0].trace) : undefined
    }, null, 2)}\n`, 'utf8');
  }
  return traces;
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const traces = runTraceRuntimePolicy(parseTraceRuntimePolicyArgs(argv));
    for (const row of traces) {
      console.log(formatTrace(row));
    }
    if (traces.length > 1) {
      console.log(formatDelta(compareRuntimeTraces(traces[1].trace, traces[0].trace)));
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function traceNamedPolicy(
  name: string,
  path: string,
  options: TraceRuntimePolicyOptions
): NamedTrace {
  return {
    name,
    path,
    trace: traceRuntimePolicy(loadWeightsPayload(readFileSync(path, 'utf8')), {
      seeds: options.seeds,
      matches: options.matches,
      frames: options.frames
    })
  };
}

function formatTrace(row: NamedTrace): string {
  const trace = row.trace;
  return [
    `${row.name}:`,
    `goals=${trace.goalsFor}-${trace.goalsAgainst}`,
    `score=${trace.score.toFixed(3)}`,
    `win=${trace.winProxy.toFixed(3)}`,
    `bp=${trace.ballProgress.toFixed(3)}`,
    `decisions=${trace.decisions}`,
    `rolloutChange=${rate(trace.tacticalRolloutChanges, trace.decisions).toFixed(3)}`,
    `staminaStop=${rate(trace.staminaConserves, trace.decisions).toFixed(3)}`,
    `criticalReg=${rate(trace.criticalStaminaRegulations, trace.decisions).toFixed(3)}`,
    `avgStamina=${trace.averageStamina.toFixed(3)}`,
    `finalActions=${trace.finalActionCounts.join(',')}`
  ].join(' ');
}

function formatDelta(delta: RuntimeTraceDelta): string {
  return [
    'delta:',
    `score=${formatNumber(delta.score)}`,
    `goalsFor=${formatNumber(delta.goalsFor)}`,
    `goalsAgainst=${formatNumber(delta.goalsAgainst)}`,
    `win=${formatNumber(delta.winProxy)}`,
    `bp=${formatNumber(delta.ballProgress)}`,
    `rolloutChangeRate=${formatNumber(delta.tacticalRolloutChangeRate)}`,
    `staminaStopRate=${formatNumber(delta.staminaConserveRate)}`,
    `criticalRegRate=${formatNumber(delta.criticalStaminaRegulationRate)}`,
    `finalActionChangeRate=${formatNumber(delta.finalActionDistributionChangeRate)}`,
    `finalActionDelta=${delta.finalActionCounts.join(',')}`
  ].join(' ');
}

function formatNumber(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : String(value);
}

function rate(count: number, total: number): number {
  return total > 0 ? count / total : 0;
}

function ensureParentDirectory(path: string): void {
  const parent = dirname(path);
  if (parent && parent !== '.' && !existsSync(parent)) {
    mkdirSync(parent, { recursive: true });
  }
}

function seedListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = listArgValue(argv, name);
  if (!value) {
    return [...fallback];
  }
  const seeds = value.split(',')
    .map((part) => Math.floor(Number(part.trim())))
    .filter((seed) => Number.isFinite(seed));
  return seeds.length > 0 ? seeds : [...fallback];
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  return index === -1 || index === argv.length - 1 ? undefined : argv[index + 1];
}

function listArgValue(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }
  const values: string[] = [];
  for (let cursor = index + 1; cursor < argv.length; cursor += 1) {
    const value = argv[cursor];
    if (value.startsWith('--')) {
      break;
    }
    values.push(value);
  }
  return values.length > 0 ? values.join(',') : undefined;
}

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = stringArg(argv, name);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, Math.floor(numberArg(argv, name, fallback)));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/trace-runtime-policy.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/trace-runtime-policy.js')) {
  main();
}
