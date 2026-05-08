import { readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { existsSync, mkdirSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';
import {
  compareRuntimeDecisionTraces,
  compareRuntimeTraces,
  traceRuntimePolicy,
  traceRuntimePolicyDecisions,
  type RuntimeDecisionTraceRecord,
  type RuntimeDecisionTraceComparison,
  type RuntimeTraceDelta,
  type RuntimeTraceSummary
} from '../src/ai/policyGate';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type TraceRuntimePolicyOptions = {
  currentPath: string;
  candidatePath?: string;
  outputPath?: string;
  anchorOutputPath?: string;
  seeds: number[];
  matches: number;
  frames: number;
  decisionAnalysis: boolean;
};

type NamedTrace = {
  name: string;
  path: string;
  trace: RuntimeTraceSummary;
};

type TraceRuntimePolicyResult = {
  traces: NamedTrace[];
  decisionAnalysis?: {
    currentDecisionCount: number;
    candidateDecisionCount: number;
    comparison: RuntimeDecisionTraceComparison;
    currentDecisions: RuntimeDecisionTraceRecord[];
  };
  policyAnchors?: PolicyAnchorExport;
};

export type PolicyAnchorSample = {
  inputs: number[];
  actionIndex: number;
  team: string;
  seed: number;
  match: number;
  frame: number;
  decisionIndex: number;
  tags: string[];
  weight: number;
};

export type PolicyAnchorExport = {
  samples: PolicyAnchorSample[];
};

const DEFAULT_STANDARD_SEEDS = [19, 31, 43, 57, 71];

export function parseTraceRuntimePolicyArgs(argv: readonly string[]): TraceRuntimePolicyOptions {
  return {
    currentPath: stringArg(argv, '--current') ?? 'public/models/neural-best.json',
    candidatePath: stringArg(argv, '--candidate'),
    outputPath: stringArg(argv, '--output'),
    anchorOutputPath: stringArg(argv, '--anchor-output'),
    seeds: seedListArg(argv, '--seeds', DEFAULT_STANDARD_SEEDS),
    matches: positiveIntegerArg(argv, '--matches', 4),
    frames: positiveIntegerArg(argv, '--frames', 600),
    decisionAnalysis: argv.includes('--decision-analysis')
  };
}

export function runTraceRuntimePolicy(options: TraceRuntimePolicyOptions): NamedTrace[] {
  return runTraceRuntimePolicyDetailed(options).traces;
}

export function runTraceRuntimePolicyDetailed(options: TraceRuntimePolicyOptions): TraceRuntimePolicyResult {
  const traces = [
    traceNamedPolicy('current', options.currentPath, options)
  ];
  if (options.candidatePath) {
    traces.push(traceNamedPolicy('candidate', options.candidatePath, options));
  }
  const decisionAnalysis = options.decisionAnalysis && options.candidatePath
    ? analyzeRuntimeDecisionVisibility(options)
    : undefined;
  const policyAnchors = options.anchorOutputPath && decisionAnalysis
    ? buildPolicyAnchorSamples(decisionAnalysis.comparison, decisionAnalysis.currentDecisions)
    : undefined;
  if (options.anchorOutputPath && policyAnchors) {
    ensureParentDirectory(options.anchorOutputPath);
    writeFileSync(options.anchorOutputPath, `${JSON.stringify(policyAnchors, null, 2)}\n`, 'utf8');
  }
  if (options.outputPath) {
    ensureParentDirectory(options.outputPath);
    writeFileSync(options.outputPath, `${JSON.stringify({
      generatedAt: new Date().toISOString(),
      options,
      traces,
      delta: traces.length > 1 ? compareRuntimeTraces(traces[1].trace, traces[0].trace) : undefined,
      decisionAnalysis: decisionAnalysis
        ? {
            currentDecisionCount: decisionAnalysis.currentDecisionCount,
            candidateDecisionCount: decisionAnalysis.candidateDecisionCount,
            comparison: decisionAnalysis.comparison
          }
        : undefined,
      policyAnchors
    }, null, 2)}\n`, 'utf8');
  }
  return {
    traces,
    decisionAnalysis,
    policyAnchors
  };
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runTraceRuntimePolicyDetailed(parseTraceRuntimePolicyArgs(argv));
    const traces = result.traces;
    for (const row of traces) {
      console.log(formatTrace(row));
    }
    if (traces.length > 1) {
      console.log(formatDelta(compareRuntimeTraces(traces[1].trace, traces[0].trace)));
    }
    if (result.decisionAnalysis) {
      console.log(formatDecisionAnalysis(result.decisionAnalysis.comparison));
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

function analyzeRuntimeDecisionVisibility(options: TraceRuntimePolicyOptions): TraceRuntimePolicyResult['decisionAnalysis'] {
  const traceOptions = {
    seeds: options.seeds,
    matches: options.matches,
    frames: options.frames
  };
  const current = traceRuntimePolicyDecisions(
    loadWeightsPayload(readFileSync(options.currentPath, 'utf8')),
    traceOptions
  );
  const candidate = traceRuntimePolicyDecisions(
    loadWeightsPayload(readFileSync(options.candidatePath!, 'utf8')),
    traceOptions
  );

  return {
    currentDecisionCount: current.decisions.length,
    candidateDecisionCount: candidate.decisions.length,
    comparison: compareRuntimeDecisionTraces(candidate.decisions, current.decisions),
    currentDecisions: current.decisions
  };
}

export function buildPolicyAnchorSamples(
  comparison: RuntimeDecisionTraceComparison,
  currentDecisions: readonly RuntimeDecisionTraceRecord[]
): PolicyAnchorExport {
  const decisionsByKey = new Map(currentDecisions.map((record) => [decisionRecordKey(record), record]));
  const seen = new Set<string>();
  const samples: PolicyAnchorSample[] = [];

  for (const divergence of comparison.firstFinalActionDivergences) {
    if (!isLowPressureForwardLossDivergence(divergence)) {
      continue;
    }
    const key = decisionRecordKey(divergence);
    if (seen.has(key)) {
      continue;
    }
    const current = decisionsByKey.get(key);
    if (!current?.inputs || current.inputs.length !== 36 || current.inputs.some((value) => !Number.isFinite(value))) {
      continue;
    }
    seen.add(key);
    samples.push({
      inputs: [...current.inputs],
      actionIndex: 8,
      team: current.controlledTeam,
      seed: current.seed,
      match: current.match,
      frame: current.frame,
      decisionIndex: current.decisionIndex,
      tags: ['policyAnchor', 'lowPressureForwardLoss'],
      weight: 1
    });
  }

  return { samples };
}

function isLowPressureForwardLossDivergence(
  divergence: RuntimeDecisionTraceComparison['firstFinalActionDivergences'][number]
): boolean {
  return divergence.currentFinalActionIndex === 8 &&
    divergence.currentRawPolicyActionIndex === 8 &&
    divergence.candidateFinalActionIndex !== 8 &&
    divergence.staminaRatio > 0.5 &&
    divergence.finishingPressure < 0.2 &&
    divergence.ownGoalPressure < 0.2 &&
    divergence.attackCornerPressure < 0.2 &&
    divergence.ownCornerPressure < 0.2;
}

function decisionRecordKey(record: {
  seed: number;
  match: number;
  controlledTeam: string;
  decisionIndex: number;
}): string {
  return `${record.seed}|${record.match}|${record.controlledTeam}|${record.decisionIndex}`;
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

function formatDecisionAnalysis(comparison: RuntimeDecisionTraceComparison): string {
  const seedSummary = comparison.seeds
    .map((seed) => `seed${seed.seed}:raw=${seed.rawPolicyChanges},final=${seed.finalActionChanges},lost=${seed.lostPolicyChanges},tactical=${seed.lostWithTacticalRollout},stamina=${seed.lostWithStaminaConserve},critical=${seed.lostWithCriticalStamina}`)
    .join(' ');
  return [
    'decisionAnalysis:',
    `compared=${comparison.comparedDecisions}`,
    `aligned=${comparison.alignedComparedDecisions}`,
    `afterDivergence=${comparison.afterFinalActionDivergenceComparedDecisions}`,
    `rawPolicyChanges=${comparison.rawPolicyChanges}`,
    `alignedRaw=${comparison.alignedRawPolicyChanges}`,
    `afterDivergenceRaw=${comparison.afterFinalActionDivergenceRawPolicyChanges}`,
    `finalActionChanges=${comparison.finalActionChanges}`,
    `lostPolicyChanges=${comparison.lostPolicyChanges}`,
    `alignedLost=${comparison.alignedLostPolicyChanges}`,
    `afterDivergenceLost=${comparison.afterFinalActionDivergenceLostPolicyChanges}`,
    `lostTactical=${comparison.lostWithTacticalRollout}`,
    `lostStamina=${comparison.lostWithStaminaConserve}`,
    `lostCritical=${comparison.lostWithCriticalStamina}`,
    seedSummary
  ].filter(Boolean).join(' ');
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
