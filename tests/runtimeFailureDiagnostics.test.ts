import { describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import {
  parseRuntimeFailureDiagnosticArgs,
  runRuntimeFailureDiagnostics
} from '../scripts/diagnose-runtime-failures';

describe('runtime failure diagnostics', () => {
  it('parses match diagnostic CLI options', () => {
    const options = parseRuntimeFailureDiagnosticArgs([
      '--weights',
      'weights.json',
      '--opponent',
      'accepted-no-rollout',
      '--opponent-weights',
      'opponent.json',
      '--output',
      'training-runs/diagnostics.json',
      '--seeds',
      '19',
      '31',
      '--matches',
      '2',
      '--frames',
      '90',
      '--include-wins',
      '--tail-decisions',
      '7'
    ]);

    expect(options).toEqual({
      weightsPath: 'weights.json',
      opponentKind: 'accepted-no-rollout',
      opponentWeightsPath: 'opponent.json',
      tacticalRollout: true,
      pairedStarts: false,
      outputPath: 'training-runs/diagnostics.json',
      seeds: [19, 31],
      matches: 2,
      frames: 90,
      includeWins: true,
      tailDecisions: 7
    });
  });

  it('summarizes each match with gate-compatible outcomes and decision windows', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-failure-diagnostic-'));
    const weightsPath = join(workdir, 'weights.json');
    writeFileSync(weightsPath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');

    const result = runRuntimeFailureDiagnostics({
      ...parseRuntimeFailureDiagnosticArgs([]),
      weightsPath,
      seeds: [5],
      matches: 2,
      frames: 60,
      includeWins: true,
      tailDecisions: 3
    });

    expect(result.summary.matches).toBe(2);
    expect(result.summary.wins + result.summary.draws + result.summary.losses).toBe(2);
    expect(result.matches).toHaveLength(2);
    expect(result.matches[0]).toMatchObject({
      seed: 5,
      match: 0,
      team: 'red'
    });
    expect(result.matches[1]).toMatchObject({
      seed: 5,
      match: 1,
      team: 'blue'
    });
    expect(result.matches[0].tailActions.length).toBeLessThanOrEqual(3);
    expect(result.matches[0].allDecisions.finalActionCounts).toHaveLength(9);
    expect(result.matches[0].earlyDecisions.finalActionCounts).toHaveLength(9);
    expect(Number.isFinite(result.matches[0].ballProgress)).toBe(true);
    expect(Number.isFinite(result.matches[0].opportunity.peakFinishingPressure)).toBe(true);
    expect(result.matches[0].opportunity.closeFinishDecisions).toBeGreaterThanOrEqual(0);
  });

  it('writes JSON diagnostics for later heuristic-learning inspection', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-failure-output-'));
    const weightsPath = join(workdir, 'weights.json');
    const outputPath = join(workdir, 'diagnostics.json');
    writeFileSync(weightsPath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');

    const result = runRuntimeFailureDiagnostics({
      ...parseRuntimeFailureDiagnosticArgs([]),
      weightsPath,
      outputPath,
      seeds: [5],
      matches: 1,
      frames: 30,
      includeWins: true
    });

    expect(existsSync(outputPath)).toBe(true);
    const payload = JSON.parse(readFileSync(outputPath, 'utf8')) as typeof result;
    expect(payload.summary.matches).toBe(1);
    expect(payload.matches).toEqual(result.matches);
  });
});
