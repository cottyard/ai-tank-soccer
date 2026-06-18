import { describe, expect, it } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { parseMacroProbeArgs, runMacroProbe } from '../scripts/probe-runtime-macros';

describe('runtime macro probe', () => {
  it('parses split list options for macro counterfactuals', () => {
    expect(parseMacroProbeArgs([
      '--seed',
      '31',
      '--match',
      '1',
      '--starts',
      '420',
      '456',
      '--durations',
      '18',
      '36',
      '--actions',
      '4',
      '8'
    ])).toMatchObject({
      seed: 31,
      match: 1,
      starts: [420, 456],
      durations: [18, 36],
      actions: [4, 8]
    });
  });

  it('runs deterministic macro rows against the runtime policy', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-macro-probe-'));
    const weightsPath = join(workdir, 'weights.json');
    writeFileSync(weightsPath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');

    const result = runMacroProbe({
      ...parseMacroProbeArgs([]),
      weightsPath,
      seed: 5,
      match: 0,
      frames: 60,
      starts: [12],
      durations: [6],
      actions: [4, 8]
    });

    expect(result.team).toBe('red');
    expect(result.rows).toHaveLength(2);
    expect(Number.isFinite(result.baseline.attackBallX)).toBe(true);
    expect(result.rows.map((row) => row.actionIndex)).toEqual([4, 8]);
  });
});
