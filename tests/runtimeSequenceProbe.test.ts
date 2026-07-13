import { describe, expect, it } from 'vitest';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { parseSequenceProbeArgs, runSequenceProbe } from '../scripts/probe-runtime-sequences';

describe('runtime sequence probe', () => {
  it('parses split list options and combination budgets', () => {
    expect(parseSequenceProbeArgs([
      '--seed',
      '71',
      '--match',
      '3',
      '--starts',
      '528',
      '552',
      '--first-durations',
      '6',
      '12',
      '--second-durations',
      '18',
      '--first-actions',
      '7',
      '8',
      '--second-actions',
      '4',
      '8',
      '--max-combinations',
      '5'
    ])).toMatchObject({
      seed: 71,
      match: 3,
      starts: [528, 552],
      firstDurations: [6, 12],
      secondDurations: [18],
      firstActions: [7, 8],
      secondActions: [4, 8],
      maxCombinations: 5
    });
  });

  it('can cap deterministic sequence rows before sorting best results', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-sequence-probe-'));
    const weightsPath = join(workdir, 'weights.json');
    writeFileSync(weightsPath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');

    const result = runSequenceProbe({
      ...parseSequenceProbeArgs([]),
      weightsPath,
      seed: 5,
      match: 0,
      frames: 72,
      starts: [12, 18],
      firstDurations: [6, 12],
      secondDurations: [6],
      firstActions: [4, 8],
      secondActions: [4, 8],
      limit: 10,
      maxCombinations: 3
    });

    expect(result.plannedRows).toBe(16);
    expect(result.completedRows).toBe(3);
    expect(result.truncated).toBe(true);
    expect(result.rows).toHaveLength(3);
    expect(Number.isFinite(result.baseline.attackBallX)).toBe(true);
  });
});
