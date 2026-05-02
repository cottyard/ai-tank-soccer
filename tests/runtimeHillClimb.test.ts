import { existsSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import {
  parseRuntimeHillClimbArgs,
  runRuntimeHillClimb
} from '../scripts/hill-climb-runtime';

describe('runtime gate hill-climb trainer', () => {
  it('parses deterministic search options', () => {
    const options = parseRuntimeHillClimbArgs([
      '--input',
      'in.json',
      '--output',
      'out.json',
      '--seed',
      '17',
      '--iterations',
      '9',
      '--sigma',
      '0.03',
      '--seeds',
      '19,31',
      '--matches',
      '3',
      '--frames',
      '240',
      '--scope',
      'output'
    ]);

    expect(options).toMatchObject({
      input: 'in.json',
      output: 'out.json',
      seed: 17,
      iterations: 9,
      sigma: 0.03,
      gateSeeds: [19, 31],
      matches: 3,
      frames: 240,
      scope: 'output'
    });
  });

  it('writes an accepted candidate only when runtime score improves', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-hill-climb-'));
    const output = join(workdir, 'candidate.json');
    const weights = defaultNeuralWeights();
    const outputBiasStart = 64 * (36 + 1) + 64 * (64 + 1) + 8 * (64 + 1) + 64;

    const result = runRuntimeHillClimb({
      weights,
      output,
      seed: 3,
      iterations: 4,
      sigma: 0.05,
      gateSeeds: [1],
      matches: 1,
      frames: 12,
      scope: 'output-bias',
      evaluate: (candidate) => ({
        score: candidate[outputBiasStart] - weights[outputBiasStart],
        goalDiff: 0,
        ballProgress: 0,
        goalsFor: 0,
        goalsAgainst: 0,
        winProxy: 0.5
      })
    });

    expect(result.accepted).toBe(true);
    expect(result.bestScore).toBeGreaterThan(result.baselineScore);
    expect(existsSync(output)).toBe(true);

    const payload = JSON.parse(readFileSync(output, 'utf8')) as {
      metadata?: { trainer?: string; accepted?: boolean };
    };
    expect(payload.metadata?.trainer).toBe('runtime-hill-climb');
    expect(payload.metadata?.accepted).toBe(true);
  });
});
