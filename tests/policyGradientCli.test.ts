import { describe, expect, it } from 'vitest';
import { existsSync, unlinkSync } from 'node:fs';
import {
  parsePolicyGradientArgs,
  runPolicyGradientCli
} from '../scripts/train-policy-gradient';

describe('policy-gradient CLI trainer', () => {
  it('parses sparse self-play training options', () => {
    const options = parsePolicyGradientArgs([
      '--input',
      'in.json',
      '--output',
      'out.json',
      '--metrics-output',
      'metrics.json',
      '--seed',
      '41',
      '--matches',
      '7',
      '--frames',
      '90',
      '--epochs',
      '3',
      '--batch-size',
      '16',
      '--learning-rate',
      '0.004',
      '--ppo-clip',
      '0.18',
      '--temperature',
      '1.15',
      '--discount',
      '0.97',
      '--start-state-mode',
      'open',
      '--native',
      '--native-bin',
      'trainer-rust/target/release/soccer-policy-trainer.exe'
    ]);

    expect(options).toMatchObject({
      input: 'in.json',
      output: 'out.json',
      metricsOutput: 'metrics.json',
      seed: 41,
      matches: 7,
      frames: 90,
      epochs: 3,
      batchSize: 16,
      learningRate: 0.004,
      ppoClip: 0.18,
      temperature: 1.15,
      discount: 0.97,
      startStateMode: 'open',
      native: true,
      nativeBin: 'trainer-rust/target/release/soccer-policy-trainer.exe'
    });
  });

  it('writes candidate weights and metrics from a tiny run', () => {
    const output = 'training-runs/test-policy-gradient-cli-weights.json';
    const metricsOutput = 'training-runs/test-policy-gradient-cli-metrics.json';

    try {
      const result = runPolicyGradientCli({
        ...parsePolicyGradientArgs([]),
        output,
        metricsOutput,
        seed: 7,
        matches: 1,
        frames: 12,
        epochs: 1,
        batchSize: 4
      });

      expect(result.samples).toBeGreaterThan(0);
      expect(existsSync(output)).toBe(true);
      expect(existsSync(metricsOutput)).toBe(true);
    } finally {
      if (existsSync(output)) {
        unlinkSync(output);
      }
      if (existsSync(metricsOutput)) {
        unlinkSync(metricsOutput);
      }
    }
  });
});
