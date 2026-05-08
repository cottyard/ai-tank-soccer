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
      'corner-fight',
      '--open-start-ratio',
      '0.35',
      '--advantage-baseline',
      'learned',
      '--action-mode',
      'runtime',
      '--runtime-survivors-only',
      '--runtime-wrapper-weight-mode',
      'tactical-downweight',
      '--runtime-tactical-rewrite-weight',
      '0.2',
      '--action-retention-weight',
      '0.35',
      '--early-forward-safety-weight',
      '0.15',
      '--early-forward-anchor-weight',
      '0.4',
      '--opponent-mode',
      'traditional',
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
      startStateMode: 'corner-fight',
      openStartRatio: 0.35,
      advantageBaseline: 'learned',
      actionMode: 'runtime',
      runtimeSurvivorsOnly: true,
      runtimeWrapperWeightMode: 'tactical-downweight',
      runtimeTacticalRewriteWeight: 0.2,
      actionRetentionWeight: 0.35,
      earlyForwardSafetyWeight: 0.15,
      earlyForwardAnchorWeight: 0.4,
      opponentMode: 'traditional',
      native: true,
      nativeBin: 'trainer-rust/target/release/soccer-policy-trainer.exe'
    });
  });

  it('parses native league opponent sampling options', () => {
    const options = parsePolicyGradientArgs([
      '--native',
      '--opponent-mode',
      'league',
      '--league-opponent-weights',
      'training-runs/recent-a.json',
      '--league-opponent-weights',
      'training-runs/historical-b.json',
      '--league-current-weight',
      '1.5',
      '--league-traditional-weight',
      '0.2'
    ]);

    expect(options).toMatchObject({
      native: true,
      opponentMode: 'league',
      leagueOpponentWeights: [
        'training-runs/recent-a.json',
        'training-runs/historical-b.json'
      ],
      leagueCurrentWeight: 1.5,
      leagueTraditionalWeight: 0.2
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
