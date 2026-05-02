import { describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  parsePolicyGradientSearchArgs,
  runPolicyGradientSearch,
  type PolicyGradientSearchEvaluation
} from '../scripts/search-policy-gradient';
import { serializeWeightsPayload } from '../scripts/coach-neural';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';

describe('policy-gradient promotion search', () => {
  it('parses a compact hyperparameter grid for short promotion searches', () => {
    const options = parsePolicyGradientSearchArgs([
      '--best',
      'best.json',
      '--output-dir',
      'training-runs/search',
      '--summary-output',
      'training-runs/search-summary.json',
      '--history-output',
      'training-runs/search-history.jsonl',
      '--seed',
      '17',
      '--matches',
      '8',
      '--frames',
      '48',
      '--gate-matches',
      '2',
      '--gate-frames',
      '90',
      '--standard-seeds',
      '3,5',
      '--learning-rates',
      '0.001,0.0005',
      '--epochs-list',
      '1,2',
      '--ppo-clips',
      '0.1',
      '--temperatures',
      '1.0,1.1',
      '--advantage-baseline',
      'start-team-time',
      '--opponent-mode',
      'self'
    ]);

    expect(options).toMatchObject({
      bestPath: 'best.json',
      outputDir: 'training-runs/search',
      summaryPath: 'training-runs/search-summary.json',
      historyPath: 'training-runs/search-history.jsonl',
      seed: 17,
      matches: 8,
      frames: 48,
      gateMatches: 2,
      gateFrames: 90,
      standardSeeds: [3, 5],
      training: {
        advantageBaseline: 'start-team-time',
        opponentMode: 'self'
      },
      grid: {
        learningRates: [0.001, 0.0005],
        epochs: [1, 2],
        ppoClips: [0.1],
        temperatures: [1, 1.1]
      }
    });
    expect(options.variants).toHaveLength(8);
  });

  it('trains each variant, ranks by standard-gate delta, and writes summary history without promoting', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    const summaryPath = join(workdir, 'summary.json');
    const historyPath = join(workdir, 'history.jsonl');
    const baseline = scoredWeights(10);
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--summary-output',
        summaryPath,
        '--history-output',
        historyPath,
        '--seed',
        '23',
        '--learning-rates',
        '0.001,0.0005',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1'
      ]),
      standardSeeds: [19],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        const score = training.learningRate === 0.001 ? 12 : 8;
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(score), training.seed), 'utf8');
        return {
          weights: scoredWeights(score),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: score,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    });

    expect(result.best.variant.learningRate).toBe(0.001);
    expect(result.best.standard.delta.score).toBe(2);
    expect(result.rows).toHaveLength(2);
    expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: baseline });
    expect(JSON.parse(readFileSync(summaryPath, 'utf8'))).toMatchObject({
      bestPath,
      bestCandidatePath: result.best.candidatePath,
      rows: [
        { rank: 1, variant: { learningRate: 0.001 }, standard: { delta: { score: 2 } } },
        { rank: 2, variant: { learningRate: 0.0005 }, standard: { delta: { score: -2 } } }
      ]
    });

    const history = readFileSync(historyPath, 'utf8').trim().split(/\r?\n/).map((line) => JSON.parse(line));
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      seed: 23,
      variants: 2,
      bestLearningRate: 0.001,
      bestStandardScoreDelta: 2,
      bestCandidatePath: result.best.candidatePath
    });
    expect(existsSync(result.best.candidatePath)).toBe(true);
  });
});

function scoredWeights(score: number): number[] {
  const weights = defaultNeuralWeights();
  weights[0] = score;
  return weights;
}

function weightsJson(weights: number[], seed: number): string {
  return serializeWeightsPayload(weights, {
    cycle: 0,
    bestCycle: 0,
    selectionScore: weights[0],
    seed,
    replaySamples: 0,
    selfPlaySamples: 0,
    loss: 0
  });
}

function scoreByFirstWeight(weights: readonly number[]): PolicyGradientSearchEvaluation {
  return {
    score: weights[0],
    goalDiff: weights[0] - 10,
    ballProgress: 0.25,
    goalsFor: weights[0],
    goalsAgainst: 0,
    winProxy: weights[0] > 10 ? 0.75 : 0.5
  };
}
