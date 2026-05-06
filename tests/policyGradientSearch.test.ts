import { describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  parsePolicyGradientSearchArgs,
  runPolicyGradientSearch,
  type PolicyGradientSearchEvaluation,
  type PolicyGradientSearchTrace
} from '../scripts/search-policy-gradient';
import type { RuntimeTraceSummary } from '../src/ai/policyGate';
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
      '--training-seeds',
      '101,103',
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
      '--holdout-seeds',
      '83,97',
      '--learning-rates',
      '0.001,0.0005',
      '--epochs-list',
      '1,2',
      '--ppo-clips',
      '0.1',
      '--temperatures',
      '1.0,1.1',
      '--start-state-modes',
      'mixed,open',
      '--open-start-ratios',
      '0.25,0.5',
      '--advantage-baselines',
      'start-team-time,learned',
      '--runtime-survivors-only',
      '--runtime-wrapper-weight-mode',
      'tactical-downweight',
      '--runtime-wrapper-modes',
      'none,runtime-survivors-only,tactical-downweight',
      '--opponent-modes',
      'self,league'
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
      holdoutSeeds: [83, 97],
      training: {
        advantageBaseline: 'start-team-time',
        startStateMode: 'mixed',
        runtimeSurvivorsOnly: true,
        runtimeWrapperWeightMode: 'tactical-downweight',
        opponentMode: 'self'
      },
      grid: {
        trainingSeeds: [101, 103],
        learningRates: [0.001, 0.0005],
        epochs: [1, 2],
        ppoClips: [0.1],
        temperatures: [1, 1.1],
        startStateModes: ['mixed', 'open'],
        openStartRatios: [0.25, 0.5],
        advantageBaselines: ['start-team-time', 'learned'],
        runtimeWrapperModes: ['none', 'runtime-survivors-only', 'tactical-downweight'],
        opponentModes: ['self', 'league']
      }
    });
    expect(options.variants).toHaveLength(576);
  });

  it('searches runtime wrapper handling modes as a variant dimension', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-wrapper-modes-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    const trainedVariants: Array<{
      runtimeSurvivorsOnly: boolean;
      runtimeWrapperWeightMode: string;
      output?: string;
    }> = [];
    writeFileSync(bestPath, weightsJson(scoredWeights(10), 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '59',
        '--learning-rates',
        '0.001',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1',
        '--runtime-wrapper-modes',
        'none,runtime-survivors-only,tactical-downweight'
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        trainedVariants.push({
          runtimeSurvivorsOnly: training.runtimeSurvivorsOnly,
          runtimeWrapperWeightMode: training.runtimeWrapperWeightMode,
          output: training.output
        });
        const score = training.runtimeSurvivorsOnly
          ? 12
          : training.runtimeWrapperWeightMode === 'tactical-downweight'
            ? 11
            : 10;
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(score), training.seed), 'utf8');
        return {
          weights: scoredWeights(score),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    });

    expect(result.rows).toHaveLength(3);
    expect(trainedVariants).toEqual([
      expect.objectContaining({ runtimeSurvivorsOnly: false, runtimeWrapperWeightMode: 'none' }),
      expect.objectContaining({ runtimeSurvivorsOnly: true, runtimeWrapperWeightMode: 'none' }),
      expect.objectContaining({ runtimeSurvivorsOnly: false, runtimeWrapperWeightMode: 'tactical-downweight' })
    ]);
    expect(trainedVariants.map((variant) => variant.output)).toEqual([
      expect.stringContaining('-wrapnone.json'),
      expect.stringContaining('-wrapsurvivors.json'),
      expect.stringContaining('-wraptactical-downweight.json')
    ]);
    expect(result.best.variant.runtimeWrapperMode).toBe('runtime-survivors-only');

    const history = readFileSync(result.historyPath ?? '', 'utf8').trim().split(/\r?\n/).map((line) => JSON.parse(line));
    expect(history[0]).toMatchObject({
      bestRuntimeWrapperMode: 'runtime-survivors-only',
      runtimeSurvivorsOnly: true,
      runtimeWrapperWeightMode: 'none'
    });
  });

  it('trains each variant, ranks promotion-safe candidates by holdout-gate delta, and writes summary history without promoting', () => {
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
        const score = training.learningRate === 0.001 ? 12 : 11;
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
      evaluate: holdoutWeightedScore
    });

    expect(result.best.variant.learningRate).toBe(0.0005);
    expect(result.best.standard.delta.score).toBe(1);
    expect(result.best.holdout.delta.score).toBe(4);
    expect(result.rows).toHaveLength(2);
    expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: baseline });
    expect(JSON.parse(readFileSync(summaryPath, 'utf8'))).toMatchObject({
      bestPath,
      bestCandidatePath: result.best.candidatePath,
      rows: [
        {
          rank: 1,
          variant: { learningRate: 0.0005 },
          standard: { delta: { score: 1 } },
          holdout: { delta: { score: 4 } }
        },
        {
          rank: 2,
          variant: { learningRate: 0.001 },
          standard: { delta: { score: 2 } },
          holdout: { delta: { score: 0 } }
        }
      ]
    });

    const history = readFileSync(historyPath, 'utf8').trim().split(/\r?\n/).map((line) => JSON.parse(line));
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      seed: 23,
      variants: 2,
      bestLearningRate: 0.0005,
      bestStandardScoreDelta: 1,
      bestHoldoutScoreDelta: 4,
      bestCandidatePath: result.best.candidatePath
    });
    expect(existsSync(result.best.candidatePath)).toBe(true);
  });

  it('prefers promotion-safe standard gates before holdout-only gains', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-safe-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    const baseline = scoredWeights(10);
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '31',
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
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        const score = training.learningRate === 0.001 ? 8 : 10;
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
      evaluate: holdoutOnlyGain
    });

    expect(result.best.variant.learningRate).toBe(0.0005);
    expect(result.best.standard.delta.score).toBe(0);
    expect(result.best.holdout.delta.score).toBe(0);
    expect(result.rows[1]).toMatchObject({
      variant: { learningRate: 0.001 },
      standard: { delta: { score: -2 } },
      holdout: { delta: { score: 4 } }
    });
  });

  it('includes categorical training choices in each searched variant', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-grid-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    const baseline = scoredWeights(10);
    const trainedVariants: Array<{
      startStateMode: string;
      advantageBaseline: string;
      opponentMode: string;
      output?: string;
    }> = [];
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '29',
        '--learning-rates',
        '0.001',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1',
        '--start-state-modes',
        'mixed,open',
        '--advantage-baselines',
        'start-team-time,learned',
        '--opponent-modes',
        'self,league'
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        trainedVariants.push({
          startStateMode: training.startStateMode,
          advantageBaseline: training.advantageBaseline,
          opponentMode: training.opponentMode,
          output: training.output
        });
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(12), training.seed), 'utf8');
        return {
          weights: scoredWeights(12),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    });

    expect(result.rows).toHaveLength(8);
    expect(trainedVariants).toEqual([
      expect.objectContaining({ startStateMode: 'mixed', advantageBaseline: 'start-team-time', opponentMode: 'self' }),
      expect.objectContaining({ startStateMode: 'mixed', advantageBaseline: 'start-team-time', opponentMode: 'league' }),
      expect.objectContaining({ startStateMode: 'mixed', advantageBaseline: 'learned', opponentMode: 'self' }),
      expect.objectContaining({ startStateMode: 'mixed', advantageBaseline: 'learned', opponentMode: 'league' }),
      expect.objectContaining({ startStateMode: 'open', advantageBaseline: 'start-team-time', opponentMode: 'self' }),
      expect.objectContaining({ startStateMode: 'open', advantageBaseline: 'start-team-time', opponentMode: 'league' }),
      expect.objectContaining({ startStateMode: 'open', advantageBaseline: 'learned', opponentMode: 'self' }),
      expect.objectContaining({ startStateMode: 'open', advantageBaseline: 'learned', opponentMode: 'league' })
    ]);
    expect(trainedVariants.map((variant) => variant.output)).toEqual([
      expect.stringContaining('-startmixed-baselinestart-team-time-oppself.json'),
      expect.stringContaining('-startmixed-baselinestart-team-time-oppleague.json'),
      expect.stringContaining('-startmixed-baselinelearned-oppself.json'),
      expect.stringContaining('-startmixed-baselinelearned-oppleague.json'),
      expect.stringContaining('-startopen-baselinestart-team-time-oppself.json'),
      expect.stringContaining('-startopen-baselinestart-team-time-oppleague.json'),
      expect.stringContaining('-startopen-baselinelearned-oppself.json'),
      expect.stringContaining('-startopen-baselinelearned-oppleague.json')
    ]);
  });

  it('searches mixed open-start ratios without applying them to non-mixed starts', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-open-ratio-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    const trainedVariants: Array<{
      startStateMode: string;
      openStartRatio?: number;
      output?: string;
    }> = [];
    writeFileSync(bestPath, weightsJson(scoredWeights(10), 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '41',
        '--learning-rates',
        '0.001',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1',
        '--start-state-modes',
        'mixed,open',
        '--open-start-ratios',
        '0.25,0.5'
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        trainedVariants.push({
          startStateMode: training.startStateMode,
          openStartRatio: training.openStartRatio,
          output: training.output
        });
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(12), training.seed), 'utf8');
        return {
          weights: scoredWeights(12),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    });

    expect(result.rows).toHaveLength(3);
    expect(trainedVariants).toEqual([
      expect.objectContaining({ startStateMode: 'mixed', openStartRatio: 0.25 }),
      expect.objectContaining({ startStateMode: 'mixed', openStartRatio: 0.5 }),
      expect.objectContaining({ startStateMode: 'open', openStartRatio: undefined })
    ]);
    expect(trainedVariants.map((variant) => variant.output)).toEqual([
      expect.stringContaining('-startmixed-open0p25-baselinestart-team-time-oppself.json'),
      expect.stringContaining('-startmixed-open0p5-baselinestart-team-time-oppself.json'),
      expect.stringContaining('-startopen-baselinestart-team-time-oppself.json')
    ]);
  });

  it('accepts PowerShell-split comma lists for search dimensions', () => {
    const options = parsePolicyGradientSearchArgs([
      '--seed',
      '43',
      '--training-seeds',
      '101',
      '103',
      '--learning-rates',
      '0.001',
      '--epochs-list',
      '1',
      '--ppo-clips',
      '0.12',
      '--temperatures',
      '1.0',
      '--start-state-modes',
      'mixed',
      'open',
      '--open-start-ratios',
      '0.2',
      '0.35',
      '0.5',
      '--advantage-baselines',
      'start-team-time',
      '--opponent-modes',
      'self'
    ]);

    expect(options.grid.trainingSeeds).toEqual([101, 103]);
    expect(options.grid.startStateModes).toEqual(['mixed', 'open']);
    expect(options.grid.openStartRatios).toEqual([0.2, 0.35, 0.5]);
    expect(options.variants).toHaveLength(8);
  });

  it('can search explicit training seeds without changing output seed bookkeeping', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-seeds-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    const trainedSeeds: number[] = [];
    writeFileSync(bestPath, weightsJson(scoredWeights(10), 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '37',
        '--training-seeds',
        '101,103',
        '--learning-rates',
        '0.001',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1'
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        trainedSeeds.push(training.seed);
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(training.seed === 103 ? 12 : 11), training.seed), 'utf8');
        return {
          weights: scoredWeights(12),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    });

    expect(trainedSeeds).toEqual([101, 103]);
    expect(result.rows).toHaveLength(2);
    expect(result.best.training.seed).toBe(103);
    expect(result.best.candidatePath).toContain('seed103');
    expect(result.seed).toBe(37);
  });

  it('can attach runtime behavior-visibility traces and prefer visible holdout changes after gate ties', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-trace-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    writeFileSync(bestPath, weightsJson(scoredWeights(10), 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--summary-output',
        join(workdir, 'summary.json'),
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '47',
        '--learning-rates',
        '0.001,0.0005',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1',
        '--trace-gate'
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        const score = training.learningRate === 0.001 ? 12 : 11;
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(score), training.seed), 'utf8');
        return {
          weights: scoredWeights(score),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: identicalGateScore,
      trace: actionVisibilityTrace
    });

    expect(result.best.variant.learningRate).toBe(0.001);
    expect(result.best.holdout.trace?.delta.finalActionDistributionChangeCount).toBe(12);
    expect(result.best.holdout.trace?.delta.finalActionDistributionChangeRate).toBe(0.12);
    expect(result.rows[1].holdout.trace?.delta.finalActionDistributionChangeCount).toBe(0);

    const summary = JSON.parse(readFileSync(result.summaryPath, 'utf8'));
    expect(summary.rows[0].holdout.trace.delta).toMatchObject({
      finalActionDistributionChangeCount: 12,
      finalActionDistributionChangeRate: 0.12
    });

    const history = readFileSync(result.historyPath ?? '', 'utf8').trim().split(/\r?\n/).map((line) => JSON.parse(line));
    expect(history[0]).toMatchObject({
      bestHoldoutFinalActionChangeRate: 0.12,
      bestStandardFinalActionChangeRate: 0.12
    });
  });

  it('keeps candidates with higher standard stamina stops behind standard-trace-safe rows', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-policy-search-trace-stamina-'));
    const bestPath = join(workdir, 'best.json');
    const outputDir = join(workdir, 'candidates');
    writeFileSync(bestPath, weightsJson(scoredWeights(10), 1), 'utf8');

    const result = runPolicyGradientSearch({
      ...parsePolicyGradientSearchArgs([
        '--best',
        bestPath,
        '--output-dir',
        outputDir,
        '--history-output',
        join(workdir, 'history.jsonl'),
        '--seed',
        '53',
        '--learning-rates',
        '0.001,0.0005',
        '--epochs-list',
        '1',
        '--ppo-clips',
        '0.12',
        '--temperatures',
        '1.1',
        '--trace-gate'
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      gateMatches: 1,
      gateFrames: 30
    }, {
      train: (training) => {
        const score = training.learningRate === 0.001 ? 12 : 11;
        writeFileSync(training.output ?? join(workdir, 'missing.json'), weightsJson(scoredWeights(score), training.seed), 'utf8');
        return {
          weights: scoredWeights(score),
          loss: 0.1,
          trainedSamples: 4,
          samples: 4,
          frames: training.matches * training.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: identicalGateScore,
      trace: staminaRiskTrace
    });

    expect(result.best.variant.learningRate).toBe(0.0005);
    expect(result.best.standard.trace?.delta.staminaConserveRate).toBe(0);
    expect(result.rows[1]).toMatchObject({
      variant: { learningRate: 0.001 },
      standard: { trace: { delta: { staminaConserveRate: 0.2 } } },
      holdout: { trace: { delta: { finalActionDistributionChangeRate: 0.12 } } }
    });
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

function identicalGateScore(_weights: readonly number[]): PolicyGradientSearchEvaluation {
  return scoreByFirstWeight(scoredWeights(10));
}

function actionVisibilityTrace(weights: readonly number[], options: { seeds?: readonly number[] }): PolicyGradientSearchTrace {
  const current = traceFixture([50, 50, 0, 0, 0, 0, 0, 0, 0]);
  const candidate = weights[0] === 12
    ? traceFixture([38, 50, 12, 0, 0, 0, 0, 0, 0])
    : traceFixture([50, 50, 0, 0, 0, 0, 0, 0, 0]);
  return weights[0] === 10 ? current : {
    ...candidate,
    seeds: (options.seeds ?? []).map((seed) => traceSeedFixture(seed))
  };
}

function staminaRiskTrace(weights: readonly number[], options: { seeds?: readonly number[] }): PolicyGradientSearchTrace {
  const isHoldout = (options.seeds?.[0] ?? 0) >= 80;
  const current = traceFixture([50, 50, 0, 0, 0, 0, 0, 0, 0]);
  if (weights[0] !== 12) {
    return current;
  }

  return {
    ...traceFixture(isHoldout ? [38, 50, 12, 0, 0, 0, 0, 0, 0] : [50, 50, 0, 0, 0, 0, 0, 0, 0]),
    staminaConserves: isHoldout ? 0 : 20
  };
}

function traceFixture(finalActionCounts: readonly number[]): RuntimeTraceSummary {
  const decisions = finalActionCounts.reduce((total, count) => total + count, 0);
  return {
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0,
    decisions,
    policyActionCounts: [...finalActionCounts],
    tacticalActionCounts: [...finalActionCounts],
    finalActionCounts: [...finalActionCounts],
    tacticalRolloutUses: 0,
    tacticalRolloutChanges: 0,
    staminaConserves: 0,
    criticalStaminaRegulations: 0,
    flatPolicies: 0,
    averageStamina: 0,
    averageBallDistance: 0,
    averageBallSpeed: 0,
    averageFinishingPressure: 0,
    averageOwnGoalPressure: 0,
    averageSideWallPressure: 0,
    averageAttackCornerPressure: 0,
    averageOwnCornerPressure: 0,
    seeds: []
  };
}

function traceSeedFixture(seed: number): RuntimeTraceSummary['seeds'][number] {
  return {
    seed,
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0,
    decisions: 0,
    tacticalRolloutUses: 0,
    tacticalRolloutChanges: 0,
    staminaConserves: 0,
    criticalStaminaRegulations: 0,
    flatPolicies: 0
  };
}

function holdoutWeightedScore(weights: readonly number[], options: { seed?: number }): PolicyGradientSearchEvaluation {
  const isHoldout = (options.seed ?? 0) >= 80;
  if (isHoldout && weights[0] === 11) {
    return {
      score: 14,
      goalDiff: 4,
      ballProgress: 0.4,
      goalsFor: 5,
      goalsAgainst: 1,
      winProxy: 0.8
    };
  }
  if (isHoldout && weights[0] === 12) {
    return scoreByFirstWeight(scoredWeights(10));
  }
  return scoreByFirstWeight(weights);
}

function holdoutOnlyGain(weights: readonly number[], options: { seed?: number }): PolicyGradientSearchEvaluation {
  const isHoldout = (options.seed ?? 0) >= 80;
  if (isHoldout && weights[0] === 8) {
    return scoreByFirstWeight(scoredWeights(14));
  }
  return scoreByFirstWeight(weights);
}
