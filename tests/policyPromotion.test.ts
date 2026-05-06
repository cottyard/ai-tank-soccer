import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import {
  parsePromotionLoopArgs,
  runPromotionLoop,
  type PromotionLoopEvaluation
} from '../scripts/promote-policy-gradient';
import type { PolicyGradientCliOptions } from '../scripts/train-policy-gradient';
import { serializeWeightsPayload } from '../scripts/coach-neural';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';

const defaultHistoryPath = join(process.cwd(), 'training-runs/neural-promotion-history.jsonl');
let defaultHistorySnapshot: string | undefined;

beforeEach(() => {
  defaultHistorySnapshot = existsSync(defaultHistoryPath)
    ? readFileSync(defaultHistoryPath, 'utf8')
    : undefined;
});

afterEach(() => {
  const currentHistory = existsSync(defaultHistoryPath)
    ? readFileSync(defaultHistoryPath, 'utf8')
    : undefined;
  const changed = currentHistory !== defaultHistorySnapshot;

  try {
    expect(changed).toBe(false);
  } finally {
    if (defaultHistorySnapshot === undefined) {
      if (existsSync(defaultHistoryPath)) {
        unlinkSync(defaultHistoryPath);
      }
    } else {
      mkdirSync(dirname(defaultHistoryPath), { recursive: true });
      writeFileSync(defaultHistoryPath, defaultHistorySnapshot, 'utf8');
    }
  }
});

describe('policy-gradient promotion loop', () => {
  it('defaults to the current native PPO promotion recipe and fixed runtime gates', () => {
    const options = parsePromotionLoopArgs([]);

    expect(options).toMatchObject({
      bestPath: 'public/models/neural-best.json',
      summaryPath: 'training-runs/neural-promotion-summary-s2026050208.json',
      iterations: 1,
      promote: true,
      seed: 2026050208,
      gateMatches: 4,
      gateFrames: 600,
      standardSeeds: [19, 31, 43, 57, 71],
      holdoutSeeds: [83, 97, 109, 127, 149],
      training: {
        native: true,
        matches: 960,
        frames: 240,
        epochs: 2,
        batchSize: 192,
        learningRate: 0.001,
        ppoClip: 0.12,
        temperature: 1.1,
        discount: 0.996,
        startStateMode: 'mixed',
        advantageBaseline: 'learned',
        actionMode: 'runtime',
        runtimeSurvivorsOnly: false,
        runtimeWrapperWeightMode: 'none',
        opponentMode: 'league',
        leagueCurrentWeight: 1,
        leagueTraditionalWeight: 0.15
      }
    });
  });

  it('promotes a trained candidate only after standard and holdout gates pass', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-promotion-'));
    const bestPath = join(workdir, 'best.json');
    const summaryPath = join(workdir, 'summary.json');
    const baseline = scoredWeights(10);
    const candidate = scoredWeights(12);
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    const result = runPromotionLoop({
      ...parsePromotionLoopArgs([
        '--best',
        bestPath,
        '--summary-output',
        summaryPath,
        '--candidate-output',
        join(workdir, 'candidate.json'),
        '--candidate-metrics-output',
        join(workdir, 'candidate-metrics.json'),
        '--history-output',
        join(workdir, 'history.jsonl')
      ]),
      standardSeeds: [19, 31],
      holdoutSeeds: [83, 97]
    }, {
      train: (trainOptions) => {
        writeFileSync(trainOptions.output ?? join(workdir, 'missing-candidate.json'), weightsJson(candidate, 2), 'utf8');
        if (trainOptions.metricsOutput) {
          writeFileSync(trainOptions.metricsOutput, JSON.stringify({ samples: 8 }), 'utf8');
        }
        return {
          weights: candidate,
          loss: 0.2,
          trainedSamples: 8,
          samples: 8,
          frames: trainOptions.matches * trainOptions.frames,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    });

    expect(result.promoted).toBe(true);
    expect(result.standard.accepted).toBe(true);
    expect(result.holdout.accepted).toBe(true);
    expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: candidate });
    expect(JSON.parse(readFileSync(summaryPath, 'utf8'))).toMatchObject({
      promoted: true,
      candidatePath: join(workdir, 'candidate.json'),
      gates: {
        standard: { accepted: true },
        holdout: { accepted: true }
      }
    });
  });

  it('writes the gate summary before replacing the accepted model', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-promotion-order-'));
    const bestPath = join(workdir, 'best.json');
    const summaryPath = join(workdir, 'summary.json');
    const baseline = scoredWeights(10);
    const candidate = scoredWeights(12);
    let checkedBeforePromote = false;
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    runPromotionLoop({
      ...parsePromotionLoopArgs([
        '--best',
        bestPath,
        '--summary-output',
        summaryPath,
        '--candidate-output',
        join(workdir, 'candidate.json'),
        '--history-output',
        join(workdir, 'history.jsonl')
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83]
    }, {
      train: (trainOptions) => {
        writeFileSync(trainOptions.output ?? join(workdir, 'candidate.json'), weightsJson(candidate, 2), 'utf8');
        return {
          weights: candidate,
          loss: 0.2,
          trainedSamples: 8,
          samples: 8,
          frames: 24,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight,
      beforePromote: () => {
        checkedBeforePromote = true;
        expect(JSON.parse(readFileSync(summaryPath, 'utf8'))).toMatchObject({ promoted: true });
        expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: baseline });
      }
    });

    expect(checkedBeforePromote).toBe(true);
    expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: candidate });
  });

  it('keeps the accepted model unchanged when holdout score fails', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-promotion-reject-'));
    const bestPath = join(workdir, 'best.json');
    const summaryPath = join(workdir, 'summary.json');
    const baseline = scoredWeights(10);
    const candidate = scoredWeights(12);
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    const result = runPromotionLoop({
      ...parsePromotionLoopArgs([
        '--best',
        bestPath,
        '--summary-output',
        summaryPath,
        '--candidate-output',
        join(workdir, 'candidate.json'),
        '--history-output',
        join(workdir, 'history.jsonl')
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83]
    }, {
      train: (trainOptions) => {
        writeFileSync(trainOptions.output ?? join(workdir, 'candidate.json'), weightsJson(candidate, 2), 'utf8');
        return {
          weights: candidate,
          loss: 0.2,
          trainedSamples: 8,
          samples: 8,
          frames: 24,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: (weights, options) => ({
        score: (options.seed ?? 0) >= 80 && weights[0] > 10 ? 9 : weights[0],
        goalDiff: 2,
        ballProgress: 0.3,
        goalsFor: 3,
        goalsAgainst: 1,
        winProxy: 0.75
      })
    });

    expect(result.promoted).toBe(false);
    expect(result.holdout.accepted).toBe(false);
    expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: baseline });
    expect(JSON.parse(readFileSync(summaryPath, 'utf8'))).toMatchObject({
      promoted: false,
      rejectionReason: 'holdout gate failed'
    });
  });

  it('appends compact candidate history after each promotion attempt', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-promotion-history-'));
    const bestPath = join(workdir, 'best.json');
    const historyPath = join(workdir, 'history.jsonl');
    const baseline = scoredWeights(10);
    const firstCandidate = scoredWeights(12);
    const secondCandidate = scoredWeights(9);
    let call = 0;
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');

    const options = {
      ...parsePromotionLoopArgs([
        '--best',
        bestPath,
        '--candidate-output',
        join(workdir, 'candidate.json'),
        '--history-output',
        historyPath
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83]
    };
    const dependencies = {
      train: (trainOptions: PolicyGradientCliOptions) => {
        call += 1;
        const weights = call === 1 ? firstCandidate : secondCandidate;
        writeFileSync(trainOptions.output ?? join(workdir, 'candidate.json'), weightsJson(weights, call + 1), 'utf8');
        return {
          weights,
          loss: 0.2,
          trainedSamples: 8,
          samples: 8,
          frames: 24,
          redGoals: 2,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: scoreByFirstWeight
    };

    runPromotionLoop(options, dependencies);
    runPromotionLoop(options, dependencies);

    const entries = readFileSync(historyPath, 'utf8')
      .trim()
      .split(/\r?\n/)
      .map((line) => JSON.parse(line) as {
        promoted: boolean;
        seed: number;
        candidatePath: string;
        standardScoreDelta: number;
        holdoutScoreDelta: number;
        advantageBaseline: string;
        opponentMode: string;
      });

    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      promoted: true,
      seed: 2026050208,
      candidatePath: join(workdir, 'candidate.json'),
      standardScoreDelta: 2,
      holdoutScoreDelta: 2,
      advantageBaseline: 'learned',
      opponentMode: 'league'
    });
    expect(entries[1]).toMatchObject({
      promoted: false,
      standardScoreDelta: -3,
      holdoutScoreDelta: 0
    });
  });

  it('can gate and promote an existing search candidate without retraining it', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-promotion-existing-'));
    const bestPath = join(workdir, 'best.json');
    const candidatePath = join(workdir, 'candidate.json');
    const historyPath = join(workdir, 'history.jsonl');
    const baseline = scoredWeights(10);
    const candidate = scoredWeights(12);
    let trainCalls = 0;
    writeFileSync(bestPath, weightsJson(baseline, 1), 'utf8');
    writeFileSync(candidatePath, JSON.stringify({
      weights: candidate,
      metadata: {
        seed: 23,
        epochs: 1,
        batchSize: 96,
        learningRate: 0.0008,
        ppoClip: 0.16,
        temperature: 1,
        openStartRatio: 0.35,
        discount: 0.995,
        startStateMode: 'corner-fight',
        advantageBaseline: 'start-team-time',
        actionMode: 'runtime',
        runtimeSurvivorsOnly: true,
        runtimeWrapperWeightMode: 'tactical-downweight',
        opponentMode: 'self',
        trainer: 'rust-policy-gradient'
      }
    }), 'utf8');

    const result = runPromotionLoop({
      ...parsePromotionLoopArgs([
        '--best',
        bestPath,
        '--candidate-input',
        candidatePath,
        '--history-output',
        historyPath
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83]
    }, {
      train: () => {
        trainCalls += 1;
        throw new Error('candidate input should not retrain');
      },
      evaluate: scoreByFirstWeight
    });

    expect(trainCalls).toBe(0);
    expect(result.promoted).toBe(true);
    expect(result.candidatePath).toBe(candidatePath);
    expect(result.candidateMetricsPath).toBeUndefined();
    expect(result.training).toMatchObject({
      seed: 23,
      epochs: 1,
      batchSize: 96,
      learningRate: 0.0008,
      ppoClip: 0.16,
      temperature: 1,
      openStartRatio: 0.35,
      discount: 0.995,
      startStateMode: 'corner-fight',
      advantageBaseline: 'start-team-time',
      runtimeSurvivorsOnly: true,
      runtimeWrapperWeightMode: 'tactical-downweight',
      opponentMode: 'self'
    });
    expect(JSON.parse(readFileSync(bestPath, 'utf8'))).toMatchObject({ weights: candidate });
    const history = JSON.parse(readFileSync(historyPath, 'utf8')) as {
      candidateMetricsPath?: string;
    };
    expect(history.candidateMetricsPath).toBeUndefined();
    expect(history).toMatchObject({
      candidatePath,
      seed: 23,
      advantageBaseline: 'start-team-time',
      opponentMode: 'self',
      runtimeWrapperWeightMode: 'tactical-downweight',
      epochs: 1,
      batchSize: 96,
      learningRate: 0.0008
    });
  });

  it('rejects score improvements that regress win proxy beyond tolerance', () => {
    const current: PromotionLoopEvaluation = {
      score: 100,
      goalDiff: 3,
      goalsFor: 5,
      goalsAgainst: 2,
      ballProgress: 0.1,
      winProxy: 0.8
    };
    const candidate: PromotionLoopEvaluation = {
      ...current,
      score: 120,
      winProxy: 0.6
    };

    const workdir = mkdtempSync(join(tmpdir(), 'soccer-promotion-win-'));
    const bestPath = join(workdir, 'best.json');
    const candidatePath = join(workdir, 'candidate.json');
    writeFileSync(bestPath, weightsJson(scoredWeights(10), 1), 'utf8');

    const result = runPromotionLoop({
      ...parsePromotionLoopArgs([
        '--best',
        bestPath,
        '--candidate-output',
        candidatePath,
        '--history-output',
        join(workdir, 'history.jsonl')
      ]),
      standardSeeds: [19],
      holdoutSeeds: [83],
      maxWinProxyRegression: 0.05
    }, {
      train: (trainOptions) => {
        writeFileSync(trainOptions.output ?? candidatePath, weightsJson(scoredWeights(12), 2), 'utf8');
        return {
          weights: scoredWeights(12),
          loss: 0,
          trainedSamples: 1,
          samples: 1,
          frames: 1,
          redGoals: 0,
          blueGoals: 0,
          finalState: null as never
        };
      },
      evaluate: (weights) => weights[0] > 10 ? candidate : current
    });

    expect(result.promoted).toBe(false);
    expect(result.standard.accepted).toBe(false);
    expect(result.standard.reason).toBe('win proxy regressed');
    if (existsSync(result.summaryPath)) {
      unlinkSync(result.summaryPath);
    }
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

function scoreByFirstWeight(weights: readonly number[]): PromotionLoopEvaluation {
  return {
    score: weights[0],
    goalDiff: 2,
    ballProgress: 0.3,
    goalsFor: 3,
    goalsAgainst: 1,
    winProxy: 0.75
  };
}
