import { describe, expect, it } from 'vitest';
import { unlinkSync, writeFileSync } from 'node:fs';
import {
  formatCycleMetric,
  loadReplayPayload,
  loadWeightsPayload,
  parseCoachArgs,
  runCoach,
  runEvaluationSuite,
  serializeWeightsPayload
} from '../scripts/coach-neural';
import { NEURAL_WEIGHT_COUNT, defaultNeuralWeights } from '../src/ai/neuralWeights';

describe('coach neural harness', () => {
  it('parses deterministic CLI options with compact defaults', () => {
    const options = parseCoachArgs([
      '--seed',
      '7',
      '--cycles',
      '2',
      '--eval-matches',
      '3',
      '--frames',
      '90',
      '--self-play-matches',
      '4',
      '--self-play-frames',
      '60',
      '--epochs',
      '5',
      '--accept-opponent',
      'league',
      '--gate-seeds',
      '3',
      '--curriculum-scenarios',
      '24',
      '--curriculum-frames',
      '10',
      '--rl-cycles',
      '2',
      '--rl-matches',
      '5',
      '--rl-frames',
      '72',
      '--rl-epochs',
      '3',
      '--rl-batch-size',
      '16',
      '--rl-learning-rate',
      '0.004',
      '--rl-ppo-clip',
      '0.18',
      '--rl-temperature',
      '1.2',
      '--rl-discount',
      '0.98',
      '--rl-start-state-mode',
      'open',
      '--input',
      'in.json',
      '--replay',
      'replay.json',
      '--output',
      'out.json'
    ]);

    expect(options).toMatchObject({
      seed: 7,
      cycles: 2,
      evalMatches: 3,
      frames: 90,
      selfPlayMatches: 4,
      selfPlayFrames: 60,
      epochs: 5,
      acceptOpponent: 'league',
      gateSeeds: 3,
      curriculumScenarios: 24,
      curriculumFrames: 10,
      rlCycles: 2,
      rlMatches: 5,
      rlFrames: 72,
      rlEpochs: 3,
      rlBatchSize: 16,
      rlLearningRate: 0.004,
      rlPpoClip: 0.18,
      rlTemperature: 1.2,
      rlDiscount: 0.98,
      rlStartStateMode: 'open',
      input: 'in.json',
      replay: 'replay.json',
      output: 'out.json'
    });
  });

  it('round-trips plain and object JSON weight payloads', () => {
    const weights = defaultNeuralWeights();
    const serialized = serializeWeightsPayload(weights, {
      cycle: 3,
      bestCycle: 3,
      selectionScore: 12.5,
      seed: 9,
      replaySamples: 11,
      selfPlaySamples: 22,
      loss: 0.125
    });

    expect(loadWeightsPayload(JSON.stringify(weights))).toEqual(weights);
    expect(loadWeightsPayload(serialized)).toEqual(weights);
    expect(loadWeightsPayload(JSON.stringify({ weights }))).toEqual(weights);
  });

  it('loads replay payloads with stamina-bearing policy samples', () => {
    const sample = {
      inputs: Array.from({ length: 36 }, (_, index) => index === 7 ? 0.42 : 0),
      actionIndex: 4,
      team: 'red' as const,
      frame: 12,
      tags: ['lowStamina' as const],
      weight: 1.25
    };

    expect(loadReplayPayload(JSON.stringify({ samples: [sample] }))).toEqual([sample]);
    expect(loadReplayPayload(JSON.stringify([sample]))).toEqual([sample]);
    expect(() => loadReplayPayload(JSON.stringify({ samples: [{ ...sample, inputs: [0] }] }))).toThrow(
      'Expected replay sample inputs'
    );
  });

  it('evaluates candidate against idle, traditional, default neural, and self neural opponents', () => {
    const rows = runEvaluationSuite(defaultNeuralWeights(), {
      seed: 3,
      matches: 1,
      frames: 12
    });

    expect(rows.map((row) => row.opponent)).toEqual([
      'idle',
      'traditional',
      'neural-default',
      'neural-self'
    ]);
    expect(rows.every((row) => Number.isFinite(row.score))).toBe(true);
    expect(rows.every((row) => Number.isFinite(row.goalDiff))).toBe(true);
    expect(rows.every((row) => Number.isFinite(row.goalsFor))).toBe(true);
    expect(rows.every((row) => Number.isFinite(row.goalsAgainst))).toBe(true);
    expect(rows.every((row) => Number.isFinite(row.ballProgress))).toBe(true);
  });

  it('can include the current accepted neural model as a fixed league opponent', () => {
    const rows = runEvaluationSuite(defaultNeuralWeights(), {
      seed: 5,
      matches: 1,
      frames: 12,
      fixedNeuralOpponentWeights: defaultNeuralWeights()
    });

    expect(rows.map((row) => row.opponent)).toEqual([
      'idle',
      'traditional',
      'neural-default',
      'neural-self',
      'neural-current'
    ]);
    expect(rows.every((row) => Number.isFinite(row.score))).toBe(true);
  });

  it('formats one compact metric line per evaluated opponent', () => {
    const line = formatCycleMetric({
      cycle: 1,
      phase: 'eval',
      opponent: 'idle',
      score: 12.3456,
      goalDiff: 1.25,
      goalsFor: 2,
      goalsAgainst: 1,
      ballProgress: 0.4567,
      winProxy: 0.3333,
      replaySamples: 0,
      selfPlaySamples: 18,
      loss: 0.9876,
      accepted: false,
      acceptOpponent: 'traditional',
      acceptScore: 10.25,
      acceptBaseline: 12.5
    });

    expect(line).toBe(
      'cycle=1 phase=eval opponent=idle score=12.346 goalDiff=1.250 goals=2-1 ballProgress=0.457 winProxy=0.333 replaySamples=0 selfPlaySamples=18 loss=0.988 accepted=false acceptOpponent=traditional acceptScore=10.250 acceptBaseline=12.500 acceptDelta=-2.250'
    );
  });

  it('runs a tiny self-play cycle and reports replay-trained visits separately from self-play samples', () => {
    const result = runCoach({
      ...parseCoachArgs([]),
      seed: 11,
      cycles: 1,
      evalMatches: 1,
      frames: 6,
      selfPlayMatches: 1,
      selfPlayFrames: 6,
      epochs: 1,
      batchSize: 8
    });
    const trainMetric = result.metrics.find((metric) => metric.phase === 'train');

    expect(trainMetric?.selfPlaySamples).toBeGreaterThan(0);
    expect(trainMetric?.replaySamples).toBeGreaterThan(0);
    expect(trainMetric?.loss).toBeGreaterThanOrEqual(0);
    expect(result.metadata.selfPlaySamples).toBe(trainMetric?.selfPlaySamples);
    expect(result.metadata.replaySamples).toBe(trainMetric?.replaySamples);
    expect(result.metadata.bestCycle).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.metadata.selectionScore)).toBe(true);
  });

  it('rejects a gated cycle when traditional weighted score does not improve', () => {
    const result = runCoach({
      ...parseCoachArgs(['--accept-opponent', 'traditional']),
      seed: 13,
      cycles: 1,
      evalMatches: 1,
      frames: 6,
      selfPlayMatches: 1,
      selfPlayFrames: 6,
      epochs: 0,
      batchSize: 8
    });
    const trainMetric = result.metrics.find((metric) => metric.phase === 'train');

    expect(trainMetric?.accepted).toBe(false);
    expect(trainMetric?.acceptOpponent).toBe('traditional');
    expect(trainMetric?.acceptScore).toBeCloseTo(trainMetric?.acceptBaseline ?? NaN, 8);
    expect(result.weights).toEqual(defaultNeuralWeights());
    expect(result.metadata.bestCycle).toBe(0);
    expect(result.metadata.acceptOpponent).toBe('traditional');
    expect(result.metadata.gateSeeds).toBe(1);
    expect(result.metadata.acceptedCycles).toBe(0);
    expect(result.metadata.rejectedCycles).toBe(1);
  });

  it('can gate candidates against a weighted league including the current neural model', () => {
    const result = runCoach({
      ...parseCoachArgs(['--accept-opponent', 'league']),
      seed: 23,
      cycles: 1,
      evalMatches: 1,
      frames: 6,
      selfPlayMatches: 1,
      selfPlayFrames: 6,
      epochs: 0,
      batchSize: 8
    });
    const trainMetric = result.metrics.find((metric) => metric.phase === 'train');

    expect(trainMetric?.accepted).toBe(false);
    expect(trainMetric?.acceptOpponent).toBe('league');
    expect(trainMetric?.acceptScore).toBeCloseTo(trainMetric?.acceptBaseline ?? NaN, 8);
    expect(result.weights).toEqual(defaultNeuralWeights());
  });

  it('can run gated curriculum training without replacing weights on a failed gate', () => {
    const result = runCoach({
      ...parseCoachArgs(['--accept-opponent', 'traditional']),
      seed: 29,
      cycles: 0,
      curriculumScenarios: 4,
      curriculumFrames: 2,
      evalMatches: 1,
      frames: 6,
      epochs: 1,
      batchSize: 8,
      learningRate: 0.003
    });
    const trainMetric = result.metrics.find((metric) => metric.phase === 'train');

    expect(trainMetric?.selfPlaySamples).toBe(4);
    expect(trainMetric?.replaySamples).toBeGreaterThan(0);
    expect(trainMetric?.accepted).toBe(false);
    expect(result.weights).toEqual(defaultNeuralWeights());
    expect(result.metadata.rejectedCycles).toBe(1);
  });

  it('can run a tiny sparse-reward policy-gradient cycle through the coach harness', () => {
    const result = runCoach({
      ...parseCoachArgs(['--accept-opponent', 'league']),
      seed: 31,
      cycles: 0,
      rlCycles: 1,
      rlMatches: 1,
      rlFrames: 12,
      rlEpochs: 1,
      rlBatchSize: 4,
      rlLearningRate: 0.004,
      evalMatches: 1,
      frames: 6
    });
    const trainMetric = result.metrics.find((metric) => metric.phase === 'train');

    expect(trainMetric?.selfPlaySamples).toBeGreaterThan(0);
    expect(trainMetric?.replaySamples).toBe(0);
    expect(trainMetric?.accepted).toBe(false);
    expect(trainMetric?.acceptOpponent).toBe('league');
    expect(result.weights).toEqual(defaultNeuralWeights());
    expect(result.metadata.rejectedCycles).toBe(1);
  });

  it('does not keep replay pretraining that fails the traditional gate', () => {
    const path = 'training-runs/test-replay-reject.json';
    const sample = {
      inputs: Array.from({ length: 36 }, () => 0),
      actionIndex: 4,
      team: 'red' as const,
      frame: 0,
      tags: [],
      weight: 1
    };
    writeFileSync(path, JSON.stringify({ samples: [sample] }), 'utf8');

    const result = (() => {
      try {
        return runCoach({
          ...parseCoachArgs(['--accept-opponent', 'traditional']),
          replay: path,
          seed: 17,
          cycles: 0,
          evalMatches: 1,
          frames: 6,
          selfPlayMatches: 1,
          selfPlayFrames: 6,
          epochs: 1,
          batchSize: 8,
          learningRate: 0.02
        });
      } finally {
        unlinkSync(path);
      }
    })();

    expect(result.weights).toEqual(defaultNeuralWeights());
    expect(result.metadata.replayAccepted).toBe(false);
    expect(result.metadata.replaySamples).toBe(1);
  });

  it('rejects malformed weight payloads with the expected network size', () => {
    expect(() => loadWeightsPayload(JSON.stringify([1, 2, 3]))).toThrow(
      `Expected ${NEURAL_WEIGHT_COUNT} weights`
    );
    expect(() => loadWeightsPayload(JSON.stringify({ weights: 'bad' }))).toThrow(
      'Expected weights JSON'
    );
  });
});
