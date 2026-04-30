import { describe, expect, it } from 'vitest';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { createTrainingDataset, parseExportArgs } from '../scripts/export-training-data';

describe('training data export', () => {
  it('parses repeated replay files and curriculum options', () => {
    const options = parseExportArgs([
      '--input',
      'base.json',
      '--replay',
      'human-a.json',
      '--replay',
      'human-b.json',
      '--curriculum-scenarios',
      '12',
      '--curriculum-frames',
      '9',
      '--self-play-matches',
      '3',
      '--self-play-frames',
      '18',
      '--self-play-rollout-frames',
      '5',
      '--seed',
      '77',
      '--output',
      'samples.json'
    ]);

    expect(options).toMatchObject({
      input: 'base.json',
      replays: ['human-a.json', 'human-b.json'],
      curriculumScenarios: 12,
      curriculumFrames: 9,
      selfPlayMatches: 3,
      selfPlayFrames: 18,
      selfPlayRolloutFrames: 5,
      seed: 77,
      output: 'samples.json'
    });
  });

  it('combines replay and curriculum samples with metadata', () => {
    const replaySample = {
      inputs: Array.from({ length: 36 }, (_, index) => index / 100),
      actionIndex: 8,
      team: 'red' as const,
      frame: 24,
      tags: ['finish' as const],
      weight: 1.5
    };

    const dataset = createTrainingDataset({
      weights: defaultNeuralWeights(),
      replaySamples: [replaySample],
      curriculumScenarios: 4,
      curriculumFrames: 2,
      selfPlayMatches: 2,
      selfPlayFrames: 6,
      selfPlayRolloutFrames: 1,
      seed: 19
    });

    expect(dataset.samples.length).toBeGreaterThan(1);
    expect(dataset.samples[0]).toEqual(replaySample);
    expect(dataset.metadata.replaySamples).toBe(1);
    expect(dataset.metadata.curriculumSamples).toBeGreaterThan(0);
    expect(dataset.metadata.selfPlaySamples).toBeGreaterThan(0);
    expect(dataset.metadata.curriculumSamples + dataset.metadata.selfPlaySamples).toBe(dataset.samples.length - 1);
    expect(dataset.metadata.curriculumScenarios).toBe(4);
    expect(dataset.metadata.selfPlayMatches).toBe(2);
  });
});
