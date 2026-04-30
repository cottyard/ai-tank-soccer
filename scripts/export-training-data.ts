import { readFileSync, writeFileSync } from 'node:fs';
import { collectCurriculumSamples } from '../src/ai/curriculumTraining';
import { collectSelfPlaySamples } from '../src/ai/selfPlayTraining';
import { defaultNeuralWeights, type NeuralWeights } from '../src/ai/neuralWeights';
import type { LearningSample } from '../src/ai/imitationLearning';
import { loadReplayPayload, loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
  exitCode?: number;
};

export type ExportTrainingDataOptions = {
  input?: string;
  replays: string[];
  output?: string;
  curriculumScenarios: number;
  curriculumFrames: number;
  selfPlayMatches: number;
  selfPlayFrames: number;
  selfPlayRolloutFrames: number;
  seed: number;
};

export type TrainingDataset = {
  metadata: {
    replayFiles: number;
    replaySamples: number;
    curriculumSamples: number;
    selfPlaySamples: number;
    curriculumScenarios: number;
    curriculumFrames: number;
    selfPlayMatches: number;
    selfPlayFrames: number;
    selfPlayRolloutFrames: number;
    seed: number;
  };
  samples: LearningSample[];
};

export function parseExportArgs(argv: readonly string[]): ExportTrainingDataOptions {
  return {
    input: stringArg(argv, '--input'),
    replays: repeatedStringArg(argv, '--replay'),
    output: stringArg(argv, '--output'),
    curriculumScenarios: nonNegativeIntegerArg(argv, '--curriculum-scenarios', 0),
    curriculumFrames: positiveIntegerArg(argv, '--curriculum-frames', 14),
    selfPlayMatches: nonNegativeIntegerArg(argv, '--self-play-matches', 0),
    selfPlayFrames: positiveIntegerArg(argv, '--self-play-frames', 120),
    selfPlayRolloutFrames: positiveIntegerArg(argv, '--self-play-rollout-frames', 10),
    seed: numberArg(argv, '--seed', 1)
  };
}

export function createTrainingDataset(options: {
  weights: NeuralWeights;
  replaySamples?: readonly LearningSample[];
  replayFiles?: number;
  curriculumScenarios?: number;
  curriculumFrames?: number;
  selfPlayMatches?: number;
  selfPlayFrames?: number;
  selfPlayRolloutFrames?: number;
  seed?: number;
}): TrainingDataset {
  const replaySamples = options.replaySamples ?? [];
  const curriculumScenarios = Math.max(0, Math.floor(options.curriculumScenarios ?? 0));
  const curriculumFrames = Math.max(1, Math.floor(options.curriculumFrames ?? 14));
  const selfPlayMatches = Math.max(0, Math.floor(options.selfPlayMatches ?? 0));
  const selfPlayFrames = Math.max(1, Math.floor(options.selfPlayFrames ?? 120));
  const selfPlayRolloutFrames = Math.max(1, Math.floor(options.selfPlayRolloutFrames ?? 10));
  const seed = options.seed ?? 1;
  const curriculum = curriculumScenarios > 0
    ? collectCurriculumSamples({
        weights: options.weights,
        scenarios: curriculumScenarios,
        rolloutFrames: curriculumFrames,
        seed
      }).samples
    : [];
  const selfPlay = selfPlayMatches > 0
    ? collectSelfPlaySamples({
        weights: options.weights,
        opponentWeights: options.weights,
        matches: selfPlayMatches,
        frames: selfPlayFrames,
        rolloutFrames: selfPlayRolloutFrames,
        exploration: 0.04,
        seed: seed + 53_911
      }).samples
    : [];

  return {
    metadata: {
      replayFiles: options.replayFiles ?? 0,
      replaySamples: replaySamples.length,
      curriculumSamples: curriculum.length,
      selfPlaySamples: selfPlay.length,
      curriculumScenarios,
      curriculumFrames,
      selfPlayMatches,
      selfPlayFrames,
      selfPlayRolloutFrames,
      seed
    },
    samples: [
      ...replaySamples.map(cloneSample),
      ...curriculum.map(cloneSample),
      ...selfPlay.map(cloneSample)
    ]
  };
}

export function runExport(options: ExportTrainingDataOptions): TrainingDataset {
  const weights = options.input
    ? loadWeightsPayload(readFileSync(options.input, 'utf8'))
    : defaultNeuralWeights();
  const replaySamples = options.replays.flatMap((path) =>
    loadReplayPayload(readFileSync(path, 'utf8'))
  );
  const dataset = createTrainingDataset({
    weights,
    replaySamples,
    replayFiles: options.replays.length,
    curriculumScenarios: options.curriculumScenarios,
    curriculumFrames: options.curriculumFrames,
    selfPlayMatches: options.selfPlayMatches,
    selfPlayFrames: options.selfPlayFrames,
    selfPlayRolloutFrames: options.selfPlayRolloutFrames,
    seed: options.seed
  });

  const serialized = `${JSON.stringify(dataset, null, 2)}\n`;
  if (options.output) {
    writeFileSync(options.output, serialized, 'utf8');
  } else {
    console.log(serialized);
  }

  return dataset;
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const options = parseExportArgs(argv);
    const dataset = runExport(options);
    console.error(
      `samples=${dataset.samples.length} replay=${dataset.metadata.replaySamples} ` +
        `curriculum=${dataset.metadata.curriculumSamples} selfPlay=${dataset.metadata.selfPlaySamples}`
    );
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function cloneSample(sample: LearningSample): LearningSample {
  return {
    inputs: [...sample.inputs],
    actionIndex: sample.actionIndex,
    team: sample.team,
    frame: sample.frame,
    tags: [...sample.tags],
    weight: sample.weight
  };
}

function repeatedStringArg(argv: readonly string[], name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < argv.length - 1; index += 1) {
    if (argv[index] === name) {
      values.push(argv[index + 1]);
    }
  }
  return values;
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  return index === -1 || index === argv.length - 1 ? undefined : argv[index + 1];
}

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = stringArg(argv, name);
  if (value === undefined) {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, Math.floor(numberArg(argv, name, fallback)));
}

function nonNegativeIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(0, Math.floor(numberArg(argv, name, fallback)));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/export-training-data.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/export-training-data.js')) {
  main();
}
