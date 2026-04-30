import type { GameState, Team } from '../game/model';
import type { TankCommand } from '../game/strategy';
import {
  LearningReplayBuffer,
  type OfflineTrainingOptions,
  type LearningSample,
  createLearningSample,
  trainOfflineFromReplay,
  trainOnlineFromReplay,
} from './imitationLearning';
import { NEURAL_WEIGHT_COUNT, type NeuralWeights } from './neuralWeights';

export type LearningControlMode = 'human' | 'neural';

export type LearningSession = {
  controlModes: { red: LearningControlMode; blue: LearningControlMode };
};

export type LearningSnapshot = {
  enabled: boolean;
  samples: number;
  latestLoss: number;
  modelVersion: number;
};

export type LearningModeOptions = {
  onlineBatchSize?: number;
  learningRate?: number;
  replaySamples?: readonly LearningSample[];
};

export type LearningTickResult = {
  recorded: boolean;
  trainedSamples: number;
  loss: number;
  epochs?: number;
  batches?: number;
};

export type LearnedPolicyMeta = {
  modelVersion: number;
  samples: number;
  latestLoss: number;
};

export type LearnedPolicySave = {
  weights: number[];
  meta: LearnedPolicyMeta;
};

export type ReplayExportOptions = {
  origin?: string;
  exportedAt?: string;
};

type StorageLike = Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>;

export const LEARNED_POLICY_STORAGE_KEY = 'tank-soccer-neural-policy-v1';
export const LEARNED_POLICY_META_KEY = 'tank-soccer-neural-policy-meta-v1';
export const LEARNED_REPLAY_STORAGE_KEY = 'tank-soccer-neural-replay-v1';

export class LearningModeController {
  private enabled = false;
  private weights: number[];
  private latestLoss = 0;
  private modelVersion = 0;
  private readonly replay = new LearningReplayBuffer();
  private readonly onlineBatchSize: number;
  private readonly learningRate: number;

  constructor(initialWeights: NeuralWeights, options: LearningModeOptions = {}) {
    this.weights = [...initialWeights];
    this.onlineBatchSize = options.onlineBatchSize ?? 8;
    this.learningRate = options.learningRate ?? 0.035;
    this.replay.load(options.replaySamples ?? []);
  }

  get currentWeights(): number[] {
    return [...this.weights];
  }

  get snapshot(): LearningSnapshot {
    return {
      enabled: this.enabled,
      samples: this.replay.samples.length,
      latestLoss: this.latestLoss,
      modelVersion: this.modelVersion
    };
  }

  get replaySamples(): LearningSample[] {
    return this.replay.samples.map((sample) => ({
      inputs: [...sample.inputs],
      actionIndex: sample.actionIndex,
      team: sample.team,
      frame: sample.frame,
      tags: [...sample.tags],
      weight: sample.weight
    }));
  }

  startLearningMode(): LearningSession {
    this.enabled = true;
    this.latestLoss = 0;
    return {
      controlModes: { red: 'human', blue: 'neural' }
    };
  }

  stopLearningMode(): void {
    this.enabled = false;
  }

  setWeights(weights: NeuralWeights): void {
    if (weights.length !== NEURAL_WEIGHT_COUNT) {
      throw new Error(`Expected ${NEURAL_WEIGHT_COUNT} learned weights, received ${weights.length}`);
    }
    this.weights = [...weights];
  }

  applyTrainingWeights(weights: NeuralWeights, loss: number): void {
    this.setWeights(weights);
    this.latestLoss = Number.isFinite(loss) ? loss : this.latestLoss;
    this.modelVersion += 1;
  }

  recordAiTick(
    state: Readonly<GameState>,
    humanTeam: Team,
    command: TankCommand
  ): LearningTickResult {
    if (!this.enabled) {
      return { recorded: false, trainedSamples: 0, loss: this.latestLoss };
    }

    const tank = state.tanks.find((candidate) => candidate.team === humanTeam && candidate.index === 0);
    if (!tank) {
      return { recorded: false, trainedSamples: 0, loss: this.latestLoss };
    }

    const recorded = this.replay.add(createLearningSample(state, humanTeam, tank, command));
    const result = trainOnlineFromReplay(this.weights, this.replay, {
      batchSize: this.onlineBatchSize,
      learningRate: this.learningRate,
      seed: state.frame + this.modelVersion * 17
    });
    this.weights = result.weights;
    this.latestLoss = result.loss;
    this.modelVersion = result.modelVersion;

    return {
      recorded,
      trainedSamples: result.trainedSamples,
      loss: result.loss
    };
  }

  trainReplay(options: OfflineTrainingOptions = {}): LearningTickResult {
    const result = trainOfflineFromReplay(this.weights, this.replay, options);
    this.weights = result.weights;
    this.latestLoss = result.loss;
    this.modelVersion = result.modelVersion;
    return {
      recorded: false,
      trainedSamples: result.trainedSamples,
      loss: result.loss,
      epochs: result.epochs,
      batches: result.batches
    };
  }

  reset(weights: NeuralWeights): void {
    this.weights = [...weights];
    this.latestLoss = 0;
    this.modelVersion = 0;
    this.enabled = false;
    this.replay.load([]);
  }
}

export function saveLearnedPolicy(storage: StorageLike, save: LearnedPolicySave): void {
  storage.setItem(LEARNED_POLICY_STORAGE_KEY, JSON.stringify(save.weights));
  storage.setItem(LEARNED_POLICY_META_KEY, JSON.stringify(save.meta));
}

export function loadLearnedPolicy(storage: StorageLike): LearnedPolicySave | null {
  const rawWeights = storage.getItem(LEARNED_POLICY_STORAGE_KEY);
  const rawMeta = storage.getItem(LEARNED_POLICY_META_KEY);
  if (!rawWeights || !rawMeta) {
    return null;
  }

  const weights = JSON.parse(rawWeights) as number[];
  const meta = JSON.parse(rawMeta) as LearnedPolicyMeta;
  if (!Array.isArray(weights) || weights.length !== NEURAL_WEIGHT_COUNT) {
    return null;
  }
  if (!Number.isFinite(meta.modelVersion) || !Number.isFinite(meta.samples) || !Number.isFinite(meta.latestLoss)) {
    return null;
  }

  return { weights, meta };
}

export function clearLearnedPolicy(storage: StorageLike): void {
  storage.removeItem(LEARNED_POLICY_STORAGE_KEY);
  storage.removeItem(LEARNED_POLICY_META_KEY);
  storage.removeItem(LEARNED_REPLAY_STORAGE_KEY);
}

export function saveLearningReplay(storage: StorageLike, samples: readonly LearningSample[]): void {
  storage.setItem(LEARNED_REPLAY_STORAGE_KEY, JSON.stringify(samples));
}

export function serializeReplayExport(
  snapshot: LearningSnapshot,
  samples: readonly LearningSample[],
  options: ReplayExportOptions = {}
): string {
  return `${JSON.stringify(
    {
      exportedAt: options.exportedAt ?? new Date().toISOString(),
      origin: options.origin ?? '',
      meta: snapshot,
      samples
    },
    null,
    2
  )}\n`;
}

export function loadLearningReplay(storage: StorageLike): LearningSample[] {
  const rawReplay = storage.getItem(LEARNED_REPLAY_STORAGE_KEY);
  if (!rawReplay) {
    return [];
  }

  const parsed = JSON.parse(rawReplay) as unknown;
  if (!Array.isArray(parsed)) {
    return [];
  }

  return parsed.flatMap((sample) => isLearningSample(sample) ? [sample] : []);
}

function isLearningSample(sample: unknown): sample is LearningSample {
  if (!sample || typeof sample !== 'object') {
    return false;
  }
  const candidate = sample as LearningSample;
  return Array.isArray(candidate.inputs) &&
    candidate.inputs.every((value) => typeof value === 'number' && Number.isFinite(value)) &&
    typeof candidate.actionIndex === 'number' &&
    (candidate.team === 'red' || candidate.team === 'blue') &&
    typeof candidate.frame === 'number' &&
    Array.isArray(candidate.tags) &&
    typeof candidate.weight === 'number' &&
    Number.isFinite(candidate.weight);
}
