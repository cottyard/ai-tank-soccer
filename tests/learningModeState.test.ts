import { describe, expect, it } from 'vitest';
import {
  LEARNED_POLICY_META_KEY,
  LEARNED_REPLAY_STORAGE_KEY,
  LEARNED_POLICY_STORAGE_KEY,
  LearningModeController,
  serializeReplayExport,
  loadLearnedPolicy,
  loadLearningReplay,
  saveLearningReplay,
  saveLearnedPolicy
} from '../src/ai/learningMode';
import { createPolicyWeights } from '../src/ai/policyNetwork';
import { FIELD, createInitialState } from '../src/game/model';

class MemoryStorage implements Pick<Storage, 'getItem' | 'setItem' | 'removeItem'> {
  private readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }
}

describe('learning mode state', () => {
  it('starts learning as human red versus neural blue', () => {
    const controller = new LearningModeController(createPolicyWeights(1));

    const session = controller.startLearningMode();

    expect(session.controlModes).toEqual({ red: 'human', blue: 'neural' });
    expect(controller.snapshot.enabled).toBe(true);
    expect(controller.snapshot.samples).toBe(0);
  });

  it('records human samples at ai cadence and updates the model online', () => {
    const controller = new LearningModeController(createPolicyWeights(2), {
      onlineBatchSize: 1,
      learningRate: 0.18
    });
    controller.startLearningMode();
    const state = createInitialState();
    const red = state.tanks[0];
    red.position = { x: FIELD.length - 130, y: FIELD.ballRadius + 34 };
    state.ball.position = { x: FIELD.length - FIELD.ballRadius - 14, y: FIELD.ballRadius + 12 };

    const first = controller.recordAiTick(state, 'red', { leftTrack: 1, rightTrack: 0 });
    const second = controller.recordAiTick(state, 'red', { leftTrack: 1, rightTrack: 0 });

    expect(first.recorded).toBe(true);
    expect(second.recorded).toBe(true);
    expect(controller.snapshot.samples).toBe(2);
    expect(controller.snapshot.modelVersion).toBeGreaterThan(0);
    expect(controller.snapshot.latestLoss).toBeGreaterThan(0);
  });

  it('saves and loads learned weights metadata', () => {
    const storage = new MemoryStorage();
    const weights = createPolicyWeights(3);

    saveLearnedPolicy(storage, {
      weights,
      meta: {
        modelVersion: 5,
        samples: 12,
        latestLoss: 0.42
      }
    });

    expect(storage.getItem(LEARNED_POLICY_STORAGE_KEY)).not.toBeNull();
    expect(storage.getItem(LEARNED_POLICY_META_KEY)).not.toBeNull();
    expect(loadLearnedPolicy(storage)).toEqual({
      weights,
      meta: {
        modelVersion: 5,
        samples: 12,
        latestLoss: 0.42
      }
    });
  });

  it('ignores old learned policy weights with obsolete network shape', () => {
    const storage = new MemoryStorage();
    storage.setItem(LEARNED_POLICY_STORAGE_KEY, JSON.stringify(Array.from({ length: 1113 }, () => 0)));
    storage.setItem(LEARNED_POLICY_META_KEY, JSON.stringify({
      modelVersion: 99,
      samples: 10,
      latestLoss: 0.2
    }));

    expect(loadLearnedPolicy(storage)).toBeNull();
  });

  it('persists replay samples so later replay training can reuse them', () => {
    const storage = new MemoryStorage();
    const controller = new LearningModeController(createPolicyWeights(4), {
      onlineBatchSize: 1,
      learningRate: 0.18
    });
    controller.startLearningMode();
    const state = createInitialState();
    state.ball.position = { x: FIELD.length - FIELD.ballRadius - 14, y: FIELD.ballRadius + 12 };

    controller.recordAiTick(state, 'red', { leftTrack: 1, rightTrack: 1 });
    saveLearningReplay(storage, controller.replaySamples);

    expect(storage.getItem(LEARNED_REPLAY_STORAGE_KEY)).not.toBeNull();

    const replaySamples = loadLearningReplay(storage);
    const replayController = new LearningModeController(createPolicyWeights(4), {
      replaySamples,
      onlineBatchSize: 1,
      learningRate: 0.18
    });
    const result = replayController.trainReplay();

    expect(replaySamples).toHaveLength(1);
    expect(replayController.snapshot.samples).toBe(1);
    expect(result.trainedSamples).toBeGreaterThan(0);
  });

  it('deep replay training reports a full offline training pass', () => {
    const controller = new LearningModeController(createPolicyWeights(5), {
      onlineBatchSize: 1,
      learningRate: 0.18
    });
    controller.startLearningMode();
    const state = createInitialState();

    for (let i = 0; i < 6; i += 1) {
      state.frame = i * 6;
      state.ball.position = {
        x: FIELD.length - FIELD.ballRadius - 30 - i,
        y: FIELD.width / 2 + i
      };
      controller.recordAiTick(state, 'red', { leftTrack: 1, rightTrack: 1 });
    }

    const result = controller.trainReplay({
      epochs: 20,
      batchSize: 3,
      learningRate: 0.04,
      seed: 9
    });

    expect(result.trainedSamples).toBe(120);
    expect(result.epochs).toBe(20);
    expect(controller.snapshot.modelVersion).toBeGreaterThan(6);
  });

  it('clears in-memory replay samples when the learned model is reset', () => {
    const controller = new LearningModeController(createPolicyWeights(6), {
      onlineBatchSize: 1,
      learningRate: 0.18
    });
    controller.startLearningMode();
    controller.recordAiTick(createInitialState(), 'red', { leftTrack: 1, rightTrack: 1 });

    controller.reset(createPolicyWeights(7));

    expect(controller.snapshot.samples).toBe(0);
    expect(controller.replaySamples).toHaveLength(0);
  });

  it('serializes replay exports with metadata and samples', () => {
    const controller = new LearningModeController(createPolicyWeights(8), {
      onlineBatchSize: 1,
      learningRate: 0.18
    });
    controller.startLearningMode();
    controller.recordAiTick(createInitialState(), 'red', { leftTrack: 1, rightTrack: 1 });

    const payload = JSON.parse(serializeReplayExport(controller.snapshot, controller.replaySamples, {
      origin: 'test',
      exportedAt: '2026-04-30T00:00:00.000Z'
    }));

    expect(payload).toMatchObject({
      exportedAt: '2026-04-30T00:00:00.000Z',
      origin: 'test',
      meta: {
        samples: 1
      }
    });
    expect(payload.samples).toHaveLength(1);
  });
});
