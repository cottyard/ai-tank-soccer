import { describe, expect, it } from 'vitest';
import { commandToActionIndex } from '../src/ai/policyActions';
import { POLICY_INPUT_COUNT, createPolicyWeights, evaluatePolicy, policyProbabilities } from '../src/ai/policyNetwork';
import {
  LearningReplayBuffer,
  createLearningSample,
  tagLearningSample,
  trainOfflineFromReplay,
  trainOnlineFromReplay
} from '../src/ai/imitationLearning';
import { FIELD, createInitialState } from '../src/game/model';

describe('imitation learning', () => {
  it('tags and weights corner contact samples higher than ordinary samples', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.position = { x: FIELD.length - 120, y: FIELD.ballRadius + 30 };
    state.ball.position = { x: FIELD.length - FIELD.ballRadius - 12, y: FIELD.ballRadius + 10 };
    state.ball.velocity = { x: 0, y: 0 };

    const tags = tagLearningSample(state, 'red', tank);
    const sample = createLearningSample(state, 'red', tank, { leftTrack: 1, rightTrack: 0 });

    expect(tags).toContain('corner');
    expect(tags).toContain('contact');
    expect(sample.actionIndex).toBe(commandToActionIndex({ leftTrack: 1, rightTrack: 0 }));
    expect(sample.weight).toBeGreaterThan(1);
  });

  it('stores stamina ratio in the replay input vector', () => {
    const state = createInitialState();
    const tank = state.tanks[0];
    tank.stamina = tank.maxStamina * 0.37;

    const sample = createLearningSample(state, 'red', tank, { leftTrack: 0, rightTrack: 0 });

    expect(sample.inputs).toHaveLength(POLICY_INPUT_COUNT);
    expect(sample.inputs[7]).toBeCloseTo(0.37, 6);
    expect(sample.tags).toContain('lowStamina');
  });

  it('tags contested loose balls so training can prioritize possession fights', () => {
    const state = createInitialState();
    const red = state.tanks.find((tank) => tank.team === 'red' && tank.index === 0)!;
    const blue = state.tanks.find((tank) => tank.team === 'blue' && tank.index === 0)!;
    state.ball.position = { x: FIELD.length / 2, y: FIELD.width / 2 + 22 };
    state.ball.velocity = { x: 8, y: -6 };
    red.position = { x: state.ball.position.x - 74, y: state.ball.position.y - 16 };
    red.angle = 0;
    blue.position = { x: state.ball.position.x + 54, y: state.ball.position.y + 10 };

    const sample = createLearningSample(state, 'red', red, { leftTrack: 1, rightTrack: 1 });

    expect(sample.tags).toContain('contest');
    expect(sample.weight).toBeGreaterThan(2);
  });

  it('down-samples long repeated idle input while preserving active commands', () => {
    const replay = new LearningReplayBuffer({ maxSamples: 100, idleKeepEvery: 3 });
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, () => 0);

    for (let i = 0; i < 8; i += 1) {
      replay.add({
        inputs,
        actionIndex: 4,
        team: 'red',
        frame: i * 6,
        tags: [],
        weight: 1
      });
    }

    replay.add({
      inputs,
      actionIndex: 8,
      team: 'red',
      frame: 60,
      tags: [],
      weight: 1
    });

    expect(replay.samples).toHaveLength(4);
    expect(replay.samples.at(-1)?.actionIndex).toBe(8);
  });

  it('online training updates weights and reports metadata', () => {
    const replay = new LearningReplayBuffer({ maxSamples: 100 });
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, (_, index) =>
      index % 2 === 0 ? 0.4 : -0.25
    );
    replay.add({
      inputs,
      actionIndex: 8,
      team: 'red',
      frame: 0,
      tags: ['contact'],
      weight: 2
    });
    const weights = createPolicyWeights(13);
    const before = policyProbabilities(evaluatePolicy(inputs, weights))[8];

    const result = trainOnlineFromReplay(weights, replay, {
      batchSize: 1,
      learningRate: 0.2,
      seed: 3
    });
    const after = policyProbabilities(evaluatePolicy(inputs, result.weights))[8];

    expect(result.trainedSamples).toBe(1);
    expect(result.loss).toBeGreaterThan(0);
    expect(result.modelVersion).toBe(1);
    expect(after).toBeGreaterThan(before);
  });

  it('offline replay training can sweep the full replay for many epochs', () => {
    const replay = new LearningReplayBuffer({ maxSamples: 100 });
    const targetAction = 8;
    const inputs = Array.from({ length: POLICY_INPUT_COUNT }, (_, index) =>
      index % 4 === 0 ? 0.65 : index % 4 === 1 ? -0.2 : index % 4 === 2 ? 0.35 : 0.08
    );

    for (let i = 0; i < 10; i += 1) {
      replay.add({
        inputs,
        actionIndex: targetAction,
        team: 'red',
        frame: i * 6,
        tags: i % 2 === 0 ? ['finish'] : ['contact'],
        weight: 1.4
      });
    }

    const weights = createPolicyWeights(31);
    const before = policyProbabilities(evaluatePolicy(inputs, weights))[targetAction];
    const result = trainOfflineFromReplay(weights, replay, {
      epochs: 12,
      batchSize: 5,
      learningRate: 0.05,
      seed: 17
    });
    const after = policyProbabilities(evaluatePolicy(inputs, result.weights))[targetAction];

    expect(result.trainedSamples).toBe(120);
    expect(result.epochs).toBe(12);
    expect(result.batches).toBe(24);
    expect(result.modelVersion).toBe(1);
    expect(after).toBeGreaterThan(before);
  });
});
