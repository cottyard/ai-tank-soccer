import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { evaluatePolicy, policyProbabilities } from '../src/ai/policyNetwork';
import { parsePolicyGradientArgs, runPolicyGradientCli } from '../scripts/train-policy-gradient';

const gppAvailable = spawnSync('g++.exe', ['--version'], { stdio: 'ignore' }).status === 0;
const cargoPath = resolveTool('cargo.exe');

describe('native policy trainer', () => {
  (gppAvailable ? it : it.skip)('trains a small weighted policy dataset and writes compatible weights', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-native-trainer-'));
    const binary = join(workdir, 'native-trainer.exe');
    const inputWeights = join(workdir, 'weights.json');
    const dataset = join(workdir, 'dataset.json');
    const outputWeights = join(workdir, 'trained.json');
    const sampleInputs = Array.from({ length: 36 }, (_, index) =>
      index % 3 === 0 ? 0.7 : index % 3 === 1 ? -0.25 : 0.1
    );
    const weights = defaultNeuralWeights();
    const before = policyProbabilities(evaluatePolicy(sampleInputs, weights))[8];

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    writeFileSync(
      dataset,
      JSON.stringify({
        samples: Array.from({ length: 12 }, (_, frame) => ({
          inputs: sampleInputs,
          actionIndex: 8,
          team: 'red',
          frame,
          tags: ['finish'],
          weight: 2
        }))
      }),
      'utf8'
    );

    execFileSync('g++.exe', [
      '-std=c++20',
      '-O2',
      '-o',
      binary,
      join(root, 'trainer-cpp', 'train_policy.cpp')
    ], { cwd: root, stdio: 'pipe' });
    execFileSync(binary, [
      '--weights',
      inputWeights,
      '--data',
      dataset,
      '--output',
      outputWeights,
      '--epochs',
      '4',
      '--batch-size',
      '4',
      '--learning-rate',
      '0.05'
    ], { cwd: root, stdio: 'pipe' });

    expect(existsSync(outputWeights)).toBe(true);
    const trained = JSON.parse(readFileSync(outputWeights, 'utf8')) as { weights: number[] };
    const after = policyProbabilities(evaluatePolicy(sampleInputs, trained.weights))[8];

    expect(trained.weights).toHaveLength(weights.length);
    expect(after).toBeGreaterThan(before);
  });

  (cargoPath ? it : it.skip)('trains the same dataset with the Rust parity trainer', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-trainer-'));
    const inputWeights = join(workdir, 'weights.json');
    const dataset = join(workdir, 'dataset.json');
    const outputWeights = join(workdir, 'trained.json');
    const targetDir = join(workdir, 'target');
    const sampleInputs = Array.from({ length: 36 }, (_, index) =>
      index % 3 === 0 ? 0.55 : index % 3 === 1 ? -0.35 : 0.15
    );
    const weights = defaultNeuralWeights();
    const before = policyProbabilities(evaluatePolicy(sampleInputs, weights))[8];
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    writeFileSync(
      dataset,
      JSON.stringify({
        samples: Array.from({ length: 16 }, (_, frame) => ({
          inputs: sampleInputs,
          actionIndex: 8,
          team: 'red',
          frame,
          tags: ['finish', 'contest'],
          weight: 2.2
        }))
      }),
      'utf8'
    );

    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    execFileSync(join(targetDir, 'release', 'soccer-policy-trainer.exe'), [
      '--weights',
      inputWeights,
      '--data',
      dataset,
      '--output',
      outputWeights,
      '--epochs',
      '5',
      '--batch-size',
      '4',
      '--learning-rate',
      '0.04'
    ], { cwd: root, env, stdio: 'pipe' });

    expect(existsSync(outputWeights)).toBe(true);
    const trained = JSON.parse(readFileSync(outputWeights, 'utf8')) as { weights: number[] };
    const after = policyProbabilities(evaluatePolicy(sampleInputs, trained.weights))[8];

    expect(trained.weights).toHaveLength(weights.length);
    expect(after).toBeGreaterThan(before);
  });

  (cargoPath ? it : it.skip)('runs sparse-reward Rust PPO self-play deterministically through the CLI wrapper', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-trainer-'));
    const inputWeights = join(workdir, 'weights.json');
    const firstOutput = join(workdir, 'trained-a.json');
    const secondOutput = join(workdir, 'trained-b.json');
    const firstMetrics = join(workdir, 'metrics-a.json');
    const secondMetrics = join(workdir, 'metrics-b.json');
    const targetDir = join(workdir, 'target');
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };
    const weights = defaultNeuralWeights();

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    const nativeBin = join(targetDir, 'release', 'soccer-policy-trainer.exe');

    const baseOptions = {
      ...parsePolicyGradientArgs([
        '--native',
        '--native-bin',
        nativeBin,
        '--input',
        inputWeights,
        '--seed',
        '77',
        '--matches',
        '4',
        '--frames',
        '24',
        '--epochs',
        '2',
        '--batch-size',
        '8',
        '--learning-rate',
        '0.006',
        '--ppo-clip',
        '0.2',
        '--temperature',
        '1.08',
        '--discount',
        '0.992',
        '--goal-reward',
        '1.7',
        '--win-reward',
        '2.1',
        '--start-state-mode',
        'mixed',
        '--open-start-ratio',
        '0.25',
        '--advantage-baseline',
        'learned',
        '--action-mode',
        'runtime',
        '--opponent-mode',
        'traditional'
      ])
    };

    const first = runPolicyGradientCli({
      ...baseOptions,
      output: firstOutput,
      metricsOutput: firstMetrics
    });
    const second = runPolicyGradientCli({
      ...baseOptions,
      output: secondOutput,
      metricsOutput: secondMetrics
    });

    expect(existsSync(firstOutput)).toBe(true);
    expect(existsSync(firstMetrics)).toBe(true);
    expect(first).toEqual(second);
    expect(readFileSync(firstOutput, 'utf8')).toBe(readFileSync(secondOutput, 'utf8'));
    expect(readFileSync(firstMetrics, 'utf8')).toBe(readFileSync(secondMetrics, 'utf8'));
    expect(first.samples).toBeGreaterThan(0);
    expect(first.trainedSamples).toBeGreaterThan(0);
    expect(first.frames).toBe(96);
    expect(first.redGoals + first.blueGoals).toBeGreaterThan(0);
    expect(first.weights).toHaveLength(weights.length);
    expect(totalDelta(first.weights, weights)).toBeGreaterThan(0);

    const nativeOutput = JSON.parse(readFileSync(firstOutput, 'utf8')) as {
      metadata?: {
        trainer?: string;
        startStateMode?: string;
        openStartRatio?: number;
        advantageBaseline?: string;
        actionMode?: string;
        runtimeWrapperWeightMode?: string;
        goalReward?: number;
        winReward?: number;
        opponentMode?: string;
      };
    };
    expect(nativeOutput.metadata?.trainer).toBe('rust-policy-gradient');
    expect(nativeOutput.metadata?.startStateMode).toBe('mixed');
    expect(nativeOutput.metadata?.openStartRatio).toBe(0.25);
    expect(nativeOutput.metadata?.advantageBaseline).toBe('learned');
    expect(nativeOutput.metadata?.actionMode).toBe('runtime');
    expect(nativeOutput.metadata?.runtimeWrapperWeightMode).toBe('none');
    expect(nativeOutput.metadata?.goalReward).toBe(1.7);
    expect(nativeOutput.metadata?.winReward).toBe(2.1);
    expect(nativeOutput.metadata?.opponentMode).toBe('traditional');

    const nativeMetrics = JSON.parse(readFileSync(firstMetrics, 'utf8')) as {
      advantageBaseline?: string;
      openStartRatio?: number;
      actionMode?: string;
      runtimeWrapperWeightMode?: string;
      goalReward?: number;
      winReward?: number;
      opponentMode?: string;
      startFamilies?: Record<string, number>;
      policyActionSurvival?: {
        sampled: number;
        survived: number;
        changed: number;
        tacticalChanged: number;
        staminaConserved: number;
        criticalRegulated: number;
        survivalRate: number;
      };
      runtimeDecisionOutcomes?: Record<string, {
        count: number;
        meanReturn: number;
        meanAdvantage: number;
        meanAbsAdvantage: number;
        positiveReturns: number;
        negativeReturns: number;
        positiveAdvantages: number;
        negativeAdvantages: number;
      }>;
    };
    expect(nativeMetrics.advantageBaseline).toBe('learned');
    expect(nativeMetrics.openStartRatio).toBe(0.25);
    expect(nativeMetrics.actionMode).toBe('runtime');
    expect(nativeMetrics.runtimeWrapperWeightMode).toBe('none');
    expect(nativeMetrics.goalReward).toBe(1.7);
    expect(nativeMetrics.winReward).toBe(2.1);
    expect(nativeMetrics.opponentMode).toBe('traditional');
    expect(nativeMetrics.startFamilies).toMatchObject({
      open: 2,
      outcomeCurriculum: 1,
      ownGoalDefense: 0,
      cornerFight: 1,
      looseBallContest: 0
    });
    expect(nativeMetrics.policyActionSurvival).toBeDefined();
    const survival = nativeMetrics.policyActionSurvival!;
    expect(survival).toMatchObject({
      sampled: survival.survived + survival.changed,
      survivalRate: expect.any(Number)
    });
    expect(survival.sampled).toBeGreaterThan(0);
    expect(survival.survivalRate).toBeGreaterThanOrEqual(0);
    expect(survival.survivalRate).toBeLessThanOrEqual(1);
    expect(nativeMetrics.runtimeDecisionOutcomes).toBeDefined();
    expect(nativeMetrics.runtimeDecisionOutcomes?.survived.count).toBe(survival.survived);
    expect(nativeMetrics.runtimeDecisionOutcomes?.changed.count).toBe(survival.changed);
    expect(nativeMetrics.runtimeDecisionOutcomes?.tacticalChanged.count).toBe(survival.tacticalChanged);
    expect(nativeMetrics.runtimeDecisionOutcomes?.staminaConserved.count).toBe(survival.staminaConserved);
    expect(nativeMetrics.runtimeDecisionOutcomes?.criticalRegulated.count).toBe(survival.criticalRegulated);
    expect(nativeMetrics.runtimeDecisionOutcomes?.survived.meanReturn).toEqual(expect.any(Number));
    expect(nativeMetrics.runtimeDecisionOutcomes?.survived.meanAdvantage).toEqual(expect.any(Number));
    expect(nativeMetrics.runtimeDecisionOutcomes?.survived.meanAbsAdvantage).toBeGreaterThanOrEqual(0);
  });

  (cargoPath ? it : it.skip)('supports frozen opponent weights for Rust PPO self-play', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-opponent-'));
    const inputWeights = join(workdir, 'weights.json');
    const opponentWeightsPath = join(workdir, 'opponent.json');
    const outputWeights = join(workdir, 'trained.json');
    const metricsOutput = join(workdir, 'metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const opponentWeights = weights.map((weight, index) => weight + (index % 2 === 0 ? 0.002 : -0.002));
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    writeFileSync(opponentWeightsPath, JSON.stringify({ weights: opponentWeights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    execFileSync(join(targetDir, 'release', 'soccer-policy-trainer.exe'), [
      '--mode',
      'policy-gradient',
      '--weights',
      inputWeights,
      '--opponent-weights',
      opponentWeightsPath,
      '--output',
      outputWeights,
      '--metrics-output',
      metricsOutput,
      '--seed',
      '91',
      '--matches',
      '2',
      '--frames',
      '24',
      '--epochs',
      '1',
      '--batch-size',
      '4'
    ], { cwd: root, env, stdio: 'pipe' });

    const metrics = JSON.parse(readFileSync(metricsOutput, 'utf8')) as {
      samples: number;
      decisions: number;
    };

    expect(metrics.decisions).toBeGreaterThan(metrics.samples);
    expect(metrics.samples).toBe(metrics.decisions / 2);
  });

  (cargoPath ? it : it.skip)('can train only runtime samples whose policy action survives the wrapper', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-survivors-'));
    const inputWeights = join(workdir, 'weights.json');
    const outputWeights = join(workdir, 'trained.json');
    const metricsOutput = join(workdir, 'metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    runPolicyGradientCli(parsePolicyGradientArgs([
      '--native',
      '--native-bin',
      join(targetDir, 'release', 'soccer-policy-trainer.exe'),
      '--input',
      inputWeights,
      '--output',
      outputWeights,
      '--metrics-output',
      metricsOutput,
      '--seed',
      '93',
      '--matches',
      '8',
      '--frames',
      '60',
      '--epochs',
      '1',
      '--batch-size',
      '8',
      '--action-mode',
      'runtime',
      '--runtime-survivors-only'
    ]));

    const metrics = JSON.parse(readFileSync(metricsOutput, 'utf8')) as {
      runtimeSurvivorsOnly?: boolean;
      samples: number;
      policyActionSurvival?: {
        sampled: number;
        survived: number;
        changed: number;
      };
    };

    expect(metrics.runtimeSurvivorsOnly).toBe(true);
    expect(metrics.policyActionSurvival?.sampled).toBeGreaterThan(0);
    expect(metrics.policyActionSurvival?.changed).toBeGreaterThan(0);
    expect(metrics.samples).toBe(metrics.policyActionSurvival?.survived);
    expect(metrics.samples).toBeLessThan(metrics.policyActionSurvival?.sampled ?? 0);
  });

  (cargoPath ? it : it.skip)('can down-weight tactical wrapper rewrites without dropping samples', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-wrapper-weight-'));
    const inputWeights = join(workdir, 'weights.json');
    const defaultOutputWeights = join(workdir, 'default-trained.json');
    const defaultMetricsOutput = join(workdir, 'default-metrics.json');
    const weightedOutputWeights = join(workdir, 'weighted-trained.json');
    const weightedMetricsOutput = join(workdir, 'weighted-metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    const nativeBin = join(targetDir, 'release', 'soccer-policy-trainer.exe');
    const baseArgs = [
      '--native',
      '--native-bin',
      nativeBin,
      '--input',
      inputWeights,
      '--seed',
      '94',
      '--matches',
      '8',
      '--frames',
      '60',
      '--epochs',
      '1',
      '--batch-size',
      '8',
      '--action-mode',
      'runtime'
    ];

    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      defaultOutputWeights,
      '--metrics-output',
      defaultMetricsOutput
    ]));
    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      weightedOutputWeights,
      '--metrics-output',
      weightedMetricsOutput,
      '--runtime-wrapper-weight-mode',
      'tactical-downweight',
      '--runtime-tactical-rewrite-weight',
      '0.2'
    ]));

    const defaultMetrics = JSON.parse(readFileSync(defaultMetricsOutput, 'utf8')) as {
      runtimeWrapperWeightMode?: string;
      runtimeTacticalRewriteWeight?: number;
      samples: number;
      policyActionSurvival?: { tacticalChanged: number };
    };
    const weightedMetrics = JSON.parse(readFileSync(weightedMetricsOutput, 'utf8')) as {
      runtimeWrapperWeightMode?: string;
      runtimeTacticalRewriteWeight?: number;
      samples: number;
      policyActionSurvival?: { tacticalChanged: number };
    };
    const weightedOutput = JSON.parse(readFileSync(weightedOutputWeights, 'utf8')) as {
      metadata?: { runtimeWrapperWeightMode?: string; runtimeTacticalRewriteWeight?: number };
      weights: number[];
    };

    expect(defaultMetrics.runtimeWrapperWeightMode).toBe('none');
    expect(weightedMetrics.runtimeWrapperWeightMode).toBe('tactical-downweight');
    expect(weightedMetrics.runtimeTacticalRewriteWeight).toBe(0.2);
    expect(weightedOutput.metadata?.runtimeWrapperWeightMode).toBe('tactical-downweight');
    expect(weightedOutput.metadata?.runtimeTacticalRewriteWeight).toBe(0.2);
    expect(weightedMetrics.policyActionSurvival?.tacticalChanged).toBeGreaterThan(0);
    expect(weightedMetrics.samples).toBe(defaultMetrics.samples);
    const defaultOutput = JSON.parse(readFileSync(defaultOutputWeights, 'utf8')) as { weights: number[] };
    expect(totalDelta(weightedOutput.weights, defaultOutput.weights)).toBeGreaterThan(0);
  });

  (cargoPath ? it : it.skip)('can apply an action-retention penalty during Rust PPO updates', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-retention-'));
    const inputWeights = join(workdir, 'weights.json');
    const defaultOutputWeights = join(workdir, 'default-trained.json');
    const defaultMetricsOutput = join(workdir, 'default-metrics.json');
    const retainedOutputWeights = join(workdir, 'retained-trained.json');
    const retainedMetricsOutput = join(workdir, 'retained-metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    const nativeBin = join(targetDir, 'release', 'soccer-policy-trainer.exe');
    const baseArgs = [
      '--native',
      '--native-bin',
      nativeBin,
      '--input',
      inputWeights,
      '--seed',
      '96',
      '--matches',
      '8',
      '--frames',
      '60',
      '--epochs',
      '1',
      '--batch-size',
      '8',
      '--action-mode',
      'runtime'
    ];

    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      defaultOutputWeights,
      '--metrics-output',
      defaultMetricsOutput
    ]));
    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      retainedOutputWeights,
      '--metrics-output',
      retainedMetricsOutput,
      '--action-retention-weight',
      '0.4'
    ]));

    const retainedMetrics = JSON.parse(readFileSync(retainedMetricsOutput, 'utf8')) as {
      actionRetentionWeight?: number;
      samples: number;
    };
    const retainedOutput = JSON.parse(readFileSync(retainedOutputWeights, 'utf8')) as {
      metadata?: { actionRetentionWeight?: number };
      weights: number[];
    };
    const defaultOutput = JSON.parse(readFileSync(defaultOutputWeights, 'utf8')) as { weights: number[] };

    expect(retainedMetrics.actionRetentionWeight).toBe(0.4);
    expect(retainedMetrics.samples).toBeGreaterThan(0);
    expect(retainedOutput.metadata?.actionRetentionWeight).toBe(0.4);
    expect(totalDelta(retainedOutput.weights, defaultOutput.weights)).toBeGreaterThan(0);
  });

  (cargoPath ? it : it.skip)('can down-weight early low-pressure forward actions during Rust PPO updates', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-forward-safety-'));
    const inputWeights = join(workdir, 'weights.json');
    const defaultOutputWeights = join(workdir, 'default-trained.json');
    const defaultMetricsOutput = join(workdir, 'default-metrics.json');
    const weightedOutputWeights = join(workdir, 'weighted-trained.json');
    const weightedMetricsOutput = join(workdir, 'weighted-metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    const nativeBin = join(targetDir, 'release', 'soccer-policy-trainer.exe');
    const baseArgs = [
      '--native',
      '--native-bin',
      nativeBin,
      '--input',
      inputWeights,
      '--seed',
      '97',
      '--matches',
      '8',
      '--frames',
      '60',
      '--epochs',
      '1',
      '--batch-size',
      '8',
      '--start-state-mode',
      'open',
      '--action-mode',
      'runtime'
    ];

    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      defaultOutputWeights,
      '--metrics-output',
      defaultMetricsOutput
    ]));
    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      weightedOutputWeights,
      '--metrics-output',
      weightedMetricsOutput,
      '--early-forward-safety-weight',
      '0.2'
    ]));

    const defaultMetrics = JSON.parse(readFileSync(defaultMetricsOutput, 'utf8')) as {
      earlyForwardSafetyWeight?: number;
      earlyForwardSafety?: { candidates: number; weighted: number; weight: number };
    };
    const weightedMetrics = JSON.parse(readFileSync(weightedMetricsOutput, 'utf8')) as {
      earlyForwardSafetyWeight?: number;
      earlyForwardSafety?: { candidates: number; weighted: number; weight: number };
    };
    const weightedOutput = JSON.parse(readFileSync(weightedOutputWeights, 'utf8')) as {
      metadata?: { earlyForwardSafetyWeight?: number };
      weights: number[];
    };
    const defaultOutput = JSON.parse(readFileSync(defaultOutputWeights, 'utf8')) as { weights: number[] };

    expect(defaultMetrics.earlyForwardSafetyWeight).toBe(1);
    expect(defaultMetrics.earlyForwardSafety?.weight).toBe(1);
    expect(weightedMetrics.earlyForwardSafetyWeight).toBe(0.2);
    expect(weightedMetrics.earlyForwardSafety?.weight).toBe(0.2);
    expect(weightedMetrics.earlyForwardSafety?.candidates).toBeGreaterThan(0);
    expect(weightedMetrics.earlyForwardSafety?.weighted).toBe(weightedMetrics.earlyForwardSafety?.candidates);
    expect(weightedOutput.metadata?.earlyForwardSafetyWeight).toBe(0.2);
    expect(totalDelta(weightedOutput.weights, defaultOutput.weights)).toBeGreaterThan(0);
  });

  (cargoPath ? it : it.skip)('can anchor early low-pressure forward actions during Rust PPO updates', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-forward-anchor-'));
    const inputWeights = join(workdir, 'weights.json');
    const defaultOutputWeights = join(workdir, 'default-trained.json');
    const defaultMetricsOutput = join(workdir, 'default-metrics.json');
    const anchoredOutputWeights = join(workdir, 'anchored-trained.json');
    const anchoredMetricsOutput = join(workdir, 'anchored-metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    const nativeBin = join(targetDir, 'release', 'soccer-policy-trainer.exe');
    const baseArgs = [
      '--native',
      '--native-bin',
      nativeBin,
      '--input',
      inputWeights,
      '--seed',
      '98',
      '--matches',
      '8',
      '--frames',
      '60',
      '--epochs',
      '1',
      '--batch-size',
      '8',
      '--start-state-mode',
      'open',
      '--action-mode',
      'runtime'
    ];

    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      defaultOutputWeights,
      '--metrics-output',
      defaultMetricsOutput
    ]));
    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      anchoredOutputWeights,
      '--metrics-output',
      anchoredMetricsOutput,
      '--early-forward-anchor-weight',
      '0.4'
    ]));

    const defaultMetrics = JSON.parse(readFileSync(defaultMetricsOutput, 'utf8')) as {
      earlyForwardAnchorWeight?: number;
      earlyForwardSafety?: { candidates: number; anchored: number; anchorWeight: number };
    };
    const anchoredMetrics = JSON.parse(readFileSync(anchoredMetricsOutput, 'utf8')) as {
      earlyForwardAnchorWeight?: number;
      earlyForwardSafety?: { candidates: number; anchored: number; anchorWeight: number };
    };
    const anchoredOutput = JSON.parse(readFileSync(anchoredOutputWeights, 'utf8')) as {
      metadata?: { earlyForwardAnchorWeight?: number };
      weights: number[];
    };
    const defaultOutput = JSON.parse(readFileSync(defaultOutputWeights, 'utf8')) as { weights: number[] };

    expect(defaultMetrics.earlyForwardAnchorWeight).toBe(0);
    expect(defaultMetrics.earlyForwardSafety?.anchorWeight).toBe(0);
    expect(anchoredMetrics.earlyForwardAnchorWeight).toBe(0.4);
    expect(anchoredMetrics.earlyForwardSafety?.anchorWeight).toBe(0.4);
    expect(anchoredMetrics.earlyForwardSafety?.candidates).toBeGreaterThan(0);
    expect(anchoredMetrics.earlyForwardSafety?.anchored).toBe(anchoredMetrics.earlyForwardSafety?.candidates);
    expect(anchoredOutput.metadata?.earlyForwardAnchorWeight).toBe(0.4);
    expect(totalDelta(anchoredOutput.weights, defaultOutput.weights)).toBeGreaterThan(0);
  });

  (cargoPath ? it : it.skip)('can append explicit policy-anchor samples during Rust PPO updates', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-state-anchor-'));
    const inputWeights = join(workdir, 'weights.json');
    const anchorData = join(workdir, 'anchors.json');
    const defaultOutputWeights = join(workdir, 'default-trained.json');
    const defaultMetricsOutput = join(workdir, 'default-metrics.json');
    const anchoredOutputWeights = join(workdir, 'anchored-trained.json');
    const anchoredMetricsOutput = join(workdir, 'anchored-metrics.json');
    const targetDir = join(workdir, 'target');
    const sampleInputs = Array.from({ length: 36 }, (_, index) =>
      index % 3 === 0 ? 0.4 : index % 3 === 1 ? -0.2 : 0.05
    );
    const weights = defaultNeuralWeights();
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    writeFileSync(anchorData, JSON.stringify({
      samples: [
        { inputs: sampleInputs, actionIndex: 8, team: 'red', frame: 18, tags: ['policyAnchor'], weight: 2 },
        { inputs: sampleInputs.map((value) => value * 0.5), actionIndex: 8, team: 'blue', frame: 30, tags: ['policyAnchor'], weight: 1 }
      ]
    }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    const nativeBin = join(targetDir, 'release', 'soccer-policy-trainer.exe');
    const baseArgs = [
      '--native',
      '--native-bin',
      nativeBin,
      '--input',
      inputWeights,
      '--seed',
      '99',
      '--matches',
      '4',
      '--frames',
      '48',
      '--epochs',
      '1',
      '--batch-size',
      '8',
      '--action-mode',
      'runtime'
    ];

    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      defaultOutputWeights,
      '--metrics-output',
      defaultMetricsOutput
    ]));
    runPolicyGradientCli(parsePolicyGradientArgs([
      ...baseArgs,
      '--output',
      anchoredOutputWeights,
      '--metrics-output',
      anchoredMetricsOutput,
      '--policy-anchor-data',
      anchorData,
      '--policy-anchor-weight',
      '0.5'
    ]));

    const anchoredMetrics = JSON.parse(readFileSync(anchoredMetricsOutput, 'utf8')) as {
      policyAnchorSamples?: number;
      policyAnchorWeight?: number;
      samples: number;
    };
    const anchoredOutput = JSON.parse(readFileSync(anchoredOutputWeights, 'utf8')) as {
      metadata?: { policyAnchorSamples?: number; policyAnchorWeight?: number };
      weights: number[];
    };
    const defaultOutput = JSON.parse(readFileSync(defaultOutputWeights, 'utf8')) as { weights: number[] };

    expect(anchoredMetrics.policyAnchorSamples).toBe(2);
    expect(anchoredMetrics.policyAnchorWeight).toBe(0.5);
    const policyAnchorSamples = anchoredMetrics.policyAnchorSamples;
    if (policyAnchorSamples === undefined) {
      throw new Error('Expected policy anchor sample metrics');
    }
    expect(anchoredMetrics.samples).toBeGreaterThan(policyAnchorSamples);
    expect(anchoredOutput.metadata?.policyAnchorSamples).toBe(2);
    expect(anchoredOutput.metadata?.policyAnchorWeight).toBe(0.5);
    expect(totalDelta(anchoredOutput.weights, defaultOutput.weights)).toBeGreaterThan(0);
  });

  (cargoPath ? it : it.skip)('samples Rust PPO opponents from a weighted league', () => {
    const root = process.cwd();
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-rust-ppo-league-'));
    const inputWeights = join(workdir, 'weights.json');
    const recentWeightsPath = join(workdir, 'recent.json');
    const historicalWeightsPath = join(workdir, 'historical.json');
    const outputWeights = join(workdir, 'trained.json');
    const metricsOutput = join(workdir, 'metrics.json');
    const targetDir = join(workdir, 'target');
    const weights = defaultNeuralWeights();
    const recentWeights = weights.map((weight, index) => weight + (index % 3 === 0 ? 0.003 : -0.001));
    const historicalWeights = weights.map((weight, index) => weight + (index % 5 === 0 ? -0.004 : 0.002));
    const env = {
      ...process.env,
      PATH: `${parentDirectory(cargoPath!)};${process.env.PATH ?? ''}`,
      HTTP_PROXY: process.env.HTTP_PROXY ?? 'http://localhost:10808',
      HTTPS_PROXY: process.env.HTTPS_PROXY ?? 'http://localhost:10808'
    };

    writeFileSync(inputWeights, JSON.stringify({ weights }), 'utf8');
    writeFileSync(recentWeightsPath, JSON.stringify({ weights: recentWeights }), 'utf8');
    writeFileSync(historicalWeightsPath, JSON.stringify({ weights: historicalWeights }), 'utf8');
    execFileSync(cargoPath!, [
      'build',
      '--release',
      '--manifest-path',
      join(root, 'trainer-rust', 'Cargo.toml'),
      '--target-dir',
      targetDir
    ], { cwd: root, env, stdio: 'pipe' });
    runPolicyGradientCli(parsePolicyGradientArgs([
      '--native',
      '--native-bin',
      join(targetDir, 'release', 'soccer-policy-trainer.exe'),
      '--input',
      inputWeights,
      '--league-opponent-weights',
      recentWeightsPath,
      '--league-opponent-weights',
      historicalWeightsPath,
      '--league-current-weight',
      '1.5',
      '--league-traditional-weight',
      '0.25',
      '--opponent-mode',
      'league',
      '--output',
      outputWeights,
      '--metrics-output',
      metricsOutput,
      '--seed',
      '93',
      '--matches',
      '8',
      '--frames',
      '24',
      '--epochs',
      '1',
      '--batch-size',
      '4'
    ]));

    const trained = JSON.parse(readFileSync(outputWeights, 'utf8')) as {
      metadata?: {
        opponentMode?: string;
        leagueOpponentCount?: number;
        leagueCurrentWeight?: number;
        leagueTraditionalWeight?: number;
      };
    };
    const metrics = JSON.parse(readFileSync(metricsOutput, 'utf8')) as {
      samples: number;
      decisions: number;
      opponentMode?: string;
      leagueOpponentCount?: number;
      leagueCurrentWeight?: number;
      leagueTraditionalWeight?: number;
    };

    expect(metrics.opponentMode).toBe('league');
    expect(metrics.leagueOpponentCount).toBe(2);
    expect(metrics.leagueCurrentWeight).toBe(1.5);
    expect(metrics.leagueTraditionalWeight).toBe(0.25);
    expect(metrics.decisions).toBeGreaterThan(metrics.samples);
    expect(metrics.samples).toBeGreaterThan(0);
    expect(metrics.samples).toBeLessThan(metrics.decisions);
    expect(trained.metadata?.opponentMode).toBe('league');
    expect(trained.metadata?.leagueOpponentCount).toBe(2);
    expect(trained.metadata?.leagueCurrentWeight).toBe(1.5);
    expect(trained.metadata?.leagueTraditionalWeight).toBe(0.25);
  });
});

function resolveTool(name: string): string | undefined {
  const userProfile = process.env.USERPROFILE;
  const bundled = userProfile ? join(userProfile, '.cargo', 'bin', name) : undefined;
  if (bundled && existsSync(bundled)) {
    return bundled;
  }
  return spawnSync(name, ['--version'], { stdio: 'ignore' }).status === 0 ? name : undefined;
}

function parentDirectory(path: string): string {
  const normalized = path.replace(/\//g, '\\');
  const index = normalized.lastIndexOf('\\');
  return index === -1 ? '.' : normalized.slice(0, index);
}

function totalDelta(a: readonly number[], b: readonly number[]): number {
  return a.reduce((sum, value, index) => sum + Math.abs(value - b[index]), 0);
}
