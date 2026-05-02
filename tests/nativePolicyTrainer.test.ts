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
        '--start-state-mode',
        'mixed',
        '--advantage-baseline',
        'start-team-time',
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
        advantageBaseline?: string;
        actionMode?: string;
        opponentMode?: string;
      };
    };
    expect(nativeOutput.metadata?.trainer).toBe('rust-policy-gradient');
    expect(nativeOutput.metadata?.startStateMode).toBe('mixed');
    expect(nativeOutput.metadata?.advantageBaseline).toBe('start-team-time');
    expect(nativeOutput.metadata?.actionMode).toBe('runtime');
    expect(nativeOutput.metadata?.opponentMode).toBe('traditional');

    const nativeMetrics = JSON.parse(readFileSync(firstMetrics, 'utf8')) as {
      advantageBaseline?: string;
      actionMode?: string;
      opponentMode?: string;
    };
    expect(nativeMetrics.advantageBaseline).toBe('start-team-time');
    expect(nativeMetrics.actionMode).toBe('runtime');
    expect(nativeMetrics.opponentMode).toBe('traditional');
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
