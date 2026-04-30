import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { evaluatePolicy, policyProbabilities } from '../src/ai/policyNetwork';

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
