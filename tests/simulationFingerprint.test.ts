import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  StateDigest,
  combinedDigest,
  physicsScenarios,
  runFingerprintScenario,
  runFingerprintSuite,
  runtimeScenarios
} from '../scripts/fingerprint-simulation';
import { loadWeightsPayload } from '../scripts/coach-neural';

/**
 * Deterministic trajectories are this project's only promotion evidence, so the
 * physics kernel and the runtime decision stack are pinned to exact digests.
 *
 * The physics digests were captured at commit 540980c, before the simulation hot
 * path was optimised, and have never changed since: the kernel is bit-identical.
 * Any future change that alters a single float of any frame fails here. If a
 * change intends to alter physics, update these digests in the same commit and
 * re-baseline every gate number in project.md, because all historical results
 * become incomparable.
 *
 * The runtime digests are re-baselined whenever the learned value model is
 * promoted, because that changes decisions and therefore trajectories. Each such
 * promotion must move only runtime digests and leave every physics digest
 * identical, which is the signature an AI-only change should have.
 */

const PHYSICS_DIGESTS: Record<string, string> = {
  'physics/idle-drift': '2135920cab0b2d41',
  'physics/head-on-contact': 'dbc5749ba5e9b09d',
  'physics/tank-tank-shove': '394ca6efbe2d19fe',
  'physics/corner-pin': '21dfeddf421dd0c3',
  'physics/goal-and-kickoff': 'dab8ed48a1475511',
  'physics/wall-slide': '745c0ac01205d768',
  'physics/stamina-drain': '0b683a705f21e02f',
  'physics/random-walk-a': '3143bb7504bd01bf',
  'physics/random-walk-b': 'ecdd5c0739599d2b',
  'physics/random-walk-c': 'a07e59859f8752b1'
};

const RUNTIME_DIGESTS: Record<string, string> = {
  'runtime/seed-19': 'fb1c455b6e2ee3e3',
  'runtime/seed-31': 'd80e33c46e79ea4a',
  'runtime/seed-71': 'f9eb02228aa18c32'
};

const COMBINED_DIGEST = '45bc40acd79587f7';

describe('simulation fingerprint', () => {
  it('keeps every scripted physics trajectory bit-exact', () => {
    for (const scenario of physicsScenarios()) {
      const entry = runFingerprintScenario(scenario);
      expect(`${entry.id}=${entry.digest}`).toBe(`${entry.id}=${PHYSICS_DIGESTS[entry.id]}`);
    }
  });

  it('keeps the browser runtime decision stack bit-exact', () => {
    const weights = loadWeightsPayload(readFileSync('public/models/neural-best.json', 'utf8'));
    for (const scenario of runtimeScenarios(weights)) {
      const entry = runFingerprintScenario(scenario);
      expect(`${entry.id}=${entry.digest}`).toBe(`${entry.id}=${RUNTIME_DIGESTS[entry.id]}`);
    }
  });

  it('keeps the combined suite digest stable', () => {
    const weights = loadWeightsPayload(readFileSync('public/models/neural-best.json', 'utf8'));
    const entries = runFingerprintSuite([...physicsScenarios(), ...runtimeScenarios(weights)]);
    expect(combinedDigest(entries)).toBe(COMBINED_DIGEST);
  });

  it('detects a single-ulp divergence', () => {
    const base = new StateDigest();
    base.pushNumber(1.5);
    const drifted = new StateDigest();
    drifted.pushNumber(1.5 + Number.EPSILON);
    expect(drifted.hex()).not.toBe(base.hex());
  });

  it('distinguishes negative zero from positive zero', () => {
    const positive = new StateDigest();
    positive.pushNumber(0);
    const negative = new StateDigest();
    negative.pushNumber(-0);
    expect(negative.hex()).not.toBe(positive.hex());
  });
});
