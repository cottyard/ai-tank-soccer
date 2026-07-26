import { readFileSync } from 'node:fs';
import { FIXED_DT } from '../src/game/match';
import {
  FIELD,
  createInitialState,
  type GameState,
  type Team
} from '../src/game/model';
import { stepGame } from '../src/game/simulation';
import type { CommandMap, Strategy, TankCommand } from '../src/game/strategy';
import { actionIndexToCommand } from '../src/ai/policyActions';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import type { NeuralWeights } from '../src/ai/neuralWeights';
import { loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
  exitCode?: number;
};

/**
 * Deterministic physics/runtime fingerprints.
 *
 * The whole project treats deterministic browser-runtime trajectories as its only
 * promotion evidence, so any simulation refactor must be provably behaviour
 * preserving. These fingerprints fold every float of every frame into a digest,
 * which turns "did the physics change at all" into an exact yes/no answer instead
 * of an aggregate-score guess.
 */

export type FingerprintScenario = {
  id: string;
  frames: number;
  build: () => GameState;
  commands: (frame: number, state: Readonly<GameState>) => CommandMap;
};

export type FingerprintEntry = {
  id: string;
  frames: number;
  digest: string;
};

const scratchBuffer = new ArrayBuffer(8);
const scratchFloat = new Float64Array(scratchBuffer);
const scratchBytes = new Uint8Array(scratchBuffer);

/**
 * Two independent 32-bit lanes give a 64-bit digest without BigInt overhead.
 * Hashing raw IEEE-754 bytes makes the comparison exact: `-0` and `0`, or a
 * one-ulp drift, both change the digest.
 */
export class StateDigest {
  private laneA = 0x811c9dc5;
  private laneB = 0x9e3779b9;

  pushNumber(value: number): void {
    scratchFloat[0] = value;
    for (let index = 0; index < 8; index += 1) {
      const byte = scratchBytes[index];
      this.laneA = Math.imul(this.laneA ^ byte, 16777619) >>> 0;
      this.laneB = (Math.imul(this.laneB + byte + 1, 2246822519) ^ (this.laneB >>> 13)) >>> 0;
    }
  }

  pushState(state: Readonly<GameState>): void {
    this.pushNumber(state.frame);
    this.pushNumber(state.time);
    this.pushNumber(state.score.red);
    this.pushNumber(state.score.blue);
    this.pushNumber(state.ball.position.x);
    this.pushNumber(state.ball.position.y);
    this.pushNumber(state.ball.velocity.x);
    this.pushNumber(state.ball.velocity.y);
    for (const tank of state.tanks) {
      this.pushNumber(tank.position.x);
      this.pushNumber(tank.position.y);
      this.pushNumber(tank.velocity.x);
      this.pushNumber(tank.velocity.y);
      this.pushNumber(tank.angle);
      this.pushNumber(tank.angularVelocity);
      this.pushNumber(tank.stamina);
    }
  }

  hex(): string {
    return (this.laneA >>> 0).toString(16).padStart(8, '0') +
      (this.laneB >>> 0).toString(16).padStart(8, '0');
  }
}

export function runFingerprintScenario(scenario: FingerprintScenario): FingerprintEntry {
  const state = scenario.build();
  const digest = new StateDigest();
  digest.pushState(state);
  for (let frame = 0; frame < scenario.frames; frame += 1) {
    stepGame(state, scenario.commands(frame, state), FIXED_DT);
    digest.pushState(state);
  }
  return { id: scenario.id, frames: scenario.frames, digest: digest.hex() };
}

export function runFingerprintSuite(scenarios: readonly FingerprintScenario[]): FingerprintEntry[] {
  return scenarios.map(runFingerprintScenario);
}

export function combinedDigest(entries: readonly FingerprintEntry[]): string {
  const digest = new StateDigest();
  for (const entry of entries) {
    for (let index = 0; index < entry.id.length; index += 1) {
      digest.pushNumber(entry.id.charCodeAt(index));
    }
    digest.pushNumber(entry.frames);
    for (let index = 0; index < entry.digest.length; index += 1) {
      digest.pushNumber(entry.digest.charCodeAt(index));
    }
  }
  return digest.hex();
}

/**
 * Scripted physics scenarios. These deliberately drive the tanks into walls,
 * corners, each other, and the ball so that every branch of the collision
 * resolver is exercised by the digest rather than only the common case.
 */
export function physicsScenarios(): FingerprintScenario[] {
  return [
    {
      id: 'physics/idle-drift',
      frames: 240,
      build: () => {
        const state = createInitialState();
        state.ball.velocity = { x: 210, y: 135 };
        return state;
      },
      commands: () => ({})
    },
    {
      id: 'physics/head-on-contact',
      frames: 240,
      build: () => {
        const state = createInitialState();
        state.tanks[0].position = { x: FIELD.length / 2 - 90, y: FIELD.width / 2 };
        state.tanks[1].position = { x: FIELD.length / 2 + 90, y: FIELD.width / 2 };
        return state;
      },
      commands: () => ({
        'red-0': { leftTrack: 1, rightTrack: 1 },
        'blue-0': { leftTrack: 1, rightTrack: 1 }
      })
    },
    {
      id: 'physics/tank-tank-shove',
      frames: 240,
      build: () => {
        const state = createInitialState();
        state.tanks[0].position = { x: FIELD.length / 2 - 60, y: FIELD.width / 2 + 12 };
        state.tanks[1].position = { x: FIELD.length / 2 + 60, y: FIELD.width / 2 - 12 };
        state.ball.position = { x: FIELD.length / 2, y: 120 };
        return state;
      },
      commands: () => ({
        'red-0': { leftTrack: 1, rightTrack: 1 },
        'blue-0': { leftTrack: -1, rightTrack: -1 }
      })
    },
    {
      id: 'physics/corner-pin',
      frames: 300,
      build: () => {
        const state = createInitialState();
        state.ball.position = { x: FIELD.length - FIELD.ballRadius - 6, y: FIELD.ballRadius + 6 };
        state.tanks[0].position = { x: FIELD.length - 200, y: 150 };
        state.tanks[0].angle = -0.6;
        state.tanks[1].position = { x: FIELD.length - 320, y: 90 };
        return state;
      },
      commands: () => ({
        'red-0': { leftTrack: 1, rightTrack: 1 },
        'blue-0': { leftTrack: 1, rightTrack: 0 }
      })
    },
    {
      id: 'physics/goal-and-kickoff',
      frames: 200,
      build: () => {
        const state = createInitialState();
        state.ball.position = { x: FIELD.length - 150, y: FIELD.width / 2 };
        state.ball.velocity = { x: 620, y: 0 };
        return state;
      },
      commands: () => ({
        'red-0': { leftTrack: 1, rightTrack: 1 },
        'blue-0': { leftTrack: 0, rightTrack: 0 }
      })
    },
    {
      id: 'physics/wall-slide',
      frames: 300,
      build: () => {
        const state = createInitialState();
        state.tanks[0].position = { x: 120, y: FIELD.width - 70 };
        state.tanks[0].angle = 1.2;
        state.tanks[1].position = { x: FIELD.length - 120, y: 70 };
        state.tanks[1].angle = -2.4;
        state.ball.position = { x: 200, y: FIELD.width - 60 };
        state.ball.velocity = { x: -240, y: 180 };
        return state;
      },
      commands: () => ({
        'red-0': { leftTrack: 1, rightTrack: -1 },
        'blue-0': { leftTrack: -1, rightTrack: 1 }
      })
    },
    {
      id: 'physics/stamina-drain',
      frames: 420,
      build: () => {
        const state = createInitialState();
        state.tanks[0].stamina = 6;
        state.tanks[1].stamina = 3;
        state.ball.position = { x: FIELD.length / 2 + 40, y: FIELD.width / 2 + 30 };
        return state;
      },
      commands: () => ({
        'red-0': { leftTrack: 1, rightTrack: 1 },
        'blue-0': { leftTrack: 1, rightTrack: 1 }
      })
    },
    randomWalkScenario('physics/random-walk-a', 2026, 600),
    randomWalkScenario('physics/random-walk-b', 91711, 600),
    randomWalkScenario('physics/random-walk-c', 500009, 600)
  ];
}

/**
 * Random command walks cover state space that hand-built scenarios miss. The
 * command set is exactly the nine policy actions, so the walk stays inside the
 * command space the real AI can produce.
 */
function randomWalkScenario(id: string, seed: number, frames: number): FingerprintScenario {
  return {
    id,
    frames,
    build: () => {
      const random = createSeededRandom(seed);
      const state = createInitialState();
      state.ball.position = {
        x: FIELD.ballRadius + random() * (FIELD.length - FIELD.ballRadius * 2),
        y: FIELD.ballRadius + random() * (FIELD.width - FIELD.ballRadius * 2)
      };
      state.ball.velocity = { x: (random() - 0.5) * 500, y: (random() - 0.5) * 500 };
      state.tanks[0].position = { x: 140 + random() * 300, y: 90 + random() * (FIELD.width - 180) };
      state.tanks[1].position = {
        x: FIELD.length - 140 - random() * 300,
        y: 90 + random() * (FIELD.width - 180)
      };
      state.tanks[0].angle = (random() - 0.5) * Math.PI * 2;
      state.tanks[1].angle = (random() - 0.5) * Math.PI * 2;
      return state;
    },
    commands: createRandomCommandScript(seed)
  };
}

function createRandomCommandScript(seed: number): (frame: number) => CommandMap {
  const random = createSeededRandom(seed ^ 0x5bf03635);
  let red: TankCommand = actionIndexToCommand(0);
  let blue: TankCommand = actionIndexToCommand(0);
  return (frame: number) => {
    if (frame % 6 === 0) {
      red = actionIndexToCommand(Math.floor(random() * 9));
      blue = actionIndexToCommand(Math.floor(random() * 9));
    }
    return { 'red-0': red, 'blue-0': blue };
  };
}

/**
 * Runtime scenarios drive the real browser AI (neural wrapper + tactical
 * rollout) against the traditional opponent, so the digest also pins the whole
 * decision stack, not just the physics kernel.
 */
export function runtimeScenarios(weights: NeuralWeights): FingerprintScenario[] {
  return [19, 31, 71].map((seed) => ({
    id: `runtime/seed-${seed}`,
    frames: 300,
    build: () => createSeededInitialState(seed, 0, 'red'),
    commands: createStrategyCommandScript(
      createNeuralStrategy({ weights, name: 'fingerprint-neural', tacticalRollout: true }),
      traditionalStrategy
    )
  }));
}

function createStrategyCommandScript(
  red: Strategy,
  blue: Strategy
): (frame: number, state: Readonly<GameState>) => CommandMap {
  let commands: CommandMap = {};
  let lastDecisionFrame = Number.NEGATIVE_INFINITY;
  return (_frame: number, state: Readonly<GameState>) => {
    if (state.frame - lastDecisionFrame >= 6) {
      commands = { ...red.decide(state, 'red'), ...blue.decide(state, 'blue') };
      lastDecisionFrame = state.frame;
    }
    return commands;
  };
}

/** Mirrors `policyGate` start-state generation so runtime digests match gate starts. */
function createSeededInitialState(seed: number, match: number, team: Team): GameState {
  const random = createSeededRandom(seed + match * 4099);
  const state = createInitialState();
  const attackFrameX = FIELD.length / 2 + (random() - 0.5) * FIELD.length * 0.12;
  const attackFrameY = FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.22;
  state.ball.position = {
    x: team === 'red' ? attackFrameX : FIELD.length - attackFrameX,
    y: team === 'red' ? attackFrameY : FIELD.width - attackFrameY
  };
  const vx = (random() - 0.5) * 120;
  const vy = (random() - 0.5) * 120;
  state.ball.velocity = {
    x: team === 'red' ? vx : -vx,
    y: team === 'red' ? vy : -vy
  };
  return state;
}

function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function allFingerprintScenarios(weights: NeuralWeights): FingerprintScenario[] {
  return [...physicsScenarios(), ...runtimeScenarios(weights)];
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  const weightsPath = stringArg(argv, '--weights') ?? 'public/models/neural-best.json';
  let weights: NeuralWeights;
  try {
    weights = loadWeightsPayload(readFileSync(weightsPath, 'utf8'));
  } catch {
    weights = defaultNeuralWeights();
  }

  const physicsOnly = argv.includes('--physics-only');
  const scenarios = physicsOnly ? physicsScenarios() : allFingerprintScenarios(weights);
  const started = Date.now();
  const entries = runFingerprintSuite(scenarios);
  const elapsed = Date.now() - started;

  for (const entry of entries) {
    console.log(`${entry.digest}  ${entry.id} frames=${entry.frames}`);
  }
  console.log(`combined=${combinedDigest(entries)} scenarios=${entries.length} ms=${elapsed}`);
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value && !value.startsWith('--') ? value : undefined;
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/fingerprint-simulation.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/fingerprint-simulation.js')) {
  main();
}
