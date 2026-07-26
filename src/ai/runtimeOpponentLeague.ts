import type { Strategy } from '../game/strategy';
import { evaluateRuntimePolicyAgainst, type RuntimeEvaluationResult } from './policyGate';
import { createNeuralStrategy } from './neuralStrategy';
import type { NeuralWeights } from './neuralWeights';
import { traditionalStrategy } from './traditionalStrategy';

export type RuntimeOpponentLifecycle = 'anchor' | 'rolling';

export type RuntimeOpponentKind =
  | 'traditional'
  | 'accepted-no-rollout'
  | 'accepted-runtime';

export type RuntimeOpponentProfile = {
  id: string;
  lifecycle: RuntimeOpponentLifecycle;
  kind: RuntimeOpponentKind;
  seeds: number[];
};

export type RuntimeOpponentLeagueConfig = {
  schemaVersion: 1;
  generation: number;
  opponents: RuntimeOpponentProfile[];
};

export type RuntimeOpponentLeagueOptions = {
  matches?: number;
  frames?: number;
};

export type RuntimeOpponentLeagueSeedResult = RuntimeEvaluationResult & {
  seed: number;
};

export type RuntimeOpponentLeagueRow = {
  id: string;
  lifecycle: RuntimeOpponentLifecycle;
  kind: RuntimeOpponentKind;
  result: RuntimeEvaluationResult;
  seeds: RuntimeOpponentLeagueSeedResult[];
};

export type RuntimeOpponentLeagueResult = {
  generation: number;
  rows: RuntimeOpponentLeagueRow[];
};

export function parseRuntimeOpponentLeagueConfig(value: unknown): RuntimeOpponentLeagueConfig {
  if (!isRecord(value) || value.schemaVersion !== 1) {
    throw new Error('Runtime opponent league config must use schemaVersion 1');
  }
  if (!isPositiveInteger(value.generation) || !Array.isArray(value.opponents)) {
    throw new Error('Runtime opponent league config requires a positive generation and opponents');
  }

  const opponents = value.opponents.map(parseOpponentProfile);
  const ids = new Set<string>();
  for (const opponent of opponents) {
    if (ids.has(opponent.id)) {
      throw new Error(`Duplicate runtime opponent id: ${opponent.id}`);
    }
    ids.add(opponent.id);
  }
  if (!opponents.some((opponent) => opponent.lifecycle === 'anchor')) {
    throw new Error('Runtime opponent league requires at least one fixed anchor');
  }
  if (opponents.filter((opponent) => opponent.lifecycle === 'rolling').length <=
      opponents.filter((opponent) => opponent.lifecycle === 'anchor').length) {
    throw new Error('Runtime opponent league must contain more rolling opponents than anchors');
  }

  return {
    schemaVersion: 1,
    generation: value.generation,
    opponents
  };
}

export function evaluateRuntimeOpponentLeague(
  candidateWeights: NeuralWeights,
  acceptedWeights: NeuralWeights,
  config: RuntimeOpponentLeagueConfig,
  options: RuntimeOpponentLeagueOptions = {}
): RuntimeOpponentLeagueResult {
  const matches = Math.max(1, Math.floor(options.matches ?? 8));
  if (matches % 2 !== 0) {
    throw new Error('Paired runtime opponent league evaluation requires an even match count');
  }

  return {
    generation: config.generation,
    rows: config.opponents.map((profile) => {
      const opponent = createRuntimeOpponentStrategy(profile.kind, acceptedWeights, profile.id);
      const seeds = profile.seeds.map((seed) => ({
        seed,
        ...evaluateRuntimePolicyAgainst(candidateWeights, opponent, {
          seed,
          matches,
          frames: options.frames,
          pairedStarts: true
        })
      }));
      return {
        id: profile.id,
        lifecycle: profile.lifecycle,
        kind: profile.kind,
        result: aggregateResults(seeds),
        seeds
      };
    })
  };
}

export function advanceRuntimeOpponentLeague(
  config: RuntimeOpponentLeagueConfig
): RuntimeOpponentLeagueConfig {
  const generation = config.generation + 1;
  const reserved = new Set(config.opponents.flatMap((profile) => profile.seeds));

  return {
    ...config,
    generation,
    opponents: config.opponents.map((profile, profileIndex) => {
      if (profile.lifecycle === 'anchor') {
        return { ...profile, seeds: [...profile.seeds] };
      }

      const retained = profile.seeds.length > 0 ? [profile.seeds[profile.seeds.length - 1]] : [];
      retained.forEach((seed) => reserved.add(seed));
      const seeds = [
        ...retained,
        ...generateRollingSeeds(generation, profileIndex, profile.seeds.length - retained.length, reserved)
      ];
      seeds.forEach((seed) => reserved.add(seed));
      return { ...profile, seeds };
    })
  };
}

export function createRuntimeOpponentStrategy(
  kind: RuntimeOpponentKind,
  acceptedWeights: NeuralWeights,
  name: string = kind
): Strategy {
  if (kind === 'traditional') {
    return traditionalStrategy;
  }
  return createNeuralStrategy({
    weights: acceptedWeights,
    name,
    tacticalRollout: kind === 'accepted-runtime'
  });
}

function aggregateResults(rows: readonly RuntimeOpponentLeagueSeedResult[]): RuntimeEvaluationResult {
  const divisor = Math.max(1, rows.length);
  const goalsFor = rows.reduce((sum, row) => sum + row.goalsFor, 0);
  const goalsAgainst = rows.reduce((sum, row) => sum + row.goalsAgainst, 0);
  return {
    score: rows.reduce((sum, row) => sum + row.score, 0) / divisor,
    goalDiff: goalsFor - goalsAgainst,
    ballProgress: rows.reduce((sum, row) => sum + row.ballProgress, 0) / divisor,
    goalsFor,
    goalsAgainst,
    winProxy: rows.reduce((sum, row) => sum + row.winProxy, 0) / divisor
  };
}

function generateRollingSeeds(
  generation: number,
  profileIndex: number,
  count: number,
  reserved: ReadonlySet<number>
): number[] {
  const seeds: number[] = [];
  let state = ((generation * 2654435761) ^ ((profileIndex + 1) * 2246822519)) >>> 0;
  while (seeds.length < count) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const seed = 151 + (state % 999_749);
    if (!reserved.has(seed) && !seeds.includes(seed)) {
      seeds.push(seed);
    }
  }
  return seeds;
}

function parseOpponentProfile(value: unknown, index: number): RuntimeOpponentProfile {
  if (!isRecord(value)) {
    throw new Error(`Runtime opponent at index ${index} must be an object`);
  }
  if (typeof value.id !== 'string' || value.id.trim().length === 0) {
    throw new Error(`Runtime opponent at index ${index} requires an id`);
  }
  if (value.lifecycle !== 'anchor' && value.lifecycle !== 'rolling') {
    throw new Error(`Runtime opponent ${value.id} has an invalid lifecycle`);
  }
  if (
    value.kind !== 'traditional' &&
    value.kind !== 'accepted-no-rollout' &&
    value.kind !== 'accepted-runtime'
  ) {
    throw new Error(`Runtime opponent ${value.id} has an invalid kind`);
  }
  if (!Array.isArray(value.seeds) || value.seeds.length === 0 || !value.seeds.every(isPositiveInteger)) {
    throw new Error(`Runtime opponent ${value.id} requires positive integer seeds`);
  }

  return {
    id: value.id,
    lifecycle: value.lifecycle,
    kind: value.kind,
    seeds: [...value.seeds]
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0;
}
