import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import {
  advanceRuntimeOpponentLeague,
  evaluateRuntimeOpponentLeague,
  parseRuntimeOpponentLeagueConfig
} from '../src/ai/runtimeOpponentLeague';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import { parseRuntimeLeagueArgs } from '../scripts/evaluate-runtime-league';

describe('runtime opponent league', () => {
  it('keeps a fixed classic anchor and makes the majority of opponents rolling', () => {
    const config = loadConfig();
    const anchors = config.opponents.filter((opponent) => opponent.lifecycle === 'anchor');
    const rolling = config.opponents.filter((opponent) => opponent.lifecycle === 'rolling');

    expect(anchors).toEqual([
      expect.objectContaining({ id: 'classic-traditional', kind: 'traditional' })
    ]);
    expect(rolling.length).toBeGreaterThan(anchors.length);
  });

  it('advances rolling seeds while preserving anchors and one continuity seed', () => {
    const current = loadConfig();
    const next = advanceRuntimeOpponentLeague(current);

    expect(next.generation).toBe(current.generation + 1);
    expect(next.opponents[0]).toEqual(current.opponents[0]);
    for (let index = 1; index < current.opponents.length; index += 1) {
      const before = current.opponents[index];
      const after = next.opponents[index];
      expect(after.seeds).toHaveLength(before.seeds.length);
      expect(after.seeds[0]).toBe(before.seeds[before.seeds.length - 1]);
      expect(after.seeds.slice(1)).not.toEqual(before.seeds.slice(0, -1));
    }
  });

  it('evaluates the candidate against every anchor and rolling opponent', () => {
    const config = loadConfig();
    const result = evaluateRuntimeOpponentLeague(
      defaultNeuralWeights(),
      defaultNeuralWeights(),
      {
        ...config,
        opponents: config.opponents.map((opponent) => ({ ...opponent, seeds: [opponent.seeds[0]] }))
      },
      { matches: 2, frames: 30 }
    );

    expect(result.rows.map((row) => row.id)).toEqual(config.opponents.map((opponent) => opponent.id));
    for (const row of result.rows) {
      expect(row.seeds).toHaveLength(1);
      expect(Number.isFinite(row.result.score)).toBe(true);
      expect(row.result.winProxy).toBeGreaterThanOrEqual(0);
      expect(row.result.winProxy).toBeLessThanOrEqual(1);
    }
  });

  it('rejects unpaired match counts before running a league gate', () => {
    const config = loadConfig();

    expect(() => evaluateRuntimeOpponentLeague(
      defaultNeuralWeights(),
      defaultNeuralWeights(),
      config,
      { matches: 1, frames: 30 }
    )).toThrow(/even match count/);
  });

  it('parses lightweight CLI overrides for local gate checks', () => {
    expect(parseRuntimeLeagueArgs(['--matches', '2', '--frames', '90'])).toMatchObject({
      matches: 2,
      frames: 90,
      details: false,
      opponents: []
    });
  });
});

function loadConfig() {
  return parseRuntimeOpponentLeagueConfig(
    JSON.parse(readFileSync('config/runtime-opponent-league.json', 'utf8')) as unknown
  );
}
