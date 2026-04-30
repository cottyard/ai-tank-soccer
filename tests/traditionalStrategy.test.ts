import { describe, expect, it } from 'vitest';
import { FIELD, createInitialState, type GameState, type Tank, type Team } from '../src/game/model';
import { simulateMatch } from '../src/game/match';
import { stepGame } from '../src/game/simulation';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import type { TankCommand } from '../src/game/strategy';

const LEGAL_TRACKS = [-1, 0, 1];
const CENTER = { x: FIELD.length / 2, y: FIELD.width / 2 };

function teamTank(state: GameState, team: Team): Tank {
  const found = state.tanks.find((candidate) => candidate.id === `${team}-0`);
  if (!found) {
    throw new Error(`Missing tank ${team}-0`);
  }
  return found;
}

function placeTank(
  state: GameState,
  team: Team,
  x: number,
  y: number,
  angle: number,
  stamina: number = FIELD.tankStamina
): Tank {
  const placed = teamTank(state, team);
  placed.position = { x, y };
  placed.velocity = { x: 0, y: 0 };
  placed.angle = angle;
  placed.angularVelocity = 0;
  placed.stamina = stamina;
  return placed;
}

function placeOpponentAway(state: GameState, team: Team): void {
  const opponent = team === 'red' ? 'blue' : 'red';
  placeTank(
    state,
    opponent,
    team === 'red' ? FIELD.length - 130 : 130,
    FIELD.width - 95,
    team === 'red' ? Math.PI : 0
  );
}

function decide(team: Team, state: GameState): TankCommand {
  const commands = traditionalStrategy.decide(state, team);
  return commands[`${team}-0`];
}

function applyCommandsForFrames(
  state: GameState,
  commands: Record<string, TankCommand>,
  frames: number
): void {
  for (let i = 0; i < frames; i += 1) {
    stepGame(state, commands, 1 / 30);
  }
}

function applyStrategyForFrames(state: GameState, team: Team, frames: number): void {
  let commands: Record<string, TankCommand> = {};

  for (let i = 0; i < frames; i += 1) {
    if (i % 6 === 0) {
      commands = traditionalStrategy.decide(state, team);
    }
    stepGame(state, commands, 1 / 30);
  }
}

function createSeededState(seed: number): GameState {
  const state = createInitialState();
  const redJitter = seededUnit(seed) - 0.5;
  const blueJitter = seededUnit(seed + 31) - 0.5;

  state.ball.position = {
    x: CENTER.x + (seededUnit(seed + 11) - 0.5) * 180,
    y: CENTER.y + (seededUnit(seed + 17) - 0.5) * 140
  };
  state.ball.velocity = {
    x: (seededUnit(seed + 23) - 0.5) * 220,
    y: (seededUnit(seed + 29) - 0.5) * 160
  };
  placeTank(state, 'red', 170 + redJitter * 60, CENTER.y - 85 + redJitter * 50, 0);
  placeTank(state, 'blue', FIELD.length - 170 + blueJitter * 60, CENTER.y + 85 + blueJitter * 50, Math.PI);
  return state;
}

function seededUnit(seed: number): number {
  const value = Math.sin(seed * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function expectLegal(command: TankCommand): void {
  expect(LEGAL_TRACKS).toContain(command.leftTrack);
  expect(LEGAL_TRACKS).toContain(command.rightTrack);
}

describe('traditional strategy', () => {
  it('returns deterministic legal commands only for the requested one-tank team', () => {
    const state = createInitialState();

    const firstRed = traditionalStrategy.decide(state, 'red');
    const secondRed = traditionalStrategy.decide(state, 'red');
    const blue = traditionalStrategy.decide(state, 'blue');

    expect(firstRed).toEqual(secondRed);
    expect(Object.keys(firstRed)).toEqual(['red-0']);
    expect(Object.keys(blue)).toEqual(['blue-0']);
    expectLegal(firstRed['red-0']);
    expectLegal(blue['blue-0']);
  });

  it('drives an aligned tank through the ball toward the opponent goal for either side', () => {
    for (const team of ['red', 'blue'] as const) {
      const state = createInitialState();
      const direction = team === 'red' ? 1 : -1;
      state.ball.position = { x: CENTER.x + direction * 90, y: CENTER.y };
      state.ball.velocity = { x: 0, y: 0 };
      placeTank(
        state,
        team,
        state.ball.position.x - direction * (FIELD.ballRadius + FIELD.tankLength + 2),
        CENTER.y,
        team === 'red' ? 0 : Math.PI
      );
      placeOpponentAway(state, team);

      expect(decide(team, state)).toEqual({ leftTrack: 1, rightTrack: 1 });
    }
  });

  it('moves into the own goal lane when the ball is deep and incoming', () => {
    const state = createInitialState();
    state.ball.position = { x: 245, y: CENTER.y };
    state.ball.velocity = { x: -430, y: 0 };
    const defender = placeTank(state, 'red', 112, CENTER.y - 118, Math.PI / 2);
    placeOpponentAway(state, 'red');
    const startGap = Math.abs(defender.position.y - CENTER.y);

    applyStrategyForFrames(state, 'red', 18);

    expect(Math.abs(defender.position.y - CENTER.y)).toBeLessThan(startGap);
    expect(defender.position.x).toBeLessThan(220);
  });

  it('avoids pushing a deep central ball toward its own goal from the wrong side', () => {
    const state = createInitialState();
    state.ball.position = { x: 118, y: CENTER.y };
    state.ball.velocity = { x: 0, y: 0 };
    placeTank(state, 'red', 285, CENTER.y, Math.PI);
    placeOpponentAway(state, 'red');

    const command = decide('red', state);
    applyCommandsForFrames(state, { 'red-0': command }, 18);

    expect(command).not.toEqual({ leftTrack: 1, rightTrack: 1 });
    expect(state.score.blue).toBe(0);
    expect(state.ball.position.x).toBeGreaterThanOrEqual(112);
  });

  it('rests at low stamina when away from decisive ball contact', () => {
    const state = createInitialState();
    state.ball.position = { x: 690, y: CENTER.y };
    state.ball.velocity = { x: 0, y: 0 };
    placeTank(state, 'red', 310, CENTER.y + 190, 0, 8);
    placeOpponentAway(state, 'red');

    expect(decide('red', state)).toEqual({ leftTrack: 0, rightTrack: 0 });
  });

  it('starts conserving stamina below half charge when away from urgent contact', () => {
    const state = createInitialState();
    state.ball.position = { x: 690, y: CENTER.y };
    state.ball.velocity = { x: 0, y: 0 };
    placeTank(state, 'red', 310, CENTER.y + 190, 0, FIELD.tankStamina * 0.49);
    placeOpponentAway(state, 'red');

    expect(decide('red', state)).toEqual({ leftTrack: 0, rightTrack: 0 });
  });

  it('spends sub-half stamina for urgent own-goal defense', () => {
    const state = createInitialState();
    state.ball.position = { x: 220, y: CENTER.y };
    state.ball.velocity = { x: -380, y: 0 };
    placeTank(state, 'red', 120, CENTER.y - 72, Math.PI / 2, FIELD.tankStamina * 0.42);
    placeOpponentAway(state, 'red');

    expect(decide('red', state)).not.toEqual({ leftTrack: 0, rightTrack: 0 });
  });

  it('spends low stamina when already in useful scoring contact', () => {
    const state = createInitialState();
    state.ball.position = { x: FIELD.length - 180, y: CENTER.y };
    state.ball.velocity = { x: 0, y: 0 };
    placeTank(
      state,
      'red',
      state.ball.position.x - FIELD.ballRadius - FIELD.tankLength - 2,
      CENTER.y,
      0,
      8
    );
    placeOpponentAway(state, 'red');

    expect(decide('red', state)).toEqual({ leftTrack: 1, rightTrack: 1 });
  });

  it('recycles or escapes an opponent-corner ball instead of wedging into the wall', () => {
    const state = createInitialState();
    state.ball.position = {
      x: FIELD.length - FIELD.ballRadius - 12,
      y: FIELD.ballRadius + 8
    };
    state.ball.velocity = { x: 0, y: 0 };
    placeTank(
      state,
      'red',
      state.ball.position.x - FIELD.ballRadius - FIELD.tankLength - 4,
      state.ball.position.y,
      0
    );
    placeOpponentAway(state, 'red');
    const startX = state.ball.position.x;
    const startY = state.ball.position.y;

    applyStrategyForFrames(state, 'red', 180);

    const escapedWall = state.ball.position.y > startY + 22;
    const recycledBack = state.ball.position.x < startX - 45;
    expect(state.score.blue).toBe(0);
    expect(state.score.red > 0 || escapedWall || recycledBack).toBe(true);
  });

  it('runs a seeded traditional-vs-traditional match without crashing', () => {
    const result = simulateMatch({
      red: traditionalStrategy,
      blue: traditionalStrategy,
      frames: 420,
      initialState: createSeededState(42)
    });

    expect(result.redDecisions).toBeGreaterThan(0);
    expect(result.blueDecisions).toBeGreaterThan(0);
    expect(Number.isFinite(result.state.ball.position.x)).toBe(true);
    expect(Number.isFinite(result.state.ball.position.y)).toBe(true);
    expect(result.state.score.red).toBeGreaterThanOrEqual(0);
    expect(result.state.score.blue).toBeGreaterThanOrEqual(0);
  });
});
