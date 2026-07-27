import { describe, expect, it } from 'vitest';
import {
  FIELD,
  createInitialState,
  type GameState,
  type Tank,
  type Team
} from '../src/game/model';
import {
  AUGMENTED_VALUE_INPUT_COUNT,
  PLAIN_VALUE_INPUT_COUNT,
  valueInputs
} from '../src/ai/valueFeatures';
import { evaluateValue, createValueWeights, valueWeightCount } from '../src/ai/valueNetwork';
import { BUNDLED_VALUE_INPUT_COUNT, BUNDLED_VALUE_WEIGHTS } from '../src/ai/bundledValueModel';

function finishState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.length - 200, y: FIELD.width / 2 + 20 };
  state.ball.velocity = { x: 120, y: -10 };
  return state;
}

function redTank(state: GameState): Tank {
  return state.tanks.find((candidate) => candidate.team === 'red' && candidate.index === 0)!;
}

describe('value features', () => {
  it('produces the declared input widths', () => {
    const state = finishState();
    const tank = redTank(state);
    expect(valueInputs(state, 'red', tank, false)).toHaveLength(PLAIN_VALUE_INPUT_COUNT);
    expect(valueInputs(state, 'red', tank, true)).toHaveLength(AUGMENTED_VALUE_INPUT_COUNT);
  });

  it('keeps the plain features as a prefix of the augmented ones', () => {
    const state = finishState();
    const tank = redTank(state);
    const plain = valueInputs(state, 'red', tank, false);
    const augmented = valueInputs(state, 'red', tank, true);
    expect(augmented.slice(0, PLAIN_VALUE_INPUT_COUNT)).toEqual(plain);
  });

  it('bounds every feature so a tanh network is not driven into saturation', () => {
    for (const augmented of [false, true]) {
      const state = finishState();
      // Extreme state: ball jammed in the attacking corner at high speed.
      state.ball.position = { x: FIELD.length - FIELD.ballRadius, y: FIELD.ballRadius };
      state.ball.velocity = { x: 4000, y: -4000 };
      state.score.red = 9;
      const tank = redTank(state);
      tank.stamina = 0;
      for (const value of valueInputs(state, 'red', tank, augmented)) {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value)).toBeLessThanOrEqual(1.0000001);
      }
    }
  });

  it('is team-relative for complete point-mirrored states', () => {
    const redState = attackFrameState('red');
    const blueState = attackFrameState('blue');
    const red = valueInputs(redState, 'red', controlledTank(redState, 'red'), true);
    const blue = valueInputs(blueState, 'blue', controlledTank(blueState, 'blue'), true);

    expectVectorsClose(blue, red);
  });

  it('matches the bundled model to the plain feature width', () => {
    // The strategy picks the feature set by weight count, so a mismatch here
    // would silently feed the wrong vector into the promoted model.
    expect(BUNDLED_VALUE_INPUT_COUNT).toBe(PLAIN_VALUE_INPUT_COUNT);
    expect(BUNDLED_VALUE_WEIGHTS).toHaveLength(valueWeightCount(BUNDLED_VALUE_INPUT_COUNT));

    const state = finishState();
    const value = evaluateValue(
      valueInputs(state, 'red', redTank(state), false),
      BUNDLED_VALUE_WEIGHTS
    );
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value)).toBeLessThanOrEqual(1);
  });

  it('rejects a feature vector that does not match the weight shape', () => {
    const state = finishState();
    const augmentedInputs = valueInputs(state, 'red', redTank(state), true);
    // Augmented features against plain-width weights must throw rather than
    // silently reading past the end of a layer.
    expect(() => evaluateValue(augmentedInputs, BUNDLED_VALUE_WEIGHTS)).toThrow(/value weights/);

    const augmentedWeights = createValueWeights(AUGMENTED_VALUE_INPUT_COUNT, 3);
    expect(evaluateValue(augmentedInputs, augmentedWeights)).toBeTypeOf('number');
  });
});

function attackFrameState(team: Team): GameState {
  const state = createInitialState();
  const opponent = team === 'red' ? 'blue' : 'red';
  const controlled = controlledTank(state, team);
  const defender = controlledTank(state, opponent);

  state.score[team] = 2;
  state.score[opponent] = 1;
  state.ball.position = fieldPoint(team, 790, FIELD.width / 2 + 70);
  state.ball.velocity = fieldVector(team, 150, -40);

  controlled.position = fieldPoint(team, 665, FIELD.width / 2 + 42);
  controlled.velocity = fieldVector(team, 34, -12);
  controlled.angle = fieldAngle(team, 0.38);
  controlled.angularVelocity = 0.7;
  controlled.stamina = controlled.maxStamina * 0.54;

  defender.position = fieldPoint(team, 825, FIELD.width / 2 - 78);
  defender.velocity = fieldVector(team, -18, 16);
  defender.angle = fieldAngle(opponent, -0.24);
  defender.angularVelocity = -0.3;
  defender.stamina = defender.maxStamina * 0.81;

  return state;
}

function controlledTank(state: GameState, team: Team): Tank {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0)!;
}

function fieldPoint(team: Team, attackX: number, attackY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackX : FIELD.length - attackX,
    y: team === 'red' ? attackY : FIELD.width - attackY
  };
}

function fieldVector(team: Team, attackX: number, attackY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackX : -attackX,
    y: team === 'red' ? attackY : -attackY
  };
}

function fieldAngle(team: Team, attackAngle: number): number {
  return team === 'red' ? attackAngle : attackAngle + Math.PI;
}

function expectVectorsClose(actual: readonly number[], expected: readonly number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    expect(actual[index]).toBeCloseTo(expected[index], 10);
  }
}
