import { describe, expect, it } from 'vitest';
import { FIELD, createInitialState, type GameState, type Team } from '../src/game/model';
import { chooseTacticalAction } from '../src/ai/tacticalRollout';

/**
 * The tuning knobs exist so search shape can be measured on the large-sample
 * benchmark instead of assumed. They must be inert unless asked for, and they
 * must actually do something when asked for; a knob that is silently a no-op
 * would make a benchmark result meaningless.
 */

function contestedState(): GameState {
  const state = createInitialState();
  state.ball.position = { x: FIELD.length - 300, y: FIELD.width / 2 + 20 };
  state.ball.velocity = { x: 40, y: 0 };
  state.tanks[0].position = { x: FIELD.length - 400, y: FIELD.width / 2 + 20 };
  state.tanks[0].angle = 0;
  state.tanks[1].position = { x: FIELD.length - 220, y: FIELD.width / 2 + 10 };
  state.tanks[1].angle = Math.PI;
  return state;
}

describe('tactical rollout tuning', () => {
  it('leaves the default horizon untouched when no tuning is supplied', () => {
    const state = contestedState();
    const untuned = chooseTacticalAction({ state, team: 'red', policyActionIndex: 2 });
    const explicitDefault = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { defaultFrames: 18 }
    });

    expect(untuned.actionScores).toEqual(explicitDefault.actionScores);
  });

  it('changes scores when the default horizon changes', () => {
    const state = contestedState();
    const shallow = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { defaultFrames: 18 }
    });
    const deep = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { defaultFrames: 60 }
    });

    expect(deep.actionScores).not.toEqual(shallow.actionScores);
  });

  it('ignores an opponent policy unless the opponent model asks for it', () => {
    const state = contestedState();
    const calls: Team[] = [];
    const opponentPolicy = (_state: Readonly<GameState>, team: Team): number => {
      calls.push(team);
      return 8;
    };

    const frozen = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      opponentPolicy
    });

    expect(calls).toHaveLength(0);

    const reacting = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { opponentModel: 'policy' },
      opponentPolicy
    });

    // The opponent is re-decided for its own team, and a moving opponent changes
    // the predicted outcome relative to planning against a stationary one.
    expect(calls.length).toBeGreaterThan(0);
    expect(new Set(calls)).toEqual(new Set<Team>(['blue']));
    expect(reacting.actionScores).not.toEqual(frozen.actionScores);
  });

  it('treats an explicit stop opponent model as the historical behaviour', () => {
    const state = contestedState();
    const baseline = chooseTacticalAction({ state, team: 'red', policyActionIndex: 2 });
    const explicitStop = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { opponentModel: 'stop' },
      opponentPolicy: () => 8
    });

    expect(explicitStop.actionScores).toEqual(baseline.actionScores);
  });

  it('applies an improvement margin override', () => {
    const state = contestedState();
    const permissive = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { improvementMargin: 0 }
    });
    const strict = chooseTacticalAction({
      state,
      team: 'red',
      policyActionIndex: 2,
      tuning: { improvementMargin: 1000 }
    });

    // An unreachable margin must always fall back to the raw policy action.
    expect(strict.actionIndex).toBe(2);
    expect(permissive.actionScores).toEqual(strict.actionScores);
  });
});
