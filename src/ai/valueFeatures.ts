import type { GameState, Tank, Team } from '../game/model';
import { extractTankInputs } from './neuralStrategy';
import { evaluatePosition } from './positionEvaluation';

/**
 * Inputs for the state value network.
 *
 * `plain` is the same 36 team-relative features the policy consumes.
 *
 * `augmented` appends the hand-weighted evaluator's own breakdown. This was
 * tested as a way to raise the usable learned-value blend, but it did not help:
 * pure learned play remained as weak as with the plain inputs. The optional
 * feature set remains available so that result is reproducible and future
 * training experiments can distinguish model shape explicitly.
 */

export const PLAIN_VALUE_INPUT_COUNT = 36;
export const AUGMENTED_VALUE_INPUT_COUNT = 46;

export function valueInputs(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  augmented: boolean
): number[] {
  const inputs = extractTankInputs(state, team, tank);
  if (!augmented) {
    return inputs;
  }

  const { breakdown } = evaluatePosition(state, team);
  inputs.push(
    clamp(breakdown.goal / 3),
    breakdown.ballProgress,
    breakdown.shotLane,
    breakdown.finishThreat,
    breakdown.shotVelocity,
    breakdown.contest,
    breakdown.possession,
    breakdown.ownDanger,
    breakdown.cornerEscape,
    clamp(breakdown.stamina)
  );
  return inputs;
}

function clamp(value: number): number {
  return Math.max(-1, Math.min(1, value));
}
