import { cloneState, createInitialState, type GameState, type Team } from './model';
import { stepGame } from './simulation';
import { AiClock, type CommandMap, type Strategy } from './strategy';

export const PHYSICS_HZ = 30;
export const AI_HZ = 5;
export const FIXED_DT = 1 / PHYSICS_HZ;

export type MatchOptions = {
  red: Strategy;
  blue: Strategy;
  frames: number;
  initialState?: GameState;
};

export type MatchResult = {
  state: GameState;
  redDecisions: number;
  blueDecisions: number;
};

export function simulateMatch(options: MatchOptions): MatchResult {
  const state = options.initialState ? cloneState(options.initialState) : createInitialState();
  const decisions: Record<Team, number> = { red: 0, blue: 0 };
  const red = countDecisions(options.red, 'red', decisions);
  const blue = countDecisions(options.blue, 'blue', decisions);
  const aiClock = new AiClock(red, blue, PHYSICS_HZ, AI_HZ);

  for (let i = 0; i < options.frames; i += 1) {
    const commands = aiClock.update(state);
    stepGame(state, commands, FIXED_DT);
  }

  return {
    state,
    redDecisions: decisions.red,
    blueDecisions: decisions.blue
  };
}

function countDecisions(
  strategy: Strategy,
  team: Team,
  decisions: Record<Team, number>
): Strategy {
  return {
    name: strategy.name,
    decide(state, requestedTeam): CommandMap {
      decisions[team] += 1;
      return strategy.decide(state, requestedTeam);
    }
  };
}
