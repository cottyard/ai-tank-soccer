import { AI_HZ, FIXED_DT, PHYSICS_HZ } from '../game/match';
import { cloneState, type GameState, type Tank, type Team } from '../game/model';
import { stepGame } from '../game/simulation';
import { AiClock, type CommandMap, type Strategy } from '../game/strategy';
import { actionIndexToCommand, POLICY_ACTION_COUNT } from './policyActions';
import { extractTankInputs } from './neuralStrategy';
import { createSeededInitialState } from './policyGate';
import type { ValueSample } from './valueNetwork';

/**
 * Monte-Carlo value labels.
 *
 * The hand-weighted position evaluator scores a state by ten terms someone
 * chose. This instead labels a state with what actually happened from it: the
 * signed, time-discounted identity of the next goal. That is the quantity the
 * tactical rollout is really trying to estimate when it compares candidate
 * actions, so it is the target to regress on.
 */

export type ValueSampleOptions = {
  candidate: Strategy;
  opponent: Strategy;
  seed: number;
  scenario: number;
  frames: number;
  /** Record a state every N frames; 6 matches the 5Hz decision cadence. */
  sampleEvery?: number;
  /** Frames after which an upcoming goal is worth 1/e of an immediate one. */
  decayFrames?: number;
};

type RecordedFrame = {
  frame: number;
  inputsByTeam: Record<Team, number[] | undefined>;
};

type GoalEventRecord = {
  frame: number;
  team: Team;
};

export function generateValueSamples(options: ValueSampleOptions): ValueSample[] {
  const sampleEvery = Math.max(1, Math.floor(options.sampleEvery ?? 6));
  const decayFrames = Math.max(1, options.decayFrames ?? 150);
  const initialState = createSeededInitialState(options.seed, options.scenario, 'red');
  const state = cloneState(initialState);
  const aiClock = new AiClock(options.candidate, options.opponent, PHYSICS_HZ, AI_HZ);

  const recorded: RecordedFrame[] = [];
  const goals: GoalEventRecord[] = [];

  for (let step = 0; step < options.frames; step += 1) {
    const commands = aiClock.update(state);
    if (step % sampleEvery === 0) {
      recorded.push({
        frame: state.frame,
        inputsByTeam: {
          red: teamInputs(state, 'red'),
          blue: teamInputs(state, 'blue')
        }
      });
    }
    stepGame(state, commands, FIXED_DT);
    if (state.lastGoal) {
      goals.push({ frame: state.lastGoal.frame, team: state.lastGoal.team });
    }
  }

  return buildSamples(recorded, goals, decayFrames);
}

export function buildSamples(
  recorded: readonly RecordedFrame[],
  goals: readonly GoalEventRecord[],
  decayFrames: number
): ValueSample[] {
  const samples: ValueSample[] = [];
  let goalCursor = 0;

  for (const entry of recorded) {
    while (goalCursor < goals.length && goals[goalCursor].frame < entry.frame) {
      goalCursor += 1;
    }
    const nextGoal = goals[goalCursor];
    for (const team of ['red', 'blue'] as const) {
      const inputs = entry.inputsByTeam[team];
      if (!inputs) {
        continue;
      }
      samples.push({ inputs, target: goalTarget(nextGoal, entry.frame, team, decayFrames) });
    }
  }

  return samples;
}

/**
 * +1 means "we are about to score", -1 means "we are about to concede", and a
 * state with no goal ahead of it is 0. Exponential decay makes the target
 * express urgency, so the network learns to separate a real chance from a
 * position that merely looks good.
 */
function goalTarget(
  nextGoal: GoalEventRecord | undefined,
  frame: number,
  team: Team,
  decayFrames: number
): number {
  if (!nextGoal) {
    return 0;
  }
  const decay = Math.exp(-(nextGoal.frame - frame) / decayFrames);
  return nextGoal.team === team ? decay : -decay;
}

function teamInputs(state: Readonly<GameState>, team: Team): number[] | undefined {
  const tank = controlledTank(state, team);
  return tank ? extractTankInputs(state, team, tank) : undefined;
}

function controlledTank(state: Readonly<GameState>, team: Team): Tank | undefined {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
}

/**
 * Wraps a strategy so it plays a uniformly random action a fraction of the
 * time. Purely on-policy states under-represent the positions a rollout
 * actually reaches after committing one action for many frames, and a value
 * function is only useful where it has seen data.
 */
export function createExploringStrategy(
  base: Strategy,
  explorationRate: number,
  seed: number
): Strategy {
  if (explorationRate <= 0) {
    return base;
  }
  const random = createSeededRandom(seed);

  return {
    name: `${base.name}-explore`,
    decide(state, team): CommandMap {
      const commands = base.decide(state, team);
      const result: CommandMap = {};
      for (const [tankId, command] of Object.entries(commands)) {
        result[tankId] = random() < explorationRate
          ? actionIndexToCommand(Math.floor(random() * POLICY_ACTION_COUNT))
          : command;
      }
      return result;
    }
  };
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
