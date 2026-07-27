import { AI_HZ, FIXED_DT, PHYSICS_HZ } from '../game/match';
import { cloneState, type GameState, type Tank, type Team } from '../game/model';
import { stepGame } from '../game/simulation';
import { AiClock, type CommandMap, type Strategy } from '../game/strategy';
import { actionIndexToCommand, POLICY_ACTION_COUNT } from './policyActions';
import { valueInputs } from './valueFeatures';
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
  /** Append the heuristic breakdown to the feature vector. */
  augmented?: boolean;
  /** Number of independently labelled rollout-terminal states to add. */
  forksPerMatch?: number;
  /** Frames to hold the sampled action while the opponent is stopped. */
  forkRolloutFrames?: number;
  /** Maximum continuation length used to find the next goal from a fork. */
  forkPlayoutFrames?: number;
  /** Training weight assigned to each forked sample. */
  forkWeight?: number;
  /** Unperturbed strategies used only by fork continuations. */
  forkCandidate?: Strategy;
  forkOpponent?: Strategy;
  forkSeed?: number;
};

type RecordedFrame = {
  frame: number;
  inputsByTeam: Record<Team, number[] | undefined>;
};

type GoalEventRecord = {
  frame: number;
  team: Team;
};

type ForkPlan = {
  step: number;
  team: Team;
  actionIndex: number;
};

export function generateValueSamples(options: ValueSampleOptions): ValueSample[] {
  const sampleEvery = Math.max(1, Math.floor(options.sampleEvery ?? 6));
  const decayFrames = Math.max(1, options.decayFrames ?? 150);
  const augmented = options.augmented ?? false;
  const initialState = createSeededInitialState(options.seed, options.scenario, 'red');
  const state = cloneState(initialState);
  const aiClock = new AiClock(options.candidate, options.opponent, PHYSICS_HZ, AI_HZ);
  const forks = planForks(options, sampleEvery);
  const forkCandidate = options.forkCandidate ?? options.candidate;
  const forkOpponent = options.forkOpponent ?? options.opponent;

  const recorded: RecordedFrame[] = [];
  const goals: GoalEventRecord[] = [];
  const forkedSamples: ValueSample[] = [];

  for (let step = 0; step < options.frames; step += 1) {
    const commands = aiClock.update(state);
    if (step % sampleEvery === 0) {
      recorded.push({
        frame: state.frame,
        inputsByTeam: {
          red: teamInputs(state, 'red', augmented),
          blue: teamInputs(state, 'blue', augmented)
        }
      });
    }
    for (const fork of forks.get(step) ?? []) {
      const sample = forkedValueSample(
        state,
        fork,
        forkCandidate,
        forkOpponent,
        Math.max(1, Math.floor(options.forkRolloutFrames ?? 18)),
        Math.max(1, Math.floor(options.forkPlayoutFrames ?? options.frames)),
        decayFrames,
        augmented,
        Math.max(0, options.forkWeight ?? 1)
      );
      if (sample) {
        forkedSamples.push(sample);
      }
    }
    stepGame(state, commands, FIXED_DT);
    if (state.lastGoal) {
      goals.push({ frame: state.lastGoal.frame, team: state.lastGoal.team });
    }
  }

  return [...buildSamples(recorded, goals, decayFrames), ...forkedSamples];
}

function planForks(options: ValueSampleOptions, sampleEvery: number): Map<number, ForkPlan[]> {
  const count = Math.max(0, Math.floor(options.forksPerMatch ?? 0));
  const plans = new Map<number, ForkPlan[]>();
  if (count === 0 || options.frames <= 0) {
    return plans;
  }

  const decisionCount = Math.max(1, Math.ceil(options.frames / sampleEvery));
  const random = createSeededRandom(
    options.forkSeed ?? (options.seed ^ Math.imul(options.scenario + 1, 0x9e3779b1))
  );
  for (let index = 0; index < count; index += 1) {
    // Stratification spreads a small number of expensive continuations across
    // the whole source match instead of clustering them near kickoff.
    const firstDecision = Math.floor(index * decisionCount / count);
    const nextDecision = Math.max(firstDecision + 1, Math.floor((index + 1) * decisionCount / count));
    const decision = firstDecision + Math.floor(random() * (nextDecision - firstDecision));
    const step = Math.min(options.frames - 1, decision * sampleEvery);
    const plan: ForkPlan = {
      step,
      team: (index + options.seed) % 2 === 0 ? 'red' : 'blue',
      actionIndex: Math.floor(random() * POLICY_ACTION_COUNT)
    };
    const atStep = plans.get(step) ?? [];
    atStep.push(plan);
    plans.set(step, atStep);
  }
  return plans;
}

function forkedValueSample(
  source: Readonly<GameState>,
  plan: ForkPlan,
  candidate: Strategy,
  opponent: Strategy,
  rolloutFrames: number,
  playoutFrames: number,
  decayFrames: number,
  augmented: boolean,
  weight: number
): ValueSample | undefined {
  const terminal = cloneState(source as GameState);
  const tank = controlledTank(terminal, plan.team);
  if (!tank) {
    return undefined;
  }

  // This is the default tactical-search transition: one controlled action is
  // held while omitted opponent commands sanitize to a full stop.
  const commands: CommandMap = {
    [tank.id]: actionIndexToCommand(plan.actionIndex)
  };
  for (let frame = 0; frame < rolloutFrames; frame += 1) {
    stepGame(terminal, commands, FIXED_DT);
  }

  const inputs = teamInputs(terminal, plan.team, augmented);
  if (!inputs) {
    return undefined;
  }
  const sampleFrame = terminal.frame;
  const continuationClock = new AiClock(candidate, opponent, PHYSICS_HZ, AI_HZ);
  let nextGoal: GoalEventRecord | undefined;
  for (let frame = 0; frame < playoutFrames; frame += 1) {
    const continuationCommands = continuationClock.update(terminal);
    stepGame(terminal, continuationCommands, FIXED_DT);
    if (terminal.lastGoal) {
      nextGoal = {
        frame: terminal.lastGoal.frame,
        team: terminal.lastGoal.team
      };
      break;
    }
  }

  return {
    inputs,
    target: goalTarget(nextGoal, sampleFrame, plan.team, decayFrames),
    weight
  };
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

function teamInputs(
  state: Readonly<GameState>,
  team: Team,
  augmented: boolean
): number[] | undefined {
  const tank = controlledTank(state, team);
  return tank ? valueInputs(state, team, tank, augmented) : undefined;
}

function controlledTank(state: Readonly<GameState>, team: Team): Tank | undefined {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
}

/**
 * Wraps a strategy so it sometimes commits to a uniformly random action for
 * several consecutive decisions.
 *
 * This was tested as a way to expose the model to rollout-like states. Holding
 * an action for a full rollout horizon halved label variance and regressed play,
 * because disturbing the source match also removed its outcome signal. Forked
 * samples above preserve this wrapper for baseline reproduction while moving
 * the disruptive action and its label playout onto an independent clone.
 */
export function createExploringStrategy(
  base: Strategy,
  explorationRate: number,
  seed: number,
  commitDecisions = 1
): Strategy {
  if (explorationRate <= 0) {
    return base;
  }
  const random = createSeededRandom(seed);
  const holds = new Map<string, { actionIndex: number; remaining: number }>();
  const hold = Math.max(1, Math.floor(commitDecisions));

  return {
    name: `${base.name}-explore`,
    decide(state, team): CommandMap {
      const commands = base.decide(state, team);
      const result: CommandMap = {};
      for (const [tankId, command] of Object.entries(commands)) {
        const active = holds.get(tankId);
        if (active && active.remaining > 0) {
          active.remaining -= 1;
          result[tankId] = actionIndexToCommand(active.actionIndex);
          continue;
        }
        if (random() < explorationRate) {
          const actionIndex = Math.floor(random() * POLICY_ACTION_COUNT);
          holds.set(tankId, { actionIndex, remaining: hold - 1 });
          result[tankId] = actionIndexToCommand(actionIndex);
          continue;
        }
        result[tankId] = command;
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
