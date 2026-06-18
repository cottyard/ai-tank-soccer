import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { POLICY_ACTION_COUNT, actionIndexToCommand } from '../src/ai/policyActions';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { FIELD, cloneState, createInitialState, type GameState, type Team, type Vec2 } from '../src/game/model';
import { AI_HZ, FIXED_DT, PHYSICS_HZ, simulateMatch } from '../src/game/match';
import { stepGame } from '../src/game/simulation';
import { AiClock } from '../src/game/strategy';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type MacroProbeOptions = {
  weightsPath: string;
  seed: number;
  match: number;
  frames: number;
  starts: number[];
  durations: number[];
  actions: number[];
};

type MacroProbeResult = {
  seed: number;
  match: number;
  team: Team;
  baseline: MacroProbeRow;
  rows: MacroProbeRow[];
};

type MacroProbeRow = {
  start?: number;
  duration?: number;
  actionIndex?: number;
  goalsFor: number;
  goalsAgainst: number;
  attackBallX: number;
  ballY: number;
};

export function parseMacroProbeArgs(argv: readonly string[]): MacroProbeOptions {
  return {
    weightsPath: stringArg(argv, '--weights') ?? 'public/models/neural-best.json',
    seed: integerArg(argv, '--seed', 31),
    match: integerArg(argv, '--match', 0),
    frames: positiveIntegerArg(argv, '--frames', 600),
    starts: numberListArg(argv, '--starts', [360, 420, 480, 540]),
    durations: numberListArg(argv, '--durations', [18, 36]),
    actions: numberListArg(argv, '--actions', [...Array(POLICY_ACTION_COUNT).keys()])
      .filter((action) => action >= 0 && action < POLICY_ACTION_COUNT)
  };
}

export function runMacroProbe(options: MacroProbeOptions): MacroProbeResult {
  const weights = loadWeightsPayload(readFileSync(options.weightsPath, 'utf8'));
  const team: Team = options.match % 2 === 0 ? 'red' : 'blue';
  const baseline = runOne(options, team, weights);
  const rows: MacroProbeRow[] = [];
  const startStates = createAlignedStartStates(options, team, weights);

  for (const start of options.starts) {
    for (const duration of options.durations) {
      for (const actionIndex of options.actions) {
        const startState = startStates.get(start);
        rows.push(runOne(options, team, weights, { start, duration, actionIndex }, startState));
      }
    }
  }

  return {
    seed: options.seed,
    match: options.match,
    team,
    baseline,
    rows
  };
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runMacroProbe(parseMacroProbeArgs(argv));
    console.log(formatRow('baseline', result.baseline));
    for (const row of result.rows) {
      console.log(formatRow(`start=${row.start} duration=${row.duration} action=${row.actionIndex}`, row));
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function runOne(
  options: MacroProbeOptions,
  team: Team,
  weights: number[],
  macro?: { start: number; duration: number; actionIndex: number },
  startState?: GameState
): MacroProbeRow {
  const base = createNeuralStrategy({ weights, tacticalRollout: true });
  const strategy = macro
    ? {
        name: 'macro-probe',
        decide(state: Readonly<GameState>, requestedTeam: Team) {
          if (requestedTeam === team && state.frame >= macro.start && state.frame < macro.start + macro.duration) {
            return { [`${team}-0`]: actionIndexToCommand(macro.actionIndex) };
          }
          return base.decide(state, requestedTeam);
        }
      }
    : base;
  const initialState = startState ?? createSeededInitialState(options.seed, options.match, team);
  const frames = startState && macro
    ? Math.max(0, options.frames - macro.start)
    : options.frames;
  const state = simulateMatch({
    red: team === 'red' ? strategy : traditionalStrategy,
    blue: team === 'blue' ? strategy : traditionalStrategy,
    frames,
    initialState
  }).state;

  return {
    start: macro?.start,
    duration: macro?.duration,
    actionIndex: macro?.actionIndex,
    goalsFor: team === 'red' ? state.score.red : state.score.blue,
    goalsAgainst: team === 'red' ? state.score.blue : state.score.red,
    attackBallX: round(attackX(team, state.ball.position.x)),
    ballY: round(state.ball.position.y)
  };
}

function createAlignedStartStates(
  options: MacroProbeOptions,
  team: Team,
  weights: number[]
): Map<number, GameState> {
  const framesPerDecision = Math.max(1, Math.round(PHYSICS_HZ / AI_HZ));
  const starts = [...new Set(options.starts)]
    .filter((start) => start >= 0 && start < options.frames && start % framesPerDecision === 0)
    .sort((a, b) => a - b);
  const states = new Map<number, GameState>();
  if (starts.length === 0) {
    return states;
  }

  const base = createNeuralStrategy({ weights, tacticalRollout: true });
  const state = createSeededInitialState(options.seed, options.match, team);
  const clock = new AiClock(
    team === 'red' ? base : traditionalStrategy,
    team === 'blue' ? base : traditionalStrategy,
    PHYSICS_HZ,
    AI_HZ
  );
  const wanted = new Set(starts);

  for (let frame = 0; frame <= starts[starts.length - 1]; frame += 1) {
    if (wanted.has(state.frame)) {
      states.set(state.frame, cloneState(state));
    }
    if (frame === starts[starts.length - 1]) {
      break;
    }
    stepGame(state, clock.update(state), FIXED_DT);
  }

  return states;
}

function formatRow(label: string, row: MacroProbeRow): string {
  return [
    label,
    `goals=${row.goalsFor}-${row.goalsAgainst}`,
    `ball=(${row.attackBallX},${row.ballY})`
  ].join(' ');
}

function createSeededInitialState(seed: number, match: number, team: Team): GameState {
  const random = createSeededRandom(seed + match * 4099);
  const state = createInitialState();
  const attackFrameX = FIELD.length / 2 + (random() - 0.5) * FIELD.length * 0.12;
  const attackFrameY = FIELD.width / 2 + (random() - 0.5) * FIELD.width * 0.22;

  state.ball.position = fieldPoint(team, attackFrameX, attackFrameY);
  state.ball.velocity = fieldVector(team, (random() - 0.5) * 120, (random() - 0.5) * 120);
  return state;
}

function attackX(team: Team, fieldX: number): number {
  return team === 'red' ? fieldX : FIELD.length - fieldX;
}

function fieldPoint(team: Team, attackFrameX: number, attackFrameY: number): Vec2 {
  return {
    x: team === 'red' ? attackFrameX : FIELD.length - attackFrameX,
    y: team === 'red' ? attackFrameY : FIELD.width - attackFrameY
  };
}

function fieldVector(team: Team, attackFrameX: number, attackFrameY: number): Vec2 {
  return {
    x: team === 'red' ? attackFrameX : -attackFrameX,
    y: team === 'red' ? attackFrameY : -attackFrameY
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

function numberListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = listArgValue(argv, name);
  if (!value) {
    return [...fallback];
  }

  const values = value.split(',')
    .map((part) => Math.floor(Number(part.trim())))
    .filter((seed) => Number.isFinite(seed));
  return values.length > 0 ? values : [...fallback];
}

function listArgValue(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }

  const values: string[] = [];
  for (let cursor = index + 1; cursor < argv.length; cursor += 1) {
    const value = argv[cursor];
    if (value.startsWith('--')) {
      break;
    }
    values.push(value);
  }
  return values.length > 0 ? values.join(',') : undefined;
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  return index === -1 || index === argv.length - 1 ? undefined : argv[index + 1];
}

function integerArg(argv: readonly string[], name: string, fallback: number): number {
  const value = stringArg(argv, name);
  const parsed = value === undefined ? fallback : Math.floor(Number(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, integerArg(argv, name, fallback));
}

function round(value: number): number {
  return Number(value.toFixed(3));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/probe-runtime-macros.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/probe-runtime-macros.js')) {
  main();
}
