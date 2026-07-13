import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';
import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { POLICY_ACTION_COUNT, actionIndexToCommand } from '../src/ai/policyActions';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { AI_HZ, FIXED_DT, PHYSICS_HZ, simulateMatch } from '../src/game/match';
import { FIELD, cloneState, createInitialState, type GameState, type Team, type Vec2 } from '../src/game/model';
import { stepGame } from '../src/game/simulation';
import { AiClock } from '../src/game/strategy';

declare const process: {
  argv: string[];
  exitCode?: number;
};

export type SequenceProbeOptions = {
  weightsPath: string;
  seed: number;
  match: number;
  frames: number;
  starts: number[];
  firstDurations: number[];
  secondDurations: number[];
  firstActions: number[];
  secondActions: number[];
  limit: number;
  maxCombinations: number;
};

export type SequenceProbeRow = {
  start?: number;
  firstDuration?: number;
  secondDuration?: number;
  firstActionIndex?: number;
  secondActionIndex?: number;
  goalsFor: number;
  goalsAgainst: number;
  attackBallX: number;
  ballY: number;
  ballSpeed: number;
  lane: number;
  score: number;
};

export function parseSequenceProbeArgs(argv: readonly string[]): SequenceProbeOptions {
  return {
    weightsPath: stringArg(argv, '--weights') ?? 'public/models/neural-best.json',
    seed: integerArg(argv, '--seed', 71),
    match: integerArg(argv, '--match', 3),
    frames: positiveIntegerArg(argv, '--frames', 600),
    starts: numberListArg(argv, '--starts', [504, 516, 528, 540]),
    firstDurations: numberListArg(argv, '--first-durations', [6, 12, 18, 24]),
    secondDurations: numberListArg(argv, '--second-durations', [18, 36, 54]),
    firstActions: actionListArg(argv, '--first-actions'),
    secondActions: actionListArg(argv, '--second-actions'),
    limit: positiveIntegerArg(argv, '--limit', 40),
    maxCombinations: positiveIntegerArg(argv, '--max-combinations', Number.POSITIVE_INFINITY)
  };
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runSequenceProbe(parseSequenceProbeArgs(argv));
    console.log(formatRow('baseline', result.baseline));
    if (result.truncated) {
      console.log(`partial completed=${result.completedRows}/${result.plannedRows}`);
    }
    for (const row of result.rows) {
      console.log(formatRow(
        `start=${row.start} first=${row.firstDuration}:${row.firstActionIndex} second=${row.secondDuration}:${row.secondActionIndex}`,
        row
      ));
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

export function runSequenceProbe(options: SequenceProbeOptions): {
  baseline: SequenceProbeRow;
  rows: SequenceProbeRow[];
  plannedRows: number;
  completedRows: number;
  truncated: boolean;
} {
  const weights = loadWeightsPayload(readFileSync(options.weightsPath, 'utf8'));
  const team: Team = options.match % 2 === 0 ? 'red' : 'blue';
  const baseline = runOne(options, team, weights);
  const startStates = createAlignedStartStates(options, team, weights);
  const rows: SequenceProbeRow[] = [];
  const plannedRows =
    options.starts.filter((start) => startStates.has(start)).length *
    options.firstDurations.length *
    options.secondDurations.length *
    options.firstActions.length *
    options.secondActions.length;
  let completedRows = 0;
  let truncated = false;

  sequenceRows:
  for (const start of options.starts) {
    const startState = startStates.get(start);
    if (!startState) {
      continue;
    }
    for (const firstDuration of options.firstDurations) {
      for (const secondDuration of options.secondDurations) {
        for (const firstActionIndex of options.firstActions) {
          for (const secondActionIndex of options.secondActions) {
            if (completedRows >= options.maxCombinations) {
              truncated = true;
              break sequenceRows;
            }
            rows.push(runOne(options, team, weights, {
              start,
              firstDuration,
              secondDuration,
              firstActionIndex,
              secondActionIndex
            }, startState));
            completedRows += 1;
          }
        }
      }
    }
  }

  rows.sort((left, right) => right.score - left.score);
  return {
    baseline,
    rows: rows.slice(0, options.limit),
    plannedRows,
    completedRows,
    truncated
  };
}

function runOne(
  options: SequenceProbeOptions,
  team: Team,
  weights: number[],
  macro?: {
    start: number;
    firstDuration: number;
    secondDuration: number;
    firstActionIndex: number;
    secondActionIndex: number;
  },
  startState?: GameState
): SequenceProbeRow {
  const base = createNeuralStrategy({ weights, tacticalRollout: true });
  const strategy = macro
    ? {
        name: 'sequence-probe',
        decide(state: Readonly<GameState>, requestedTeam: Team) {
          if (requestedTeam === team) {
            if (state.frame >= macro.start && state.frame < macro.start + macro.firstDuration) {
              return { [`${team}-0`]: actionIndexToCommand(macro.firstActionIndex) };
            }
            if (
              state.frame >= macro.start + macro.firstDuration &&
              state.frame < macro.start + macro.firstDuration + macro.secondDuration
            ) {
              return { [`${team}-0`]: actionIndexToCommand(macro.secondActionIndex) };
            }
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
  const attackBallX = attackX(team, state.ball.position.x);
  const lane = goalLaneScore(state.ball.position.y);
  const goalsFor = team === 'red' ? state.score.red : state.score.blue;
  const goalsAgainst = team === 'red' ? state.score.blue : state.score.red;

  return {
    start: macro?.start,
    firstDuration: macro?.firstDuration,
    secondDuration: macro?.secondDuration,
    firstActionIndex: macro?.firstActionIndex,
    secondActionIndex: macro?.secondActionIndex,
    goalsFor,
    goalsAgainst,
    attackBallX: round(attackBallX),
    ballY: round(state.ball.position.y),
    ballSpeed: round(Math.hypot(state.ball.velocity.x, state.ball.velocity.y)),
    lane: round(lane),
    score: round((goalsFor - goalsAgainst) * 10000 + attackBallX + lane * 260 - Math.abs(state.ball.position.y - FIELD.width / 2) * 0.25)
  };
}

function createAlignedStartStates(
  options: SequenceProbeOptions,
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

function formatRow(label: string, row: SequenceProbeRow): string {
  return [
    label,
    `goals=${row.goalsFor}-${row.goalsAgainst}`,
    `ball=(${row.attackBallX},${row.ballY})`,
    `speed=${row.ballSpeed}`,
    `lane=${row.lane}`,
    `score=${row.score}`
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

function actionListArg(argv: readonly string[], name: string): number[] {
  return numberListArg(argv, name, [...Array(POLICY_ACTION_COUNT).keys()])
    .filter((action) => action >= 0 && action < POLICY_ACTION_COUNT);
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

function attackX(team: Team, fieldX: number): number {
  return team === 'red' ? fieldX : FIELD.length - fieldX;
}

function goalLaneScore(y: number): number {
  return 1 - clamp01(Math.abs(y - FIELD.width / 2) / (FIELD.goalMouth * 0.72));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function round(value: number): number {
  return Number(value.toFixed(3));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/probe-runtime-sequences.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/probe-runtime-sequences.js')) {
  main();
}
