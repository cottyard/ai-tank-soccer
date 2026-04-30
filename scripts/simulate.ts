import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { FIELD, createInitialState } from '../src/game/model';
import { simulateMatch } from '../src/game/match';
import type { Strategy } from '../src/game/strategy';
import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
};

type ResultRow = {
  red: string;
  blue: string;
  redGoals: number;
  blueGoals: number;
  ballX: number;
};

const args = parseArgs(process.argv.slice(2));
const neural = createNeuralStrategy({
  name: 'neural',
  weights: args.input ? loadWeightsPayload(readFileSync(args.input, 'utf8')) : undefined
});
const pairings: Array<[Strategy, Strategy]> = [
  [traditionalStrategy, neural],
  [neural, traditionalStrategy]
];

const rows: ResultRow[] = [];
for (let i = 0; i < args.matches; i += 1) {
  const [red, blue] = pairings[i % pairings.length];
  const result = simulateMatch({
    red,
    blue,
    frames: args.frames,
    initialState: createSeededInitialState(args.seed, i)
  });
  rows.push({
    red: red.name,
    blue: blue.name,
    redGoals: result.state.score.red,
    blueGoals: result.state.score.blue,
    ballX: result.state.ball.position.x
  });
}

let traditionalGoals = 0;
let neuralGoals = 0;
for (const row of rows) {
  const redTraditional = row.red === traditionalStrategy.name;
  traditionalGoals += redTraditional ? row.redGoals : row.blueGoals;
  neuralGoals += redTraditional ? row.blueGoals : row.redGoals;
}

console.log(`matches=${args.matches}`);
console.log(`frames=${args.frames}`);
console.log(`seed=${args.seed}`);
if (args.input) {
  console.log(`input=${args.input}`);
}
console.log(`traditionalGoals=${traditionalGoals}`);
console.log(`neuralGoals=${neuralGoals}`);
for (const [index, row] of rows.entries()) {
  console.log(
    `match=${index + 1} red=${row.red} blue=${row.blue} score=${row.redGoals}-${row.blueGoals} ballX=${row.ballX.toFixed(1)}`
  );
}

function parseArgs(argv: string[]): { matches: number; frames: number; seed: number; input?: string } {
  return {
    matches: numberArg(argv, '--matches', 4),
    frames: numberArg(argv, '--frames', 30 * 45),
    seed: numberArg(argv, '--seed', 1),
    input: stringArg(argv, '--input')
  };
}

function numberArg(argv: string[], name: string, fallback: number): number {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return fallback;
  }

  const parsed = Number(argv[index + 1]);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function stringArg(argv: string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return undefined;
  }
  return argv[index + 1];
}

function createSeededInitialState(seed: number, match: number) {
  const random = createSeededRandom(seed + match * 4099);
  const state = createInitialState();
  state.ball.position.x += (random() - 0.5) * FIELD.length * 0.12;
  state.ball.position.y += (random() - 0.5) * FIELD.width * 0.22;
  state.ball.velocity.x = (random() - 0.5) * 120;
  state.ball.velocity.y = (random() - 0.5) * 120;
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
