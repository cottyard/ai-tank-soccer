import { createNeuralStrategy } from '../src/ai/neuralStrategy';
import { evaluateNeuralWeights, trainNeuralWeights } from '../src/ai/neuralTraining';
import { ZERO_NEURAL_WEIGHTS, defaultNeuralWeights } from '../src/ai/neuralWeights';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { idleCommands, type Strategy } from '../src/game/strategy';

declare const process: {
  argv: string[];
  exitCode?: number;
};

const idle: Strategy = {
  name: 'idle',
  decide: idleCommands
};
const opponents = {
  idle,
  traditional: traditionalStrategy
} as const;

const args = parseArgs(process.argv.slice(2));
const opponent = opponents[args.opponent];
const baseWeights = defaultNeuralWeights();
const zero = evaluateNeuralWeights(ZERO_NEURAL_WEIGHTS, {
  seed: args.seed,
  opponent,
  matches: args.matches,
  frames: args.frames
});
const baseline = evaluateNeuralWeights(baseWeights, {
  seed: args.seed,
  opponent,
  matches: args.matches,
  frames: args.frames
});
const result = trainNeuralWeights({
  baseWeights,
  seed: args.seed,
  opponent,
  generations: args.generations,
  population: args.population,
  sigma: args.sigma,
  matches: args.matches,
  frames: args.frames
});

const trainedStrategy = createNeuralStrategy({
  weights: result.weights,
  name: 'neural-trained'
});

console.log(`strategy=${trainedStrategy.name}`);
console.log(`seed=${args.seed}`);
console.log(`opponent=${args.opponent}`);
console.log(`zero=${zero.score.toFixed(3)}`);
console.log(`default=${baseline.score.toFixed(3)}`);
for (const generation of result.history) {
  console.log(
    `generation=${generation.generation} score=${generation.score.toFixed(3)} ` +
      `goalDiff=${generation.goalDiff.toFixed(3)} ballProgress=${generation.ballProgress.toFixed(3)}`
  );
}
console.log(`best=${result.best.score.toFixed(3)}`);
for (const [name, evalOpponent] of Object.entries(opponents)) {
  const before = evaluateNeuralWeights(baseWeights, {
    seed: args.seed,
    opponent: evalOpponent,
    matches: args.matches,
    frames: args.frames
  });
  const after = evaluateNeuralWeights(result.weights, {
    seed: args.seed,
    opponent: evalOpponent,
    matches: args.matches,
    frames: args.frames
  });
  console.log(
    `eval=${name} default=${before.score.toFixed(3)} trained=${after.score.toFixed(3)} ` +
      `goalDiff=${after.goalDiff.toFixed(3)} ballProgress=${after.ballProgress.toFixed(3)}`
  );
}
if (args.printWeights) {
  console.log(`weights=${JSON.stringify(result.weights)}`);
}

function parseArgs(argv: string[]): {
  seed: number;
  opponent: keyof typeof opponents;
  generations: number;
  population: number;
  sigma: number;
  matches: number;
  frames: number;
  printWeights: boolean;
} {
  return {
    seed: numberArg(argv, '--seed', 1),
    opponent: opponentArg(argv, '--opponent', 'traditional'),
    generations: numberArg(argv, '--generations', 8),
    population: numberArg(argv, '--population', 10),
    sigma: numberArg(argv, '--sigma', 0.12),
    matches: numberArg(argv, '--matches', 4),
    frames: numberArg(argv, '--frames', 30 * 20),
    printWeights: booleanArg(argv, '--print-weights')
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

function opponentArg(
  argv: string[],
  name: string,
  fallback: keyof typeof opponents
): keyof typeof opponents {
  const index = argv.indexOf(name);
  if (index === -1 || index === argv.length - 1) {
    return fallback;
  }

  const parsed = argv[index + 1];
  return parsed === 'idle' || parsed === 'traditional' ? parsed : fallback;
}

function booleanArg(argv: string[], name: string): boolean {
  return argv.includes(name);
}
