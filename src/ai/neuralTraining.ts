import { FIELD, createInitialState, type GameState, type Team } from '../game/model';
import { simulateMatch } from '../game/match';
import type { Strategy } from '../game/strategy';
import { createNeuralStrategy } from './neuralStrategy';
import { NEURAL_WEIGHT_COUNT, defaultNeuralWeights, type NeuralWeights } from './neuralWeights';

export type EvaluationOptions = {
  seed?: number;
  opponent?: Strategy;
  matches?: number;
  frames?: number;
};

export type EvaluationResult = {
  score: number;
  goalDiff: number;
  ballProgress: number;
};

export type TrainingOptions = EvaluationOptions & {
  baseWeights?: NeuralWeights;
  generations?: number;
  population?: number;
  sigma?: number;
};

export type TrainingGeneration = {
  generation: number;
  score: number;
  goalDiff: number;
  ballProgress: number;
};

export type TrainingResult = {
  weights: number[];
  best: EvaluationResult;
  history: TrainingGeneration[];
};

type EvaluationScenario = {
  name: string;
  weight: number;
  create(seed: number, match: number, team: Team): GameState;
  objective(state: GameState, initialState: GameState, team: Team, goalDiff: number): number;
  progress(state: GameState, initialState: GameState, team: Team): number;
};

const IDLE_OPPONENT: Strategy = {
  name: 'idle',
  decide() {
    return {};
  }
};

const EVALUATION_SCENARIOS: EvaluationScenario[] = [
  {
    name: 'kickoff-pressure',
    weight: 0.9,
    create: createKickoffPressureState,
    objective: kickoffObjective,
    progress: openPlayProgress
  },
  {
    name: 'central-finish-high',
    weight: 1.25,
    create(seed, match, team) {
      return createCentralFinishState(seed, match, team, -1);
    },
    objective: centralFinishObjective,
    progress: attackProgressGain
  },
  {
    name: 'central-finish-low',
    weight: 1.25,
    create(seed, match, team) {
      return createCentralFinishState(seed, match, team, 1);
    },
    objective: centralFinishObjective,
    progress: attackProgressGain
  },
  {
    name: 'corner-recycle-high',
    weight: 1.15,
    create(seed, match, team) {
      return createCornerRecycleState(seed, match, team, -1);
    },
    objective: cornerRecycleObjective,
    progress: attackProgressGain
  },
  {
    name: 'corner-recycle-low',
    weight: 1.15,
    create(seed, match, team) {
      return createCornerRecycleState(seed, match, team, 1);
    },
    objective: cornerRecycleObjective,
    progress: attackProgressGain
  },
  {
    name: 'own-goal-defense-high',
    weight: 1.25,
    create(seed, match, team) {
      return createOwnGoalDefenseState(seed, match, team, -1);
    },
    objective: ownGoalDefenseObjective,
    progress: attackProgressGain
  },
  {
    name: 'own-goal-defense-low',
    weight: 1.25,
    create(seed, match, team) {
      return createOwnGoalDefenseState(seed, match, team, 1);
    },
    objective: ownGoalDefenseObjective,
    progress: attackProgressGain
  },
  {
    name: 'duel-contact-high',
    weight: 1,
    create(seed, match, team) {
      return createDuelContactState(seed, match, team, -1);
    },
    objective: duelContactObjective,
    progress: attackProgressGain
  },
  {
    name: 'duel-contact-low',
    weight: 1,
    create(seed, match, team) {
      return createDuelContactState(seed, match, team, 1);
    },
    objective: duelContactObjective,
    progress: attackProgressGain
  }
];

export function evaluateNeuralWeights(
  weights: NeuralWeights,
  options: EvaluationOptions = {}
): EvaluationResult {
  validateWeights(weights);

  const seed = options.seed ?? 1;
  const matches = options.matches ?? 4;
  const frames = options.frames ?? 30 * 20;
  const opponent = options.opponent ?? IDLE_OPPONENT;
  let totalGoalDiff = 0;
  let totalBallProgress = 0;
  let totalScore = 0;
  let totalWeight = 0;

  for (let match = 0; match < matches; match += 1) {
    const team: Team = match % 2 === 0 ? 'red' : 'blue';

    for (let scenarioIndex = 0; scenarioIndex < EVALUATION_SCENARIOS.length; scenarioIndex += 1) {
      const scenario = EVALUATION_SCENARIOS[scenarioIndex];
      const initialState = scenario.create(seed + scenarioIndex * 7919, match, team);
      const neural = createNeuralStrategy({
        weights,
        name: `neural-eval-${team}`,
        tacticalRollout: false
      });
      const result = simulateMatch({
        red: team === 'red' ? neural : opponent,
        blue: team === 'blue' ? neural : opponent,
        frames,
        initialState
      });

      const goalDiff = teamGoalDiff(result.state, team);
      const progress = scenario.progress(result.state, initialState, team);
      const objective = scenario.objective(result.state, initialState, team, goalDiff);

      totalGoalDiff += goalDiff * scenario.weight;
      totalBallProgress += progress * scenario.weight;
      totalScore += (goalDiff * 1000 + objective * 100) * scenario.weight;
      totalWeight += scenario.weight;
    }
  }

  return {
    score: totalScore / totalWeight,
    goalDiff: totalGoalDiff / totalWeight,
    ballProgress: totalBallProgress / totalWeight
  };
}

export function trainNeuralWeights(options: TrainingOptions = {}): TrainingResult {
  const generations = options.generations ?? 8;
  const population = Math.max(2, options.population ?? 10);
  const sigma = options.sigma ?? 0.12;
  const seed = options.seed ?? 1;
  const random = createSeededRandom(seed);
  let bestWeights = [...(options.baseWeights ?? defaultNeuralWeights())];
  let best = evaluateNeuralWeights(bestWeights, options);
  let elites = [{ weights: bestWeights, result: best }];
  const history: TrainingGeneration[] = [
    {
      generation: 0,
      score: best.score,
      goalDiff: best.goalDiff,
      ballProgress: best.ballProgress
    }
  ];

  const eliteCount = Math.max(1, Math.min(5, Math.floor(population / 4)));

  for (let generation = 1; generation <= generations; generation += 1) {
    const progress = generations === 0 ? 0 : (generation - 1) / generations;
    const generationSigma = sigma * (1 - progress * 0.55);
    const candidates = [...elites];

    for (let candidate = candidates.length; candidate < population; candidate += 1) {
      const parent = elites[candidate % elites.length];
      const scale = candidate % 3 === 0 ? 0.55 : candidate % 3 === 1 ? 1 : 1.6;
      const candidateWeights = mutateWeights(parent.weights, generationSigma * scale, random);
      candidates.push({
        weights: candidateWeights,
        result: evaluateNeuralWeights(candidateWeights, options)
      });
    }

    candidates.sort((a, b) => b.result.score - a.result.score);
    elites = candidates.slice(0, eliteCount).map((candidate) => ({
      weights: [...candidate.weights],
      result: candidate.result
    }));

    bestWeights = [...elites[0].weights];
    best = elites[0].result;
    history.push({
      generation,
      score: best.score,
      goalDiff: best.goalDiff,
      ballProgress: best.ballProgress
    });
  }

  return {
    weights: bestWeights,
    best,
    history
  };
}

export function mutateWeights(
  weights: NeuralWeights,
  sigma: number,
  random: () => number = createSeededRandom(1)
): number[] {
  validateWeights(weights);
  return weights.map((weight) => clampWeight(weight + gaussian(random) * sigma));
}

export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function createKickoffPressureState(seed: number, match: number, team: Team): GameState {
  const random = createSeededRandom(seed + match * 9973);
  const state = createInitialState();
  const yJitter = (random() - 0.5) * FIELD.width * 0.2;

  placeBall(
    state,
    team,
    FIELD.length / 2 + (random() - 0.5) * FIELD.length * 0.08,
    FIELD.width / 2 + yJitter,
    (random() - 0.5) * 85,
    (random() - 0.5) * 85
  );
  placeTank(state, team, 0, 170 + random() * 20, FIELD.width / 2 - yJitter * 0.22, 0);
  placeOpponentInTeamFrame(state, team, FIELD.length - 170 - random() * 20, FIELD.width / 2 + yJitter * 0.22, Math.PI);
  return state;
}

function createCentralFinishState(seed: number, match: number, team: Team, verticalSide: 1 | -1): GameState {
  const random = createSeededRandom(seed + match * 3571);
  const state = createInitialState();
  const laneJitter = (random() - 0.5) * 18;
  const ballY = FIELD.width / 2 + verticalSide * (42 + laneJitter);
  const ballX = FIELD.length - 228 - random() * 36;

  placeBall(state, team, ballX, ballY, 0, 0);
  placeTank(state, team, 0, ballX - 178, ballY + verticalSide * (18 + random() * 10), 0);
  placeOpponentInTeamFrame(state, team, FIELD.length - 140, FIELD.width / 2 - verticalSide * 170, Math.PI);
  return state;
}

function createCornerRecycleState(seed: number, match: number, team: Team, verticalSide: 1 | -1): GameState {
  const random = createSeededRandom(seed + match * 4447);
  const state = createInitialState();
  const wallY =
    verticalSide < 0
      ? FIELD.ballRadius + 14 + random() * 14
      : FIELD.width - FIELD.ballRadius - 14 - random() * 14;
  const ballX = FIELD.length - FIELD.ballRadius - 36 - random() * 24;

  placeBall(state, team, ballX, wallY, 0, 0);
  placeTank(state, team, 0, ballX - 168, wallY - verticalSide * (106 + random() * 16), 0);
  placeOpponentInTeamFrame(state, team, ballX - 62, wallY - verticalSide * 64, Math.PI);
  return state;
}

function createOwnGoalDefenseState(seed: number, match: number, team: Team, verticalSide: 1 | -1): GameState {
  const random = createSeededRandom(seed + match * 6421);
  const state = createInitialState();
  const ballY = FIELD.width / 2 + verticalSide * (50 + random() * 26);
  const incomingSpeed = -120 - random() * 55;
  const ballX = 224 + random() * 34;

  placeBall(state, team, ballX, ballY, incomingSpeed, verticalSide * (random() * 22));
  placeTank(state, team, 0, 108 + random() * 26, FIELD.width / 2 + verticalSide * 26, 0);
  placeOpponentInTeamFrame(state, team, FIELD.length - 172, FIELD.width / 2 - verticalSide * 142, Math.PI);
  return state;
}

function createDuelContactState(seed: number, match: number, team: Team, verticalSide: 1 | -1): GameState {
  const random = createSeededRandom(seed + match * 5209);
  const state = createInitialState();
  const ballY = FIELD.width / 2 + verticalSide * (36 + random() * 28);
  const ballX = FIELD.length / 2 + 72 + random() * 36;

  placeBall(state, team, ballX, ballY, 10 + random() * 20, verticalSide * (random() * 16));
  placeTank(state, team, 0, ballX - 160, ballY + verticalSide * 44, 0.08 * verticalSide);
  placeOpponentInTeamFrame(state, team, ballX + 96, ballY - verticalSide * 52, Math.PI - 0.16 * verticalSide);
  return state;
}

function kickoffObjective(state: GameState, initialState: GameState, team: Team, goalDiff: number): number {
  const progress = openPlayProgress(state, initialState, team);
  const ballVelocity = attackVelocity(state, team) / 1000;
  const nearestDistance = nearestTeamDistanceToBall(state, team) / FIELD.length;
  return goalDiff * 0.35 + progress + ballVelocity - nearestDistance * 0.55;
}

function centralFinishObjective(state: GameState, initialState: GameState, team: Team, goalDiff: number): number {
  const progress = attackProgressGain(state, initialState, team);
  const laneScore = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth / 2));
  const velocity = attackVelocity(state, team) / 760;
  return goalDiff * 1.6 + progress * 1.35 + laneScore * 0.55 + velocity * 0.32;
}

function cornerRecycleObjective(state: GameState, initialState: GameState, team: Team, goalDiff: number): number {
  const progress = attackProgressGain(state, initialState, team);
  const centerGain =
    (Math.abs(initialState.ball.position.y - FIELD.width / 2) -
      Math.abs(state.ball.position.y - FIELD.width / 2)) /
    (FIELD.width / 2);
  const wallClear =
    (sideWallDistance(state.ball.position.y) - sideWallDistance(initialState.ball.position.y)) /
    (FIELD.width / 2);
  const stuckPenalty = sideWallDistance(state.ball.position.y) < FIELD.ballRadius * 1.35 ? 0.28 : 0;

  return goalDiff + progress * 0.55 + centerGain * 0.85 + wallClear * 0.75 +
    attackVelocity(state, team) / 940 - stuckPenalty;
}

function ownGoalDefenseObjective(state: GameState, initialState: GameState, team: Team, goalDiff: number): number {
  const clearGain = attackProgressGain(state, initialState, team);
  const ownDistance = attackX(team, state.ball.position.x);
  const centralDanger = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  const dangerPenalty = ownDistance < 220 ? centralDanger * 0.9 : 0;

  return goalDiff * 1.7 + clearGain * 1.25 + attackVelocity(state, team) / 880 - dangerPenalty;
}

function duelContactObjective(state: GameState, initialState: GameState, team: Team, goalDiff: number): number {
  const progress = attackProgressGain(state, initialState, team);
  const velocity = attackVelocity(state, team) / 860;
  const nearestDistance = nearestTeamDistanceToBall(state, team) / FIELD.length;

  return goalDiff * 1.2 + progress * 1.1 + velocity * 0.45 - nearestDistance * 0.25;
}

function openPlayProgress(state: GameState, _initialState: GameState, team: Team): number {
  return (attackX(team, state.ball.position.x) - FIELD.length / 2) / FIELD.length;
}

function attackProgressGain(state: GameState, initialState: GameState, team: Team): number {
  return (attackX(team, state.ball.position.x) - attackX(team, initialState.ball.position.x)) / FIELD.length;
}

function teamGoalDiff(state: GameState, team: Team): number {
  return team === 'red'
    ? state.score.red - state.score.blue
    : state.score.blue - state.score.red;
}

function attackVelocity(state: GameState, team: Team): number {
  const sign = team === 'red' ? 1 : -1;
  return state.ball.velocity.x * sign;
}

function attackX(team: Team, fieldX: number): number {
  return team === 'red' ? fieldX : FIELD.length - fieldX;
}

function placeBall(
  state: GameState,
  team: Team,
  attackFrameX: number,
  attackFrameY: number,
  attackFrameVelocityX: number,
  attackFrameVelocityY: number
): void {
  state.ball.position = fieldPoint(team, attackFrameX, attackFrameY);
  state.ball.velocity = fieldVector(team, attackFrameVelocityX, attackFrameVelocityY);
}

function placeTank(
  state: GameState,
  team: Team,
  index: number,
  attackFrameX: number,
  attackFrameY: number,
  attackFrameAngle: number
): void {
  const tank = state.tanks.find((candidate) => candidate.team === team && candidate.index === index);
  if (!tank) {
    throw new Error(`Missing ${team} tank ${index}`);
  }
  tank.position = fieldPoint(team, attackFrameX, attackFrameY);
  tank.velocity = { x: 0, y: 0 };
  tank.angle = fieldAngle(team, attackFrameAngle);
  tank.angularVelocity = 0;
  tank.stamina = tank.maxStamina;
}

function placeOpponentInTeamFrame(
  state: GameState,
  team: Team,
  attackFrameX: number,
  attackFrameY: number,
  attackFrameAngle: number
): void {
  const opponent = team === 'red' ? 'blue' : 'red';
  const tank = state.tanks.find((candidate) => candidate.team === opponent && candidate.index === 0);
  if (!tank) {
    throw new Error(`Missing ${opponent} tank 0`);
  }
  tank.position = fieldPoint(team, attackFrameX, attackFrameY);
  tank.velocity = { x: 0, y: 0 };
  tank.angle = fieldAngle(team, attackFrameAngle);
  tank.angularVelocity = 0;
  tank.stamina = tank.maxStamina;
}

function fieldPoint(team: Team, attackFrameX: number, attackFrameY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackFrameX : FIELD.length - attackFrameX,
    y: team === 'red' ? attackFrameY : FIELD.width - attackFrameY
  };
}

function fieldVector(team: Team, attackFrameX: number, attackFrameY: number): { x: number; y: number } {
  return {
    x: team === 'red' ? attackFrameX : -attackFrameX,
    y: team === 'red' ? attackFrameY : -attackFrameY
  };
}

function fieldAngle(team: Team, attackFrameAngle: number): number {
  return normalizeAngle(team === 'red' ? attackFrameAngle : attackFrameAngle + Math.PI);
}

function nearestTeamDistanceToBall(state: GameState, team: Team): number {
  let best = Number.POSITIVE_INFINITY;
  for (const tank of state.tanks) {
    if (tank.team !== team) {
      continue;
    }
    best = Math.min(
      best,
      Math.hypot(tank.position.x - state.ball.position.x, tank.position.y - state.ball.position.y)
    );
  }
  return best;
}

function sideWallDistance(y: number): number {
  return Math.min(y - FIELD.ballRadius, FIELD.width - FIELD.ballRadius - y);
}

function gaussian(random: () => number): number {
  const u1 = Math.max(Number.EPSILON, random());
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(Math.PI * 2 * u2);
}

function clampWeight(value: number): number {
  return Math.max(-3, Math.min(3, value));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function normalizeAngle(angle: number): number {
  let normalized = angle;
  while (normalized <= -Math.PI) {
    normalized += Math.PI * 2;
  }
  while (normalized > Math.PI) {
    normalized -= Math.PI * 2;
  }
  return normalized;
}

function validateWeights(weights: NeuralWeights): void {
  if (weights.length !== NEURAL_WEIGHT_COUNT) {
    throw new Error(`Expected ${NEURAL_WEIGHT_COUNT} neural weights, received ${weights.length}`);
  }
}
