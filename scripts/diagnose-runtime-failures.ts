import { dirname } from 'node:path';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createNeuralStrategy, type NeuralDecisionTrace } from '../src/ai/neuralStrategy';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { FIELD, createInitialState, type GameState, type Team, type Vec2 } from '../src/game/model';
import { simulateMatch } from '../src/game/match';
import { loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
  exitCode?: number;
};

export type RuntimeFailureDiagnosticOptions = {
  weightsPath: string;
  outputPath?: string;
  seeds: number[];
  matches: number;
  frames: number;
  includeWins: boolean;
  tailDecisions: number;
};

export type RuntimeFailureDiagnosticResult = {
  generatedAt: string;
  options: RuntimeFailureDiagnosticOptions;
  summary: {
    matches: number;
    wins: number;
    draws: number;
    losses: number;
    goalsFor: number;
    goalsAgainst: number;
  };
  matches: RuntimeMatchDiagnostic[];
};

export type RuntimeMatchDiagnostic = {
  seed: number;
  match: number;
  team: Team;
  outcome: 'win' | 'draw' | 'loss';
  goalsFor: number;
  goalsAgainst: number;
  ballProgress: number;
  initialBall: RoundedPoint;
  finalBall: RoundedPoint;
  finalBallVelocity: RoundedPoint;
  attackBallX: number;
  sideWallDistance: number;
  finalBallSpeed: number;
  controlledTank: TankSnapshot;
  opponentTank: TankSnapshot;
  allDecisions: DecisionWindowSummary;
  tailDecisions: DecisionWindowSummary;
  tailActions: TailActionSnapshot[];
};

type RoundedPoint = {
  x: number;
  y: number;
};

type TankSnapshot = {
  position: RoundedPoint;
  angle: number;
  staminaRatio: number;
  ballDistance: number;
};

type DecisionWindowSummary = {
  decisions: number;
  rawActionCounts: number[];
  tacticalActionCounts: number[];
  finalActionCounts: number[];
  tacticalRolloutUses: number;
  tacticalRolloutChanges: number;
  staminaConserves: number;
  criticalStaminaRegulations: number;
  averageStamina: number;
  averageBallDistance: number;
  averageBallSpeed: number;
  averageFinishingPressure: number;
  averageOwnGoalPressure: number;
  averageSideWallPressure: number;
  averageAttackCornerPressure: number;
  averageOwnCornerPressure: number;
};

type TailActionSnapshot = {
  frame: number;
  rawActionIndex?: number;
  tacticalActionIndex?: number;
  finalActionIndex: number;
  tacticalRolloutChanged: boolean;
  staminaConserved: boolean;
  criticalStaminaRegulated: boolean;
  staminaRatio: number;
  ballDistance: number;
  ballSpeed: number;
  finishingPressure: number;
  ownGoalPressure: number;
  attackCornerPressure: number;
  ownCornerPressure: number;
};

const DEFAULT_STANDARD_SEEDS = [19, 31, 43, 57, 71];
const ACTION_COUNT = 9;

export function parseRuntimeFailureDiagnosticArgs(argv: readonly string[]): RuntimeFailureDiagnosticOptions {
  return {
    weightsPath: stringArg(argv, '--weights') ?? 'public/models/neural-best.json',
    outputPath: stringArg(argv, '--output'),
    seeds: seedListArg(argv, '--seeds', DEFAULT_STANDARD_SEEDS),
    matches: positiveIntegerArg(argv, '--matches', 4),
    frames: positiveIntegerArg(argv, '--frames', 600),
    includeWins: argv.includes('--include-wins'),
    tailDecisions: positiveIntegerArg(argv, '--tail-decisions', 20)
  };
}

export function runRuntimeFailureDiagnostics(
  options: RuntimeFailureDiagnosticOptions
): RuntimeFailureDiagnosticResult {
  const weights = loadWeightsPayload(readFileSync(options.weightsPath, 'utf8'));
  const matches: RuntimeMatchDiagnostic[] = [];
  const summary = {
    matches: 0,
    wins: 0,
    draws: 0,
    losses: 0,
    goalsFor: 0,
    goalsAgainst: 0
  };

  for (const seed of options.seeds) {
    for (let match = 0; match < options.matches; match += 1) {
      const team: Team = match % 2 === 0 ? 'red' : 'blue';
      const traces: NeuralDecisionTrace[] = [];
      const neural = createNeuralStrategy({
        weights,
        name: 'runtime-failure-diagnostic',
        tacticalRollout: true,
        onDecision: (trace) => traces.push(trace)
      });
      const initialState = createSeededInitialState(seed, match, team);
      const result = simulateMatch({
        red: team === 'red' ? neural : traditionalStrategy,
        blue: team === 'blue' ? neural : traditionalStrategy,
        frames: options.frames,
        initialState
      }).state;
      const goalsFor = team === 'red' ? result.score.red : result.score.blue;
      const goalsAgainst = team === 'red' ? result.score.blue : result.score.red;
      const outcome = goalsFor > goalsAgainst ? 'win' : goalsFor === goalsAgainst ? 'draw' : 'loss';

      summary.matches += 1;
      summary.wins += outcome === 'win' ? 1 : 0;
      summary.draws += outcome === 'draw' ? 1 : 0;
      summary.losses += outcome === 'loss' ? 1 : 0;
      summary.goalsFor += goalsFor;
      summary.goalsAgainst += goalsAgainst;

      if (outcome === 'win' && !options.includeWins) {
        continue;
      }

      matches.push(matchDiagnostic({
        seed,
        match,
        team,
        initialState,
        result,
        goalsFor,
        goalsAgainst,
        outcome,
        traces,
        tailDecisions: options.tailDecisions
      }));
    }
  }

  const diagnostics = {
    generatedAt: new Date().toISOString(),
    options,
    summary,
    matches
  };

  if (options.outputPath) {
    ensureParentDirectory(options.outputPath);
    writeFileSync(options.outputPath, `${JSON.stringify(diagnostics, null, 2)}\n`, 'utf8');
  }

  return diagnostics;
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const result = runRuntimeFailureDiagnostics(parseRuntimeFailureDiagnosticArgs(argv));
    console.log(formatSummary(result));
    for (const row of result.matches) {
      console.log(formatMatch(row));
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function matchDiagnostic(options: {
  seed: number;
  match: number;
  team: Team;
  initialState: GameState;
  result: GameState;
  goalsFor: number;
  goalsAgainst: number;
  outcome: RuntimeMatchDiagnostic['outcome'];
  traces: NeuralDecisionTrace[];
  tailDecisions: number;
}): RuntimeMatchDiagnostic {
  const tail = options.traces.slice(-options.tailDecisions);
  const controlled = tankSnapshot(options.result, options.team);
  const opponent = tankSnapshot(options.result, options.team === 'red' ? 'blue' : 'red');
  const finalBallSpeed = Math.hypot(options.result.ball.velocity.x, options.result.ball.velocity.y);

  return {
    seed: options.seed,
    match: options.match,
    team: options.team,
    outcome: options.outcome,
    goalsFor: options.goalsFor,
    goalsAgainst: options.goalsAgainst,
    ballProgress: round(
      (attackX(options.team, options.result.ball.position.x) -
        attackX(options.team, options.initialState.ball.position.x)) / FIELD.length
    ),
    initialBall: roundPoint(options.initialState.ball.position),
    finalBall: roundPoint(options.result.ball.position),
    finalBallVelocity: roundPoint(options.result.ball.velocity),
    attackBallX: round(attackX(options.team, options.result.ball.position.x)),
    sideWallDistance: round(sideWallDistance(options.result.ball.position.y)),
    finalBallSpeed: round(finalBallSpeed),
    controlledTank: controlled,
    opponentTank: opponent,
    allDecisions: summarizeDecisions(options.traces),
    tailDecisions: summarizeDecisions(tail),
    tailActions: tail.map((trace) => ({
      frame: trace.frame,
      rawActionIndex: trace.rawPolicyActionIndex,
      tacticalActionIndex: trace.tacticalActionIndex,
      finalActionIndex: trace.finalActionIndex,
      tacticalRolloutChanged: trace.tacticalRolloutChanged,
      staminaConserved: trace.staminaConserved,
      criticalStaminaRegulated: trace.criticalStaminaRegulated,
      staminaRatio: round(trace.staminaRatio),
      ballDistance: round(trace.ballDistance),
      ballSpeed: round(trace.ballSpeed),
      finishingPressure: round(trace.finishingPressure),
      ownGoalPressure: round(trace.ownGoalPressure),
      attackCornerPressure: round(trace.attackCornerPressure),
      ownCornerPressure: round(trace.ownCornerPressure)
    }))
  };
}

function summarizeDecisions(records: readonly NeuralDecisionTrace[]): DecisionWindowSummary {
  return {
    decisions: records.length,
    rawActionCounts: actionCounts(records, (trace) => trace.rawPolicyActionIndex),
    tacticalActionCounts: actionCounts(records, (trace) => trace.tacticalActionIndex),
    finalActionCounts: actionCounts(records, (trace) => trace.finalActionIndex),
    tacticalRolloutUses: records.filter((trace) => trace.tacticalRolloutUsed).length,
    tacticalRolloutChanges: records.filter((trace) => trace.tacticalRolloutChanged).length,
    staminaConserves: records.filter((trace) => trace.staminaConserved).length,
    criticalStaminaRegulations: records.filter((trace) => trace.criticalStaminaRegulated).length,
    averageStamina: average(records.map((trace) => trace.staminaRatio)),
    averageBallDistance: average(records.map((trace) => trace.ballDistance)),
    averageBallSpeed: average(records.map((trace) => trace.ballSpeed)),
    averageFinishingPressure: average(records.map((trace) => trace.finishingPressure)),
    averageOwnGoalPressure: average(records.map((trace) => trace.ownGoalPressure)),
    averageSideWallPressure: average(records.map((trace) => trace.sideWallPressure)),
    averageAttackCornerPressure: average(records.map((trace) => trace.attackCornerPressure)),
    averageOwnCornerPressure: average(records.map((trace) => trace.ownCornerPressure))
  };
}

function actionCounts(
  records: readonly NeuralDecisionTrace[],
  pick: (trace: NeuralDecisionTrace) => number | undefined
): number[] {
  const counts = Array.from({ length: ACTION_COUNT }, () => 0);
  for (const record of records) {
    const value = pick(record);
    if (value !== undefined && value >= 0 && value < counts.length) {
      counts[value] += 1;
    }
  }
  return counts;
}

function tankSnapshot(state: GameState, team: Team): TankSnapshot {
  const tank = state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
  if (!tank) {
    throw new Error(`missing ${team} tank`);
  }

  return {
    position: roundPoint(tank.position),
    angle: round(tank.angle),
    staminaRatio: round(tank.maxStamina > 0 ? tank.stamina / tank.maxStamina : 0),
    ballDistance: round(Math.hypot(tank.position.x - state.ball.position.x, tank.position.y - state.ball.position.y))
  };
}

function formatSummary(result: RuntimeFailureDiagnosticResult): string {
  return [
    'diagnostic:',
    `seeds=${result.options.seeds.join(',')}`,
    `matches=${result.summary.matches}`,
    `goals=${result.summary.goalsFor}-${result.summary.goalsAgainst}`,
    `wins=${result.summary.wins}`,
    `draws=${result.summary.draws}`,
    `losses=${result.summary.losses}`,
    `reported=${result.matches.length}`
  ].join(' ');
}

function formatMatch(row: RuntimeMatchDiagnostic): string {
  const tail = row.tailDecisions;
  return [
    `seed=${row.seed}`,
    `match=${row.match}`,
    `team=${row.team}`,
    `outcome=${row.outcome}`,
    `goals=${row.goalsFor}-${row.goalsAgainst}`,
    `ball=(${row.finalBall.x},${row.finalBall.y})`,
    `attackX=${row.attackBallX}`,
    `side=${row.sideWallDistance}`,
    `speed=${row.finalBallSpeed}`,
    `tailPressure=finish:${tail.averageFinishingPressure},own:${tail.averageOwnGoalPressure},attackCorner:${tail.averageAttackCornerPressure},ownCorner:${tail.averageOwnCornerPressure}`,
    `tailFinal=${tail.finalActionCounts.join(',')}`,
    `tailRolloutChanges=${tail.tacticalRolloutChanges}`,
    `tailStaminaStops=${tail.staminaConserves}`,
    `tailCriticalRegs=${tail.criticalStaminaRegulations}`
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

function sideWallDistance(y: number): number {
  return Math.min(y - FIELD.ballRadius, FIELD.width - FIELD.ballRadius - y);
}

function average(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }
  return round(values.reduce((total, value) => total + value, 0) / values.length);
}

function round(value: number): number {
  return Number(value.toFixed(3));
}

function roundPoint(point: Vec2): RoundedPoint {
  return {
    x: round(point.x),
    y: round(point.y)
  };
}

function ensureParentDirectory(path: string): void {
  const parent = dirname(path);
  if (parent && parent !== '.' && !existsSync(parent)) {
    mkdirSync(parent, { recursive: true });
  }
}

function seedListArg(argv: readonly string[], name: string, fallback: readonly number[]): number[] {
  const value = listArgValue(argv, name);
  if (!value) {
    return [...fallback];
  }

  const seeds = value.split(',')
    .map((part) => Math.floor(Number(part.trim())))
    .filter((seed) => Number.isFinite(seed));
  return seeds.length > 0 ? seeds : [...fallback];
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  return index === -1 || index === argv.length - 1 ? undefined : argv[index + 1];
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

function numberArg(argv: readonly string[], name: string, fallback: number): number {
  const value = stringArg(argv, name);
  if (value === undefined) {
    return fallback;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  return Math.max(1, Math.floor(numberArg(argv, name, fallback)));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/diagnose-runtime-failures.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/diagnose-runtime-failures.js')) {
  main();
}
