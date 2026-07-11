import { readFileSync } from 'node:fs';
import { loadWeightsPayload } from './coach-neural';
import { createNeuralStrategy, type NeuralDecisionTrace } from '../src/ai/neuralStrategy';
import { chooseTacticalAction } from '../src/ai/tacticalRollout';
import { traditionalStrategy } from '../src/ai/traditionalStrategy';
import { AI_HZ, FIXED_DT, PHYSICS_HZ } from '../src/game/match';
import { FIELD, cloneState, createInitialState, type GameState, type Team, type Vec2 } from '../src/game/model';
import { stepGame } from '../src/game/simulation';
import { AiClock } from '../src/game/strategy';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type InspectOptions = {
  weightsPath: string;
  seed: number;
  match: number;
  frames: number;
  from: number;
  rolloutFrames: number[];
};

type RoundedPoint = {
  x: number;
  y: number;
};

export function parseInspectRuntimeMatchArgs(argv: readonly string[]): InspectOptions {
  return {
    weightsPath: stringArg(argv, '--weights') ?? 'public/models/neural-best.json',
    seed: integerArg(argv, '--seed', 71),
    match: integerArg(argv, '--match', 3),
    frames: positiveIntegerArg(argv, '--frames', 600),
    from: integerArg(argv, '--from', 360),
    rolloutFrames: numberListArg(argv, '--rollout-frames', [])
  };
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const options = parseInspectRuntimeMatchArgs(argv);
    const weights = loadWeightsPayload(readFileSync(options.weightsPath, 'utf8'));
    const team: Team = options.match % 2 === 0 ? 'red' : 'blue';
    const traces: NeuralDecisionTrace[] = [];
    const neural = createNeuralStrategy({
      weights,
      tacticalRollout: true,
      onDecision: (trace) => traces.push(trace)
    });
    const state = createSeededInitialState(options.seed, options.match, team);
    const clock = new AiClock(
      team === 'red' ? neural : traditionalStrategy,
      team === 'blue' ? neural : traditionalStrategy,
      PHYSICS_HZ,
      AI_HZ
    );
    const frameStates = new Map<number, GameState>();

    for (let frame = 0; frame <= options.frames; frame += 1) {
      if (state.frame >= options.from && state.frame % aiFrameStep() === 0) {
        frameStates.set(state.frame, cloneState(state));
      }
      if (frame === options.frames) {
        break;
      }
      stepGame(state, clock.update(state), FIXED_DT);
    }

    console.log(JSON.stringify({
      seed: options.seed,
      match: options.match,
      team,
      score: state.score,
      finalBall: roundPoint(state.ball.position),
      finalBallVelocity: roundPoint(state.ball.velocity),
      finalAttackX: round(attackX(team, state.ball.position.x)),
      finalSideWallDistance: round(sideWallDistance(state.ball.position.y))
    }));

    for (const trace of traces.filter((candidate) => candidate.frame >= options.from)) {
      const sampled = frameStates.get(trace.frame);
      if (!sampled) {
        continue;
      }
      console.log(JSON.stringify(inspectDecision(sampled, team, trace, options.rolloutFrames)));
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function inspectDecision(
  state: GameState,
  team: Team,
  trace: NeuralDecisionTrace,
  rolloutFrames: readonly number[]
): Record<string, unknown> {
  const tank = controlledTank(state, team);
  const opponent = controlledTank(state, team === 'red' ? 'blue' : 'red');
  const sign = team === 'red' ? 1 : -1;
  const ballAttackX = attackX(team, state.ball.position.x);
  const ballAttackY = (state.ball.position.y - FIELD.width / 2) * sign;
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  const scores = trace.tacticalActionScores?.map((value) => Number.isFinite(value) ? round(value) : null);
  const counterfactualTactical = trace.rawPolicyActionIndex === undefined || trace.tacticalRolloutUsed
    ? undefined
    : chooseTacticalAction({
        state,
        team,
        policyActionIndex: trace.rawPolicyActionIndex
      });
  const fixedRollouts = trace.rawPolicyActionIndex === undefined || rolloutFrames.length === 0
    ? undefined
    : rolloutFrames.map((frames) => {
        const choice = chooseTacticalAction({
          state,
          team,
          policyActionIndex: trace.rawPolicyActionIndex ?? 4,
          rolloutFrames: frames
        });
        return {
          frames,
          actionIndex: choice.actionIndex,
          policyScore: round(choice.policyScore),
          score: round(choice.score),
          actionScores: choice.actionScores.map((value) => Number.isFinite(value) ? round(value) : null)
        };
      });

  return {
    frame: trace.frame,
    rawActionIndex: trace.rawPolicyActionIndex,
    tacticalActionIndex: trace.tacticalActionIndex,
    finalActionIndex: trace.finalActionIndex,
    tacticalRolloutUsed: trace.tacticalRolloutUsed,
    tacticalRolloutChanged: trace.tacticalRolloutChanged,
    staminaConserved: trace.staminaConserved,
    criticalStaminaRegulated: trace.criticalStaminaRegulated,
    staminaRatio: round(trace.staminaRatio),
    ballDistance: round(trace.ballDistance),
    ballSpeed: round(trace.ballSpeed),
    finishingPressure: round(trace.finishingPressure),
    ownGoalPressure: round(trace.ownGoalPressure),
    sideWallPressure: round(trace.sideWallPressure),
    attackCornerPressure: round(trace.attackCornerPressure),
    ownCornerPressure: round(trace.ownCornerPressure),
    ball: roundPoint(state.ball.position),
    attackBallX: round(ballAttackX),
    attackBallY: round(ballAttackY),
    lane: round(lane),
    sideWallDistance: round(sideWallDistance(state.ball.position.y)),
    attackVelocity: round(state.ball.velocity.x * sign),
    attackLateralVelocity: round(state.ball.velocity.y * sign),
    tank: tank
      ? {
          position: roundPoint(tank.position),
          angle: round(tank.angle),
          staminaRatio: round(tank.maxStamina > 0 ? tank.stamina / tank.maxStamina : 0)
        }
      : undefined,
    opponent: opponent
      ? {
          position: roundPoint(opponent.position),
          angle: round(opponent.angle),
          staminaRatio: round(opponent.maxStamina > 0 ? opponent.stamina / opponent.maxStamina : 0)
        }
      : undefined,
    opponentBallDistance: opponent
      ? round(Math.hypot(opponent.position.x - state.ball.position.x, opponent.position.y - state.ball.position.y))
      : undefined,
    tacticalActionScores: scores,
    counterfactualTactical: counterfactualTactical
      ? {
          actionIndex: counterfactualTactical.actionIndex,
          policyScore: round(counterfactualTactical.policyScore),
          score: round(counterfactualTactical.score),
          actionScores: counterfactualTactical.actionScores.map((value) => Number.isFinite(value) ? round(value) : null)
        }
      : undefined,
    fixedRollouts
  };
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

function controlledTank(state: GameState, team: Team) {
  return state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
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

function sideWallDistance(y: number): number {
  return Math.min(y - FIELD.ballRadius, FIELD.width - FIELD.ballRadius - y);
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

function aiFrameStep(): number {
  return Math.max(1, Math.round(PHYSICS_HZ / AI_HZ));
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

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/inspect-runtime-match.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/inspect-runtime-match.js')) {
  main();
}
