import { FIELD, type GameState, type Tank, type Team } from '../game/model';
import type { CommandMap, Strategy, TankCommand } from '../game/strategy';
import { actionIndexToCommand, commandToActionIndex } from './policyActions';
import {
  evaluatePolicy,
  policyProbabilities
} from './policyNetwork';
import {
  NEURAL_WEIGHT_COUNT,
  defaultNeuralWeights,
  type NeuralWeights
} from './neuralWeights';
import { chooseTacticalAction } from './tacticalRollout';

const STAMINA_CONSERVE_RATIO = 0.58;
const CRITICAL_STAMINA_RATIO = 0.22;
const DECISIVE_CONTACT_BUFFER = 28;
const STOP_COMMAND: TankCommand = { leftTrack: 0, rightTrack: 0 };

export type NeuralStrategyOptions = {
  weights?: NeuralWeights | (() => NeuralWeights);
  name?: string;
  tacticalRollout?: boolean;
  onDecision?: (trace: NeuralDecisionTrace) => void;
};

export type NeuralDecisionTrace = {
  frame: number;
  team: Team;
  tankId: string;
  inputs: number[];
  staminaRatio: number;
  ballDistance: number;
  ballSpeed: number;
  finishingPressure: number;
  ownGoalPressure: number;
  sideWallPressure: number;
  attackCornerPressure: number;
  ownCornerPressure: number;
  rawPolicyActionIndex?: number;
  policyActionIndex?: number;
  tacticalActionIndex?: number;
  tacticalActionScores?: number[];
  finalActionIndex: number;
  tacticalRolloutUsed: boolean;
  tacticalRolloutChanged: boolean;
  staminaConserved: boolean;
  criticalStaminaRegulated: boolean;
  flatPolicy: boolean;
};

type PressureSignals = {
  finishing: number;
  ownGoal: number;
  sideWall: number;
  sideWallDirection: number;
  attackCorner: number;
  ownCorner: number;
};

type LocalVector = {
  forward: number;
  lateral: number;
  distance: number;
};

export function createNeuralStrategy(options: NeuralStrategyOptions = {}): Strategy {
  const providedWeights = options.weights;
  const tacticalRollout = options.tacticalRollout ?? true;
  const onDecision = options.onDecision;
  const resolveWeights =
    typeof providedWeights === 'function'
      ? () => validateWeights(providedWeights())
      : (() => {
          const fixedWeights = validateWeights(providedWeights ?? defaultNeuralWeights());
          return () => fixedWeights;
        })();

  return {
    name: options.name ?? 'neural-policy',
    decide(state, team): CommandMap {
      const tank = state.tanks.find((candidate) => candidate.team === team && candidate.index === 0);
      if (!tank) {
        return {};
      }

      const pressures = pressureSignals(state, team);
      if (shouldConserveStamina(state, team, tank, pressures)) {
        const rawPolicyActionIndex = onDecision
          ? policyArgmaxActionIndex(evaluateTankNetwork(state, team, tank, resolveWeights()))
          : undefined;
        onDecision?.(decisionTrace(state, team, tank, pressures, {
          rawPolicyActionIndex,
          finalCommand: STOP_COMMAND,
          tacticalRolloutUsed: false,
          tacticalRolloutChanged: false,
          staminaConserved: true,
          criticalStaminaRegulated: false,
          flatPolicy: false
        }));
        return { [tank.id]: STOP_COMMAND };
      }

      const decision = policyOutputToDecision(
          state,
          team,
          evaluateTankNetwork(state, team, tank, resolveWeights()),
          tacticalRollout && shouldUseTacticalRollout(state, team, tank, pressures)
        );
      const command = decision.command;
      const regulatedCommand = regulateCriticalStaminaCommand(state, team, tank, pressures, command);
      const criticalStaminaRegulated = commandToActionIndex(regulatedCommand) !== commandToActionIndex(command);

      onDecision?.(decisionTrace(state, team, tank, pressures, {
        policyActionIndex: decision.policyActionIndex,
        rawPolicyActionIndex: decision.rawPolicyActionIndex,
        tacticalActionIndex: decision.tacticalActionIndex,
        tacticalActionScores: decision.tacticalActionScores,
        finalCommand: regulatedCommand,
        tacticalRolloutUsed: decision.tacticalRolloutUsed,
        tacticalRolloutChanged: decision.tacticalRolloutChanged,
        staminaConserved: false,
        criticalStaminaRegulated,
        flatPolicy: decision.flatPolicy
      }));

      return {
        [tank.id]: regulatedCommand
      };
    }
  };
}

export function evaluateTankNetwork(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  weights: NeuralWeights = defaultNeuralWeights()
): number[] {
  return evaluatePolicy(extractTankInputs(state, team, tank), validateWeights(weights));
}

export function extractTankInputs(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank
): number[] {
  const sign = teamSign(team);
  const heading = attackHeading(tank, team);
  const velocity = attackVelocity(tank.velocity, team);
  const forwardSpeed =
    velocity.x * Math.cos(heading) + velocity.y * Math.sin(heading);
  const lateralSpeed =
    -velocity.x * Math.sin(heading) + velocity.y * Math.cos(heading);
  const maxAngularVelocity = tank.maxTrackSpeed * 2 / tank.trackWidth;
  const ballDelta = attackDelta(team, tank.position, state.ball.position);
  const ballLocal = targetInTankFrame(tank, team, state.ball.position);
  const ballBearing = localBearing(ballLocal);
  const goal = goalPoint(team);
  const ownGoal = ownGoalPoint(team);
  const goalDelta = attackDelta(team, tank.position, goal);
  const ownGoalDelta = attackDelta(team, tank.position, ownGoal);
  const pressures = pressureSignals(state, team);
  const opponent = nearestOpponentInputs(state, team, tank);
  const target = targetInTankFrame(tank, team, tacticalTarget(state, team, tank, pressures));
  const targetBearing = localBearing(target);
  const closeScale = FIELD.ballRadius + FIELD.tankRadius;

  return [
    normalizeSigned((tank.position.x - FIELD.length / 2) * sign, FIELD.length / 2),
    normalizeSigned((tank.position.y - FIELD.width / 2) * sign, FIELD.width / 2),
    Math.cos(heading),
    Math.sin(heading),
    clamp(forwardSpeed / tank.maxTrackSpeed),
    clamp(lateralSpeed / tank.maxTrackSpeed),
    clamp(tank.angularVelocity / maxAngularVelocity),
    staminaRatio(tank),
    normalizeSigned(ballDelta.x, FIELD.length),
    normalizeSigned(ballDelta.y, FIELD.width),
    clamp(state.ball.velocity.x * sign / tank.maxTrackSpeed),
    clamp(state.ball.velocity.y * sign / tank.maxTrackSpeed),
    clamp01(ballLocal.distance / FIELD.length),
    ballBearing.forward,
    ballBearing.lateral,
    normalizeSigned(ballLocal.forward, closeScale),
    normalizeSigned(ballLocal.lateral, closeScale),
    normalizeSigned(goalDelta.x, FIELD.length),
    normalizeSigned(goalDelta.y, FIELD.width),
    normalizeSigned(ownGoalDelta.x, FIELD.length),
    normalizeSigned(ownGoalDelta.y, FIELD.width),
    pressures.finishing,
    pressures.ownGoal,
    pressures.sideWall,
    pressures.sideWallDirection,
    pressures.attackCorner,
    pressures.ownCorner,
    opponent.dx,
    opponent.dy,
    opponent.distance,
    opponent.bearingForward,
    opponent.bearingLateral,
    normalizeSigned(target.forward, FIELD.length),
    normalizeSigned(target.lateral, FIELD.width),
    targetBearing.forward,
    targetBearing.lateral
  ];
}

function tacticalTarget(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): { x: number; y: number } {
  if (pressures.ownGoal > 0.5 && pressures.ownGoal > pressures.finishing + 0.12) {
    return defensiveClearTarget(state, team);
  }

  if (Math.max(pressures.attackCorner, pressures.ownCorner) > 0.52) {
    return cornerRecycleTarget(state, team, pressures.attackCorner >= pressures.ownCorner ? 'attack' : 'own');
  }

  const finish = finishingTarget(state, team);
  const finishLocal = targetInTankFrame(tank, team, finish);
  if (finishLocal.distance < FIELD.tankRadius * 0.42) {
    return goalPoint(team);
  }

  return finish;
}

function shouldConserveStamina(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (shouldWaitForSafeOwnCornerRelease(state, team, tank, pressures)) {
    return true;
  }

  if (shouldWaitForDriftingFinish(state, team, tank, pressures)) {
    return true;
  }

  if (staminaRatio(tank) >= STAMINA_CONSERVE_RATIO) {
    return false;
  }

  return !urgentStaminaSpend(state, team, tank, pressures);
}

function shouldWaitForSafeOwnCornerRelease(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (pressures.ownCorner < 0.56 || pressures.ownGoal > 0.34 || pressures.attackCorner > 0.1) {
    return false;
  }

  const sign = teamSign(team);
  const attackVelocity = state.ball.velocity.x * sign;
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (attackVelocity < 12 || ballSpeed > 90) {
    return false;
  }

  const ballDistance = ballDistanceToTank(state, tank);
  return ballDistance > tank.radius + state.ball.radius - 8 &&
    ballDistance < FIELD.tankRadius * 1.55;
}

function shouldWaitForDriftingFinish(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (shouldWaitForOffsetRollingFinish(state, team, tank, pressures)) {
    return true;
  }

  if (staminaRatio(tank) > 0.24 || pressures.finishing < 0.9 || pressures.ownGoal > 0.24) {
    return false;
  }

  const sign = teamSign(team);
  const attackBallX = (state.ball.position.x - FIELD.length / 2) * sign + FIELD.length / 2;
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  if (attackBallX < FIELD.length - 190 || lane < 0.58) {
    return false;
  }

  const attackVelocity = state.ball.velocity.x * sign;
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (attackVelocity < -10 || ballSpeed > 70) {
    return false;
  }

  const ballDistance = ballDistanceToTank(state, tank);
  return ballDistance <= tank.radius + state.ball.radius + DECISIVE_CONTACT_BUFFER;
}

function shouldWaitForOffsetRollingFinish(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (staminaRatio(tank) > 0.3 || pressures.ownGoal > 0.24) {
    return false;
  }

  const sign = teamSign(team);
  const attackBallX = (state.ball.position.x - FIELD.length / 2) * sign + FIELD.length / 2;
  const attackBallY = (state.ball.position.y - FIELD.width / 2) * sign;
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  if (
    attackBallX < FIELD.length - 225 ||
    lane < 0.45 ||
    Math.abs(attackBallY) < 30
  ) {
    return false;
  }

  const attackVelocity = state.ball.velocity.x * sign;
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (attackVelocity < 0 || ballSpeed > 90) {
    return false;
  }

  const ballDistance = ballDistanceToTank(state, tank);
  if (ballDistance > tank.radius + state.ball.radius + DECISIVE_CONTACT_BUFFER) {
    return false;
  }

  const opponent = nearestOpponentTank(state, team);
  if (!opponent) {
    return false;
  }

  const opponentDistance = Math.hypot(
    opponent.position.x - state.ball.position.x,
    opponent.position.y - state.ball.position.y
  );
  return opponentDistance <= tank.radius + state.ball.radius + DECISIVE_CONTACT_BUFFER;
}

function urgentStaminaSpend(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (pressures.ownGoal > 0.5) {
    return true;
  }

  if (shouldRecoverCriticalStamina(state, team, tank, pressures)) {
    return false;
  }

  if (shouldRecoverLowPressureContactStamina(state, team, tank, pressures)) {
    return false;
  }

  if (isLooseBallContest(state, team, tank)) {
    return true;
  }

  return decisiveBallContact(state, team, tank, pressures);
}

function shouldRecoverLowPressureContactStamina(
  state: Readonly<GameState>,
  _team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (staminaRatio(tank) >= 0.34 || pressures.finishing > 0.45 || pressures.ownGoal > 0.35) {
    return false;
  }

  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (ballSpeed > 90) {
    return false;
  }

  const contactDistance = tank.radius + state.ball.radius + DECISIVE_CONTACT_BUFFER;
  return ballDistanceToTank(state, tank) <= contactDistance;
}

function shouldRecoverCriticalStamina(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (staminaRatio(tank) > CRITICAL_STAMINA_RATIO) {
    return false;
  }

  if (pressures.ownGoal > 0.35 || isClinchingFinish(state, team)) {
    return false;
  }

  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (ballSpeed > 80) {
    return false;
  }

  const ballDistance = ballDistanceToTank(state, tank);
  if (ballDistance > tank.radius + state.ball.radius + DECISIVE_CONTACT_BUFFER) {
    return false;
  }

  const opponent = nearestOpponentTank(state, team);
  if (!opponent) {
    return false;
  }

  const opponentDistance = Math.hypot(
    opponent.position.x - state.ball.position.x,
    opponent.position.y - state.ball.position.y
  );
  return opponentDistance <= ballDistance + FIELD.tankRadius * 0.75;
}

function isClinchingFinish(state: Readonly<GameState>, team: Team): boolean {
  const sign = teamSign(team);
  const attackX = (state.ball.position.x - FIELD.length / 2) * sign + FIELD.length / 2;
  const lane = Math.abs(state.ball.position.y - FIELD.width / 2) < FIELD.goalMouth * 0.46;
  const attackVelocity = state.ball.velocity.x * sign;
  return attackX > FIELD.length - 270 && lane && attackVelocity > -25;
}

function shouldUseTacticalRollout(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  if (decisiveBallContact(state, team, tank, pressures)) {
    return true;
  }

  if (pressures.ownGoal > 0.5) {
    return true;
  }

  if (Math.max(pressures.attackCorner, pressures.ownCorner) > 0.52) {
    return true;
  }

  return pressures.sideWall > 0.72 && ballDistanceToTank(state, tank) < FIELD.tankRadius * 3.4;
}

function regulateCriticalStaminaCommand(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals,
  command: TankCommand
): TankCommand {
  if (staminaRatio(tank) > CRITICAL_STAMINA_RATIO || pressures.ownGoal > 0.5) {
    return command;
  }

  if (Math.abs(command.leftTrack) + Math.abs(command.rightTrack) <= 1) {
    return command;
  }

  if (shouldPreserveCriticalRollingFinishPush(state, team, tank, pressures, command)) {
    return command;
  }

  const local = targetInTankFrame(tank, team, state.ball.position);
  if (Math.abs(local.lateral) < FIELD.tankRadius * 0.18) {
    return command.leftTrack === command.rightTrack
      ? { leftTrack: command.leftTrack, rightTrack: 0 }
      : STOP_COMMAND;
  }

  const turnTowardBall = local.lateral > 0 ? 1 : -1;
  return turnTowardBall > 0
    ? { leftTrack: command.leftTrack, rightTrack: 0 }
    : { leftTrack: 0, rightTrack: command.rightTrack };
}

function shouldPreserveCriticalRollingFinishPush(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals,
  command: TankCommand
): boolean {
  if (command.leftTrack !== 1 || command.rightTrack !== 1) {
    return false;
  }

  if (pressures.finishing < 0.78 || pressures.finishing > 0.86 || pressures.ownGoal > 0.32) {
    return false;
  }

  const sign = teamSign(team);
  const attackBallX = (state.ball.position.x - FIELD.length / 2) * sign + FIELD.length / 2;
  const attackBallY = (state.ball.position.y - FIELD.width / 2) * sign;
  const lane = 1 - clamp01(Math.abs(state.ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  if (
    attackBallX < FIELD.length - 275 ||
    attackBallX > FIELD.length - 240 ||
    attackBallY < -12 ||
    lane < 0.94
  ) {
    return false;
  }

  const attackVelocity = state.ball.velocity.x * sign;
  const attackLateralVelocity = state.ball.velocity.y * sign;
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  if (
    attackVelocity < 8 ||
    attackVelocity > 80 ||
    Math.abs(attackLateralVelocity) > 8 ||
    ballSpeed > 85
  ) {
    return false;
  }

  const local = targetInTankFrame(tank, team, state.ball.position);
  return local.forward > FIELD.ballRadius &&
    Math.abs(local.lateral) < FIELD.tankRadius * 0.42 &&
    local.distance <= FIELD.tankRadius;
}

function isLooseBallContest(state: Readonly<GameState>, team: Team, tank: Tank): boolean {
  const ballDistance = ballDistanceToTank(state, tank);
  if (ballDistance > FIELD.tankRadius * 4.1) {
    return false;
  }

  const opponent = nearestOpponentTank(state, team);
  const opponentDistance = opponent
    ? Math.hypot(opponent.position.x - state.ball.position.x, opponent.position.y - state.ball.position.y)
    : Number.POSITIVE_INFINITY;
  const ballSpeed = Math.hypot(state.ball.velocity.x, state.ball.velocity.y);
  const midfield = Math.abs(state.ball.position.x - FIELD.length / 2) < FIELD.length * 0.25;
  const contested = opponentDistance < ballDistance + FIELD.tankRadius * 1.25;

  return ballSpeed < 160 && (midfield || contested);
}

function decisiveBallContact(
  state: Readonly<GameState>,
  _team: Team,
  tank: Tank,
  pressures: PressureSignals
): boolean {
  const contactDistance = tank.radius + state.ball.radius + DECISIVE_CONTACT_BUFFER;
  const nearBall = Math.hypot(
    tank.position.x - state.ball.position.x,
    tank.position.y - state.ball.position.y
  ) <= contactDistance;

  if (!nearBall) {
    return false;
  }

  if (Math.max(pressures.attackCorner, pressures.ownCorner) > 0.52 && pressures.ownGoal <= 0.35) {
    return false;
  }

  return pressures.finishing > 0.35 ||
    pressures.ownGoal > 0.35;
}

function ballDistanceToTank(state: Readonly<GameState>, tank: Tank): number {
  return Math.hypot(
    tank.position.x - state.ball.position.x,
    tank.position.y - state.ball.position.y
  );
}

function finishingTarget(state: Readonly<GameState>, team: Team): { x: number; y: number } {
  const goal = goalPoint(team);
  const ball = state.ball.position;
  const shot = {
    x: goal.x - ball.x,
    y: goal.y - ball.y
  };
  const shotDistance = Math.hypot(shot.x, shot.y) || 1;
  const shotUnit = {
    x: shot.x / shotDistance,
    y: shot.y / shotDistance
  };
  const setupDistance = FIELD.ballRadius + FIELD.tankRadius + 8;

  return clampTankPoint({
    x: ball.x - shotUnit.x * setupDistance,
    y: ball.y - shotUnit.y * setupDistance
  });
}

function defensiveClearTarget(state: Readonly<GameState>, team: Team): { x: number; y: number } {
  const sign = teamSign(team);
  const ball = state.ball;
  const ownX = team === 'red' ? 0 : FIELD.length;
  const blockX = ownX + sign * (FIELD.tankRadius + FIELD.ballRadius + 22);
  const incomingVelocity = ball.velocity.x * sign;
  const secondsToBlock =
    incomingVelocity < -20
      ? clamp01((blockX - ball.position.x) / ball.velocity.x)
      : 0;
  const predictedY = clampRange(
    ball.position.y + ball.velocity.y * secondsToBlock,
    FIELD.width / 2 - FIELD.goalMouth * 0.54,
    FIELD.width / 2 + FIELD.goalMouth * 0.54
  );
  const clearSetup = FIELD.ballRadius + FIELD.tankRadius + 4;
  const clearPoint = {
    x: ball.position.x - sign * clearSetup,
    y: predictedY
  };

  return clampTankPoint({
    x: pressuresPreferBlock(state, team) ? blockX : clearPoint.x,
    y: clearPoint.y
  });
}

function pressuresPreferBlock(state: Readonly<GameState>, team: Team): boolean {
  const sign = teamSign(team);
  return state.ball.velocity.x * sign < -110;
}

function cornerRecycleTarget(
  state: Readonly<GameState>,
  team: Team,
  mode: 'attack' | 'own'
): { x: number; y: number } {
  const sign = teamSign(team);
  const ball = state.ball.position;
  const attackFrameY = (ball.y - FIELD.width / 2) * sign;
  const wallAway = attackFrameY >= 0 ? -1 : 1;
  const xDirection = mode === 'attack' ? -1 : 1;
  const offsetX = xDirection * FIELD.tankRadius * 2.3;
  const offsetY = wallAway * FIELD.tankRadius * 2.15;

  return clampTankPoint({
    x: ball.x + offsetX * sign,
    y: ball.y + offsetY * sign
  });
}

function pressureSignals(state: Readonly<GameState>, team: Team): PressureSignals {
  const sign = teamSign(team);
  const ball = state.ball;
  const attackProgress = ((ball.position.x - FIELD.length / 2) * sign) / (FIELD.length / 2);
  const ownGoalX = team === 'red' ? 0 : FIELD.length;
  const distanceFromOwnGoal = (ball.position.x - ownGoalX) * sign;
  const ownDepth = 1 - clamp01(distanceFromOwnGoal / (FIELD.length * 0.44));
  const lane = 1 - clamp01(Math.abs(ball.position.y - FIELD.width / 2) / (FIELD.goalMouth * 0.74));
  const incoming = clamp01((-ball.velocity.x * sign) / 260);
  const sideWall = sideWallPressure(ball.position.y);
  const sideWallDirection = normalizeSigned((ball.position.y - FIELD.width / 2) * sign, FIELD.width / 2);
  const finishing = clamp(attackProgress * 1.15 + lane * 0.36 - 0.12);
  const ownGoal = clamp01(ownDepth * 0.68 + lane * 0.44 + incoming * 0.34 - 0.18);

  return {
    finishing,
    ownGoal,
    sideWall,
    sideWallDirection,
    attackCorner: clamp01(Math.max(0, attackProgress) * sideWall),
    ownCorner: clamp01(ownDepth * sideWall)
  };
}

function nearestOpponentInputs(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank
): {
  dx: number;
  dy: number;
  distance: number;
  bearingForward: number;
  bearingLateral: number;
} {
  let best: Tank | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const candidate of state.tanks) {
    if (candidate.team === team) {
      continue;
    }
    const distance = squaredDistance(tank.position, candidate.position);
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }

  if (!best) {
    return {
      dx: 0,
      dy: 0,
      distance: 1,
      bearingForward: 0,
      bearingLateral: 0
    };
  }

  const delta = attackDelta(team, tank.position, best.position);
  const local = targetInTankFrame(tank, team, best.position);
  const bearing = localBearing(local);

  return {
    dx: normalizeSigned(delta.x, FIELD.length),
    dy: normalizeSigned(delta.y, FIELD.width),
    distance: clamp01(local.distance / FIELD.length),
    bearingForward: bearing.forward,
    bearingLateral: bearing.lateral
  };
}

function nearestOpponentTank(state: Readonly<GameState>, team: Team): Tank | undefined {
  let best: Tank | undefined;
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const candidate of state.tanks) {
    if (candidate.team === team) {
      continue;
    }
    const distance = squaredDistance(state.ball.position, candidate.position);
    if (distance < bestDistance) {
      best = candidate;
      bestDistance = distance;
    }
  }

  return best;
}

function targetInTankFrame(tank: Tank, team: Team, target: { x: number; y: number }): LocalVector {
  const heading = attackHeading(tank, team);
  const delta = attackDelta(team, tank.position, target);

  return {
    forward: Math.cos(heading) * delta.x + Math.sin(heading) * delta.y,
    lateral: -Math.sin(heading) * delta.x + Math.cos(heading) * delta.y,
    distance: Math.hypot(delta.x, delta.y)
  };
}

function localBearing(local: LocalVector): { forward: number; lateral: number } {
  const distance = local.distance || 1;
  return {
    forward: clamp(local.forward / distance),
    lateral: clamp(local.lateral / distance)
  };
}

function policyOutputToDecision(
  state: Readonly<GameState>,
  team: Team,
  logits: readonly number[],
  useTacticalRollout: boolean
): {
  command: TankCommand;
  policyActionIndex?: number;
  rawPolicyActionIndex?: number;
  tacticalActionIndex?: number;
  tacticalActionScores?: number[];
  tacticalRolloutUsed: boolean;
  tacticalRolloutChanged: boolean;
  flatPolicy: boolean;
} {
  const maxLogit = Math.max(...logits);
  const minLogit = Math.min(...logits);
  if (Math.abs(maxLogit - minLogit) < 1e-9) {
    return {
      command: STOP_COMMAND,
      tacticalRolloutUsed: false,
      tacticalRolloutChanged: false,
      flatPolicy: true
    };
  }

  const policyActionIndex = policyArgmaxActionIndex(logits) ?? 4;
  if (!useTacticalRollout) {
    return {
      command: actionIndexToCommand(policyActionIndex),
      policyActionIndex,
      rawPolicyActionIndex: policyActionIndex,
      tacticalActionIndex: policyActionIndex,
      tacticalRolloutUsed: false,
      tacticalRolloutChanged: false,
      flatPolicy: false
    };
  }

  const bestIndex = chooseTacticalAction({
    state,
    team,
    policyActionIndex
  });

  return {
    command: actionIndexToCommand(bestIndex.actionIndex),
    policyActionIndex,
    rawPolicyActionIndex: policyActionIndex,
    tacticalActionIndex: bestIndex.actionIndex,
    tacticalActionScores: bestIndex.actionScores,
    tacticalRolloutUsed: true,
    tacticalRolloutChanged: bestIndex.actionIndex !== policyActionIndex,
    flatPolicy: false
  };
}

function decisionTrace(
  state: Readonly<GameState>,
  team: Team,
  tank: Tank,
  pressures: PressureSignals,
  decision: {
    policyActionIndex?: number;
    tacticalActionIndex?: number;
    tacticalActionScores?: number[];
    finalCommand: TankCommand;
    tacticalRolloutUsed: boolean;
    tacticalRolloutChanged: boolean;
    staminaConserved: boolean;
    criticalStaminaRegulated: boolean;
    flatPolicy: boolean;
    rawPolicyActionIndex?: number;
  }
): NeuralDecisionTrace {
  return {
    frame: state.frame,
    team,
    tankId: tank.id,
    inputs: extractTankInputs(state, team, tank),
    staminaRatio: staminaRatio(tank),
    ballDistance: ballDistanceToTank(state, tank),
    ballSpeed: Math.hypot(state.ball.velocity.x, state.ball.velocity.y),
    finishingPressure: pressures.finishing,
    ownGoalPressure: pressures.ownGoal,
    sideWallPressure: pressures.sideWall,
    attackCornerPressure: pressures.attackCorner,
    ownCornerPressure: pressures.ownCorner,
    rawPolicyActionIndex: decision.rawPolicyActionIndex,
    policyActionIndex: decision.policyActionIndex,
    tacticalActionIndex: decision.tacticalActionIndex,
    tacticalActionScores: decision.tacticalActionScores,
    finalActionIndex: commandToActionIndex(decision.finalCommand),
    tacticalRolloutUsed: decision.tacticalRolloutUsed,
    tacticalRolloutChanged: decision.tacticalRolloutChanged,
    staminaConserved: decision.staminaConserved,
    criticalStaminaRegulated: decision.criticalStaminaRegulated,
    flatPolicy: decision.flatPolicy
  };
}

function policyArgmaxActionIndex(logits: readonly number[]): number | undefined {
  const maxLogit = Math.max(...logits);
  const minLogit = Math.min(...logits);
  if (Math.abs(maxLogit - minLogit) < 1e-9) {
    return undefined;
  }

  const probabilities = policyProbabilities(logits);
  return probabilities.reduce(
    (best, value, index) => value > probabilities[best] ? index : best,
    0
  );
}

function validateWeights(weights: NeuralWeights): NeuralWeights {
  if (weights.length !== NEURAL_WEIGHT_COUNT) {
    throw new Error(`Expected ${NEURAL_WEIGHT_COUNT} neural weights, received ${weights.length}`);
  }
  return weights;
}

function goalPoint(team: Team): { x: number; y: number } {
  return {
    x: team === 'red' ? FIELD.length : 0,
    y: FIELD.width / 2
  };
}

function ownGoalPoint(team: Team): { x: number; y: number } {
  return {
    x: team === 'red' ? 0 : FIELD.length,
    y: FIELD.width / 2
  };
}

function attackDelta(
  team: Team,
  from: { x: number; y: number },
  to: { x: number; y: number }
): { x: number; y: number } {
  const sign = teamSign(team);
  return {
    x: (to.x - from.x) * sign,
    y: (to.y - from.y) * sign
  };
}

function attackVelocity(velocity: { x: number; y: number }, team: Team): { x: number; y: number } {
  const sign = teamSign(team);
  return {
    x: velocity.x * sign,
    y: velocity.y * sign
  };
}

function attackHeading(tank: Tank, team: Team): number {
  return normalizeAngle(tank.angle - (team === 'red' ? 0 : Math.PI));
}

function teamSign(team: Team): 1 | -1 {
  return team === 'red' ? 1 : -1;
}

function staminaRatio(tank: Tank): number {
  return tank.maxStamina > 0 ? clamp01(tank.stamina / tank.maxStamina) : 0;
}

function sideWallPressure(y: number): number {
  const wallDistance = Math.min(y - FIELD.ballRadius, FIELD.width - FIELD.ballRadius - y);
  return clamp01(1 - wallDistance / (FIELD.goalMouth * 0.72));
}

function squaredDistance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function clampTankPoint(point: { x: number; y: number }): { x: number; y: number } {
  return {
    x: clampRange(point.x, FIELD.tankRadius, FIELD.length - FIELD.tankRadius),
    y: clampRange(point.y, FIELD.tankRadius, FIELD.width - FIELD.tankRadius)
  };
}

function normalizeSigned(value: number, scale: number): number {
  return clamp(value / scale);
}

function clamp(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

function clampRange(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
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
