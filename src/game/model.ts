export const TEAM_SIZE = 1;

export type Team = 'red' | 'blue';

export type Vec2 = {
  x: number;
  y: number;
};

export type Ball = {
  position: Vec2;
  velocity: Vec2;
  radius: number;
  mass: number;
};

export type Tank = {
  id: string;
  team: Team;
  index: number;
  position: Vec2;
  velocity: Vec2;
  angle: number;
  angularVelocity: number;
  length: number;
  width: number;
  noseLength: number;
  radius: number;
  trackWidth: number;
  maxTrackSpeed: number;
  mass: number;
  momentOfInertia: number;
  trackForce: number;
  staticFriction: number;
  kineticFriction: number;
  stamina: number;
  maxStamina: number;
};

export type Score = Record<Team, number>;

export type GoalEvent = {
  team: Team;
  frame: number;
  time: number;
};

export type GameState = {
  frame: number;
  time: number;
  tanks: Tank[];
  ball: Ball;
  score: Score;
  lastGoal: GoalEvent | null;
};

export const FIELD = {
  length: 1050,
  width: 680,
  ballRadius: 34,
  tankLength: 102,
  tankWidth: 102,
  tankNoseLength: 51,
  tankRadius: Math.hypot(102, 51),
  gravity: 600,
  tankMass: 9,
  ballMass: 1.4,
  tankTrackForce: 2000,
  tankStaticFriction: 4.4,
  tankKineticFriction: 0.12,
  tankStamina: 100,
  tankStaminaDrainPerTrackSecond: 10,
  tankStaminaRecoveryPerSecond: 20,
  goalMouth: 230,
  wallRestitution: 0.78,
  ballDampingPerSecond: 0.46
} as const;

const RED_Y = [FIELD.width / 2];
const BLUE_Y = [FIELD.width / 2];

export function createInitialState(): GameState {
  return {
    frame: 0,
    time: 0,
    tanks: [
      ...createTeamTanks('red', 170, RED_Y, 0),
      ...createTeamTanks('blue', FIELD.length - 170, BLUE_Y, Math.PI)
    ],
    ball: {
      position: { x: FIELD.length / 2, y: FIELD.width / 2 },
      velocity: { x: 0, y: 0 },
      radius: FIELD.ballRadius,
      mass: FIELD.ballMass
    },
    score: { red: 0, blue: 0 },
    lastGoal: null
  };
}

export function resetForKickoff(state: GameState): void {
  const score = state.score;
  const frame = state.frame;
  const time = state.time;
  const lastGoal = state.lastGoal;
  const fresh = createInitialState();
  state.frame = frame;
  state.time = time;
  state.score = score;
  state.lastGoal = lastGoal;
  state.tanks = fresh.tanks;
  state.ball = fresh.ball;
}

export function cloneState(state: GameState): GameState {
  return {
    frame: state.frame,
    time: state.time,
    score: { ...state.score },
    lastGoal: state.lastGoal ? { ...state.lastGoal } : null,
    ball: {
      radius: state.ball.radius,
      mass: state.ball.mass,
      position: { ...state.ball.position },
      velocity: { ...state.ball.velocity }
    },
    tanks: state.tanks.map((tank) => ({
      ...tank,
      position: { ...tank.position },
      velocity: { ...tank.velocity }
    }))
  };
}

export function goalTeamForSide(side: 'left' | 'right'): Team {
  return side === 'left' ? 'blue' : 'red';
}

export function isInsideGoalMouth(y: number): boolean {
  const halfMouth = FIELD.goalMouth / 2;
  return y >= FIELD.width / 2 - halfMouth && y <= FIELD.width / 2 + halfMouth;
}

function createTeamTanks(team: Team, x: number, yValues: number[], angle: number): Tank[] {
  return yValues.map((y, index) => ({
    id: `${team}-${index}`,
    team,
    index,
    position: { x, y },
    velocity: { x: 0, y: 0 },
    angle,
    angularVelocity: 0,
    length: FIELD.tankLength,
    width: FIELD.tankWidth,
    noseLength: FIELD.tankNoseLength,
    radius: FIELD.tankRadius,
    trackWidth: FIELD.tankWidth * 0.82,
    maxTrackSpeed: 245,
    mass: FIELD.tankMass,
    momentOfInertia: FIELD.tankMass * (FIELD.tankLength ** 2 + FIELD.tankWidth ** 2) / 12,
    trackForce: FIELD.tankTrackForce,
    staticFriction: FIELD.tankStaticFriction,
    kineticFriction: FIELD.tankKineticFriction,
    stamina: FIELD.tankStamina,
    maxStamina: FIELD.tankStamina
  }));
}
