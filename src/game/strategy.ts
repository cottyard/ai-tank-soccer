import type { GameState, Team } from './model';

export type TrackPower = -1 | 0 | 1;

export type TankCommand = {
  leftTrack: TrackPower;
  rightTrack: TrackPower;
};

export type CommandMap = Record<string, TankCommand>;

export type Strategy = {
  name: string;
  decide(state: Readonly<GameState>, team: Team): CommandMap;
};

export const STOP: TankCommand = { leftTrack: 0, rightTrack: 0 };

export function idleCommands(): CommandMap {
  return {};
}

export function sanitizeCommand(command: TankCommand | undefined): TankCommand {
  if (!command) {
    return STOP;
  }

  return {
    leftTrack: sanitizeTrack(command.leftTrack),
    rightTrack: sanitizeTrack(command.rightTrack)
  };
}

export class AiClock {
  private readonly framesPerDecision: number;
  private commands: CommandMap = {};
  private lastDecisionFrame = Number.NEGATIVE_INFINITY;

  constructor(
    private readonly redStrategy: Strategy,
    private readonly blueStrategy: Strategy,
    physicsHz = 30,
    aiHz = 5
  ) {
    this.framesPerDecision = Math.max(1, Math.round(physicsHz / aiHz));
  }

  update(state: GameState): CommandMap {
    if (state.frame - this.lastDecisionFrame >= this.framesPerDecision) {
      this.commands = {
        ...this.redStrategy.decide(state, 'red'),
        ...this.blueStrategy.decide(state, 'blue')
      };
      this.lastDecisionFrame = state.frame;
    }

    return this.commands;
  }
}

function sanitizeTrack(track: number): TrackPower {
  if (track > 0) {
    return 1;
  }
  if (track < 0) {
    return -1;
  }
  return 0;
}
