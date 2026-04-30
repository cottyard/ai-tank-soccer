import type { TankCommand, TrackPower } from '../game/strategy';

export const POLICY_ACTIONS = [
  { leftTrack: -1, rightTrack: -1 },
  { leftTrack: -1, rightTrack: 0 },
  { leftTrack: -1, rightTrack: 1 },
  { leftTrack: 0, rightTrack: -1 },
  { leftTrack: 0, rightTrack: 0 },
  { leftTrack: 0, rightTrack: 1 },
  { leftTrack: 1, rightTrack: -1 },
  { leftTrack: 1, rightTrack: 0 },
  { leftTrack: 1, rightTrack: 1 }
] as const satisfies readonly TankCommand[];

export const POLICY_ACTION_COUNT = POLICY_ACTIONS.length;

export function actionIndexToCommand(index: number): TankCommand {
  const action = POLICY_ACTIONS[clampActionIndex(index)];
  return {
    leftTrack: action.leftTrack,
    rightTrack: action.rightTrack
  };
}

export function commandToActionIndex(command: TankCommand): number {
  const left = sanitizeTrack(command.leftTrack);
  const right = sanitizeTrack(command.rightTrack);
  const index = POLICY_ACTIONS.findIndex(
    (action) => action.leftTrack === left && action.rightTrack === right
  );

  return index === -1 ? 4 : index;
}

function clampActionIndex(index: number): number {
  if (!Number.isFinite(index)) {
    return 4;
  }
  return Math.max(0, Math.min(POLICY_ACTION_COUNT - 1, Math.round(index)));
}

function sanitizeTrack(track: number): TrackPower {
  if (track > 0.5) {
    return 1;
  }
  if (track < -0.5) {
    return -1;
  }
  return 0;
}
