import type { Team } from './model';
import type { CommandMap, TrackPower } from './strategy';

type TrackKeys = {
  forward: string;
  backward: string;
};

export type HumanControlScheme = {
  left: TrackKeys;
  right: TrackKeys;
};

export const HUMAN_CONTROL_SCHEMES: Record<Team, HumanControlScheme> = {
  red: {
    left: { forward: 'q', backward: 'a' },
    right: { forward: 'w', backward: 's' }
  },
  blue: {
    left: { forward: 'p', backward: ';' },
    right: { forward: '[', backward: "'" }
  }
};

export function humanCommandForTeam(team: Team, pressedKeys: ReadonlySet<string>): CommandMap {
  const scheme = HUMAN_CONTROL_SCHEMES[team];

  return {
    [`${team}-0`]: {
      leftTrack: trackPowerFromKeys(pressedKeys, scheme.left),
      rightTrack: trackPowerFromKeys(pressedKeys, scheme.right)
    }
  };
}

function trackPowerFromKeys(pressedKeys: ReadonlySet<string>, keys: TrackKeys): TrackPower {
  const forward = pressedKeys.has(keys.forward);
  const backward = pressedKeys.has(keys.backward);

  if (forward === backward) {
    return 0;
  }

  return forward ? 1 : -1;
}
