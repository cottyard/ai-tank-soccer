import { describe, expect, it } from 'vitest';
import { humanCommandForTeam } from '../src/game/manualControl';

describe('manual tank controls', () => {
  it('maps F and J to forward tracks', () => {
    const command = humanCommandForTeam('red', new Set(['f', 'j']));

    expect(command).toEqual({
      'red-0': { leftTrack: 1, rightTrack: 1 }
    });
  });

  it('maps D and K to reverse tracks', () => {
    const command = humanCommandForTeam('blue', new Set(['d', 'k']));

    expect(command).toEqual({
      'blue-0': { leftTrack: -1, rightTrack: -1 }
    });
  });

  it('stops a track when both directions are pressed for that track', () => {
    const command = humanCommandForTeam('red', new Set(['f', 'd', 'j']));

    expect(command).toEqual({
      'red-0': { leftTrack: 0, rightTrack: 1 }
    });
  });
});
