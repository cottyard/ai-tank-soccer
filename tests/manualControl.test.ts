import { describe, expect, it } from 'vitest';
import { humanCommandForTeam } from '../src/game/manualControl';

describe('manual tank controls', () => {
  it('maps Q and W to red forward tracks', () => {
    const command = humanCommandForTeam('red', new Set(['q', 'w']));

    expect(command).toEqual({
      'red-0': { leftTrack: 1, rightTrack: 1 }
    });
  });

  it('maps P and [ to blue forward tracks', () => {
    const command = humanCommandForTeam('blue', new Set(['p', '[']));

    expect(command).toEqual({
      'blue-0': { leftTrack: 1, rightTrack: 1 }
    });
  });

  it("maps ; and ' to blue reverse tracks", () => {
    const command = humanCommandForTeam('blue', new Set([';', "'"]));

    expect(command).toEqual({
      'blue-0': { leftTrack: -1, rightTrack: -1 }
    });
  });

  it('keeps red and blue human control schemes independent', () => {
    expect(humanCommandForTeam('red', new Set(['q', 'w']))).toEqual({
      'red-0': { leftTrack: 1, rightTrack: 1 }
    });
    expect(humanCommandForTeam('blue', new Set(['q', 'w']))).toEqual({
      'blue-0': { leftTrack: 0, rightTrack: 0 }
    });
  });

  it('stops a track when both directions are pressed for that track', () => {
    const command = humanCommandForTeam('red', new Set(['q', 'a', 'w']));

    expect(command).toEqual({
      'red-0': { leftTrack: 0, rightTrack: 1 }
    });
  });
});
