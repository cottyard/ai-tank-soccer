import { describe, expect, it } from 'vitest';
import {
  POLICY_ACTIONS,
  POLICY_ACTION_COUNT,
  actionIndexToCommand,
  commandToActionIndex
} from '../src/ai/policyActions';

describe('policy action mapping', () => {
  it('defines the nine legal left/right track combinations in stable order', () => {
    expect(POLICY_ACTION_COUNT).toBe(9);
    expect(POLICY_ACTIONS).toEqual([
      { leftTrack: -1, rightTrack: -1 },
      { leftTrack: -1, rightTrack: 0 },
      { leftTrack: -1, rightTrack: 1 },
      { leftTrack: 0, rightTrack: -1 },
      { leftTrack: 0, rightTrack: 0 },
      { leftTrack: 0, rightTrack: 1 },
      { leftTrack: 1, rightTrack: -1 },
      { leftTrack: 1, rightTrack: 0 },
      { leftTrack: 1, rightTrack: 1 }
    ]);
  });

  it('round-trips every legal action through command conversion', () => {
    for (let index = 0; index < POLICY_ACTION_COUNT; index += 1) {
      expect(commandToActionIndex(actionIndexToCommand(index))).toBe(index);
    }
  });

  it('sanitizes arbitrary commands before mapping them to an action', () => {
    expect(commandToActionIndex({ leftTrack: 2, rightTrack: -5 } as never)).toBe(6);
    expect(commandToActionIndex({ leftTrack: 0.2, rightTrack: 0 } as never)).toBe(4);
  });
});
