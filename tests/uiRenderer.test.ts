import { describe, expect, it } from 'vitest';
import { tankBatteryLayout } from '../src/ui/renderer';

describe('field renderer tank battery indicator', () => {
  it('places a horizontal battery charge indicator inside the tank body', () => {
    const bodyLength = 102;
    const bodyWidth = 102;

    const layout = tankBatteryLayout(bodyLength, bodyWidth, 0.62);

    expect(layout.body.width).toBeGreaterThan(layout.body.height * 4);
    expect(layout.body.x).toBeCloseTo(-layout.body.width / 2, 6);
    expect(layout.body.y).toBeGreaterThanOrEqual(-bodyWidth / 2);
    expect(layout.body.y + layout.body.height).toBeLessThanOrEqual(bodyWidth / 2);
    expect(layout.cap.x).toBeGreaterThan(layout.body.x + layout.body.width);
    expect(layout.fill.width).toBeCloseTo(layout.fill.maxWidth * 0.62, 6);
  });
});
