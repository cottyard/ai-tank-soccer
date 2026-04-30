import { describe, expect, it } from 'vitest';
import { BUNDLED_POLICY_URL, loadBundledPolicy } from '../src/ai/bundledPolicy';
import { NEURAL_WEIGHT_COUNT, defaultNeuralWeights } from '../src/ai/neuralWeights';

describe('bundled policy loader', () => {
  it('loads valid bundled policy payloads through fetch', async () => {
    const weights = defaultNeuralWeights();
    const payload = {
      weights,
      metadata: {
        selectionScore: 123.45,
        bestCycle: 2
      }
    };
    const fetcher = async (url: RequestInfo | URL) => {
      expect(String(url)).toBe(BUNDLED_POLICY_URL);
      return {
        ok: true,
        async json() {
          return payload;
        }
      } as Response;
    };

    await expect(loadBundledPolicy(fetcher)).resolves.toEqual(payload);
  });

  it('ignores missing or obsolete bundled policy payloads', async () => {
    const missing = async () => ({
      ok: false,
      async json() {
        return {};
      }
    }) as Response;
    const obsolete = async () => ({
      ok: true,
      async json() {
        return { weights: Array.from({ length: NEURAL_WEIGHT_COUNT - 1 }, () => 0) };
      }
    }) as Response;

    await expect(loadBundledPolicy(missing)).resolves.toBeNull();
    await expect(loadBundledPolicy(obsolete)).resolves.toBeNull();
  });
});
