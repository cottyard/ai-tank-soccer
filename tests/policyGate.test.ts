import { describe, expect, it } from 'vitest';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  compareRuntimeDecisionTraces,
  compareRuntimeTraces,
  evaluatePolicyGate,
  evaluateRuntimePolicy,
  selectAcceptedPolicy,
  traceRuntimePolicyDecisions,
  traceRuntimePolicy,
  type RuntimeDecisionTraceRecord,
  type RuntimeTraceSummary
} from '../src/ai/policyGate';
import { POLICY_ACTION_COUNT } from '../src/ai/policyActions';
import {
  buildPolicyAnchorSamples,
  parseTraceRuntimePolicyArgs,
  runTraceRuntimePolicy,
  runTraceRuntimePolicyDetailed
} from '../scripts/trace-runtime-policy';
import { defaultNeuralWeights } from '../src/ai/neuralWeights';
import type { EvaluationOptions, EvaluationResult } from '../src/ai/neuralTraining';
import type { NeuralWeights } from '../src/ai/neuralWeights';

function scoredWeights(score: number): number[] {
  const weights = defaultNeuralWeights();
  weights[0] = score;
  return weights;
}

describe('policy adoption gate', () => {
  it('accepts a candidate only when its evaluation score improves enough', () => {
    const evaluate = (weights: NeuralWeights, _options: EvaluationOptions): EvaluationResult => ({
      score: weights[0],
      goalDiff: 0,
      ballProgress: 0
    });

    const rejected = evaluatePolicyGate(scoredWeights(10), scoredWeights(10.2), {
      evaluate,
      minDelta: 0.5
    });
    const accepted = evaluatePolicyGate(scoredWeights(10), scoredWeights(10.6), {
      evaluate,
      minDelta: 0.5
    });

    expect(rejected).toMatchObject({
      accepted: false,
      currentScore: 10,
      candidateScore: 10.2
    });
    expect(accepted).toMatchObject({
      accepted: true,
      currentScore: 10,
      candidateScore: 10.6
    });
  });

  it('selects the baseline weights when a saved candidate does not pass the gate', () => {
    const evaluate = (weights: NeuralWeights, _options: EvaluationOptions): EvaluationResult => ({
      score: weights[0],
      goalDiff: 0,
      ballProgress: 0
    });
    const baseline = scoredWeights(20);
    const weakSaved = scoredWeights(18);
    const strongSaved = scoredWeights(22);

    const rejected = selectAcceptedPolicy(baseline, weakSaved, { evaluate });
    const accepted = selectAcceptedPolicy(baseline, strongSaved, { evaluate });

    expect(rejected.source).toBe('current');
    expect(rejected.weights).toEqual(baseline);
    expect(accepted.source).toBe('candidate');
    expect(accepted.weights).toEqual(strongSaved);
  });

  it('can evaluate the runtime tactical strategy against the traditional opponent', () => {
    const result = evaluateRuntimePolicy(defaultNeuralWeights(), {
      seed: 5,
      matches: 2,
      frames: 60
    });

    expect(Number.isFinite(result.score)).toBe(true);
    expect(Number.isFinite(result.goalDiff)).toBe(true);
    expect(Number.isFinite(result.ballProgress)).toBe(true);
    expect(result.goalsFor).toBeGreaterThanOrEqual(0);
    expect(result.goalsAgainst).toBeGreaterThanOrEqual(0);
    expect(result.winProxy).toBeGreaterThanOrEqual(0);
    expect(result.winProxy).toBeLessThanOrEqual(1);
  });

  it('can trace runtime decisions without changing gate scoring', () => {
    const options = {
      seeds: [5, 7],
      matches: 2,
      frames: 60
    };
    const traced = traceRuntimePolicy(defaultNeuralWeights(), options);
    const seed5 = evaluateRuntimePolicy(defaultNeuralWeights(), {
      seed: 5,
      matches: 2,
      frames: 60
    });
    const seed7 = evaluateRuntimePolicy(defaultNeuralWeights(), {
      seed: 7,
      matches: 2,
      frames: 60
    });

    expect(traced.seeds).toHaveLength(2);
    expect(traced.decisions).toBeGreaterThan(0);
    expect(traced.policyActionCounts).toHaveLength(POLICY_ACTION_COUNT);
    expect(traced.tacticalActionCounts).toHaveLength(POLICY_ACTION_COUNT);
    expect(traced.finalActionCounts).toHaveLength(POLICY_ACTION_COUNT);
    expect(sum(traced.finalActionCounts)).toBe(traced.decisions);
    expect(traced.goalsFor).toBe(seed5.goalsFor + seed7.goalsFor);
    expect(traced.goalsAgainst).toBe(seed5.goalsAgainst + seed7.goalsAgainst);
    expect(traced.score).toBeCloseTo((seed5.score + seed7.score) / 2, 9);
    expect(traced.averageStamina).toBeGreaterThanOrEqual(0);
    expect(traced.averageStamina).toBeLessThanOrEqual(1);
  });

  it('parses split seed lists for runtime trace diagnostics', () => {
    const options = parseTraceRuntimePolicyArgs([
      '--seeds',
      '83',
      '97',
      '109',
      '127',
      '149',
      '--matches',
      '4'
    ]);

    expect(options.seeds).toEqual([83, 97, 109, 127, 149]);
    expect(options.matches).toBe(4);
  });

  it('writes decision-level policy visibility analysis from the trace CLI', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-trace-'));
    const currentPath = join(workdir, 'current.json');
    const candidatePath = join(workdir, 'candidate.json');
    const outputPath = join(workdir, 'trace.json');
    writeFileSync(currentPath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');
    writeFileSync(candidatePath, JSON.stringify({ weights: defaultNeuralWeights() }), 'utf8');

    const traces = runTraceRuntimePolicy({
      ...parseTraceRuntimePolicyArgs([]),
      currentPath,
      candidatePath,
      outputPath,
      seeds: [5],
      matches: 1,
      frames: 30,
      decisionAnalysis: true
    });

    expect(traces).toHaveLength(2);
    expect(existsSync(outputPath)).toBe(true);
    const payload = JSON.parse(readFileSync(outputPath, 'utf8')) as {
      decisionAnalysis?: {
        currentDecisionCount: number;
        candidateDecisionCount: number;
        comparison: { comparedDecisions: number; seeds: Array<{ seed: number }> };
      };
    };
    expect(payload.decisionAnalysis?.currentDecisionCount).toBeGreaterThan(0);
    expect(payload.decisionAnalysis?.candidateDecisionCount).toBe(payload.decisionAnalysis?.currentDecisionCount);
    expect(payload.decisionAnalysis?.comparison.comparedDecisions).toBe(payload.decisionAnalysis?.currentDecisionCount);
    expect(payload.decisionAnalysis?.comparison.seeds).toEqual([expect.objectContaining({ seed: 5 })]);
  });

  it('exports low-pressure forward-loss decision states as policy anchors', () => {
    const currentInputs = Array.from({ length: 36 }, (_, index) => index / 36);
    const anchors = buildPolicyAnchorSamples({
      comparedDecisions: 1,
      alignedComparedDecisions: 1,
      afterFinalActionDivergenceComparedDecisions: 0,
      missingCurrentDecisions: 0,
      missingCandidateDecisions: 0,
      rawPolicyChanges: 1,
      alignedRawPolicyChanges: 1,
      afterFinalActionDivergenceRawPolicyChanges: 0,
      tacticalActionChanges: 1,
      finalActionChanges: 1,
      lostPolicyChanges: 0,
      alignedLostPolicyChanges: 0,
      afterFinalActionDivergenceLostPolicyChanges: 0,
      lostWithTacticalRollout: 0,
      lostWithStaminaConserve: 0,
      lostWithCriticalStamina: 0,
      lostWithFlatPolicy: 0,
      firstFinalActionDivergences: [{
        seed: 71,
        match: 0,
        controlledTeam: 'red',
        decisionIndex: 0,
        frame: 18,
        currentRawPolicyActionIndex: 8,
        candidateRawPolicyActionIndex: 7,
        currentFinalActionIndex: 8,
        candidateFinalActionIndex: 7,
        staminaRatio: 0.82,
        ballDistance: 220,
        ballSpeed: 20,
        finishingPressure: 0.1,
        ownGoalPressure: 0.05,
        sideWallPressure: 0,
        attackCornerPressure: 0,
        ownCornerPressure: 0
      }],
      seeds: [],
      samples: []
    }, [decisionRecord({
      seed: 71,
      match: 0,
      decisionIndex: 0,
      frame: 18,
      inputs: currentInputs,
      rawPolicyActionIndex: 8,
      finalActionIndex: 8,
      staminaRatio: 0.82,
      finishingPressure: 0.1,
      ownGoalPressure: 0.05,
      attackCornerPressure: 0,
      ownCornerPressure: 0
    })]);

    expect(anchors.samples).toEqual([{
      inputs: currentInputs,
      actionIndex: 8,
      team: 'red',
      seed: 71,
      match: 0,
      frame: 18,
      decisionIndex: 0,
      tags: ['policyAnchor', 'lowPressureForwardLoss'],
      weight: 1
    }]);
  });

  it('can export nearby low-pressure full-forward states around forward-loss divergences', () => {
    const anchors = buildPolicyAnchorSamples({
      comparedDecisions: 1,
      alignedComparedDecisions: 1,
      afterFinalActionDivergenceComparedDecisions: 0,
      missingCurrentDecisions: 0,
      missingCandidateDecisions: 0,
      rawPolicyChanges: 1,
      alignedRawPolicyChanges: 1,
      afterFinalActionDivergenceRawPolicyChanges: 0,
      tacticalActionChanges: 1,
      finalActionChanges: 1,
      lostPolicyChanges: 0,
      alignedLostPolicyChanges: 0,
      afterFinalActionDivergenceLostPolicyChanges: 0,
      lostWithTacticalRollout: 0,
      lostWithStaminaConserve: 0,
      lostWithCriticalStamina: 0,
      lostWithFlatPolicy: 0,
      firstFinalActionDivergences: [{
        seed: 109,
        match: 3,
        controlledTeam: 'blue',
        decisionIndex: 5,
        frame: 30,
        currentRawPolicyActionIndex: 8,
        candidateRawPolicyActionIndex: 5,
        currentFinalActionIndex: 8,
        candidateFinalActionIndex: 5,
        staminaRatio: 0.84,
        ballDistance: 140,
        ballSpeed: 15,
        finishingPressure: 0.08,
        ownGoalPressure: 0.06,
        sideWallPressure: 0,
        attackCornerPressure: 0,
        ownCornerPressure: 0
      }],
      seeds: [],
      samples: []
    }, [
      decisionRecord({
        seed: 109,
        match: 3,
        controlledTeam: 'blue',
        decisionIndex: 3,
        frame: 18,
        inputs: Array.from({ length: 36 }, () => 0.3),
        rawPolicyActionIndex: 8,
        finalActionIndex: 8,
        staminaRatio: 0.9,
        finishingPressure: 0.1,
        ownGoalPressure: 0.1
      }),
      decisionRecord({
        seed: 109,
        match: 3,
        controlledTeam: 'blue',
        decisionIndex: 5,
        frame: 30,
        inputs: Array.from({ length: 36 }, () => 0.5),
        rawPolicyActionIndex: 8,
        finalActionIndex: 8,
        staminaRatio: 0.84,
        finishingPressure: 0.08,
        ownGoalPressure: 0.06
      }),
      decisionRecord({
        seed: 109,
        match: 3,
        controlledTeam: 'blue',
        decisionIndex: 7,
        frame: 42,
        inputs: Array.from({ length: 36 }, () => 0.7),
        rawPolicyActionIndex: 8,
        finalActionIndex: 8,
        staminaRatio: 0.8,
        finishingPressure: 0.15,
        ownGoalPressure: 0.05
      }),
      decisionRecord({
        seed: 109,
        match: 3,
        controlledTeam: 'blue',
        decisionIndex: 8,
        frame: 48,
        inputs: Array.from({ length: 36 }, () => 0.8),
        rawPolicyActionIndex: 8,
        finalActionIndex: 8,
        staminaRatio: 0.8,
        finishingPressure: 0.15,
        ownGoalPressure: 0.05
      })
    ], { neighborRadius: 2 });

    expect(anchors.samples.map((sample) => sample.decisionIndex)).toEqual([3, 5, 7]);
    expect(anchors.samples.map((sample) => sample.tags)).toEqual([
      ['policyAnchor', 'lowPressureForwardLossNeighbor'],
      ['policyAnchor', 'lowPressureForwardLoss'],
      ['policyAnchor', 'lowPressureForwardLossNeighbor']
    ]);
  });

  it('writes exported policy anchors from the trace CLI', () => {
    const workdir = mkdtempSync(join(tmpdir(), 'soccer-runtime-anchor-'));
    const currentPath = join(workdir, 'current.json');
    const candidatePath = join(workdir, 'candidate.json');
    const anchorOutputPath = join(workdir, 'anchors.json');
    const currentWeights = defaultNeuralWeights();
    writeFileSync(currentPath, JSON.stringify({ weights: currentWeights }), 'utf8');
    writeFileSync(candidatePath, JSON.stringify({ weights: currentWeights }), 'utf8');

    const result = runTraceRuntimePolicyDetailed({
      ...parseTraceRuntimePolicyArgs([]),
      currentPath,
      candidatePath,
      anchorOutputPath,
      seeds: [5],
      matches: 1,
      frames: 30,
      decisionAnalysis: true
    });

    expect(result.policyAnchors?.samples.every((sample) => sample.inputs.length === 36)).toBe(true);
    expect(existsSync(anchorOutputPath)).toBe(true);
    const payload = JSON.parse(readFileSync(anchorOutputPath, 'utf8')) as {
      samples: Array<{ inputs: number[]; actionIndex: number; tags: string[]; weight: number }>;
    };
    expect(payload.samples).toEqual(result.policyAnchors?.samples);
  });

  it('parses the trace anchor neighbor radius', () => {
    const options = parseTraceRuntimePolicyArgs([
      '--anchor-neighbor-radius',
      '4'
    ]);

    expect(options.anchorNeighborRadius).toBe(4);
  });

  it('compares runtime traces as behavior-visibility deltas', () => {
    const current = traceSummary({
      decisions: 100,
      finalActionCounts: [50, 50, 0, 0, 0, 0, 0, 0, 0],
      policyActionCounts: [40, 60, 0, 0, 0, 0, 0, 0, 0],
      tacticalActionCounts: [45, 55, 0, 0, 0, 0, 0, 0, 0],
      tacticalRolloutChanges: 20,
      staminaConserves: 10,
      criticalStaminaRegulations: 5
    });
    const candidate = traceSummary({
      decisions: 100,
      finalActionCounts: [40, 55, 5, 0, 0, 0, 0, 0, 0],
      policyActionCounts: [35, 60, 5, 0, 0, 0, 0, 0, 0],
      tacticalActionCounts: [35, 60, 5, 0, 0, 0, 0, 0, 0],
      tacticalRolloutChanges: 30,
      staminaConserves: 8,
      criticalStaminaRegulations: 9
    });

    const delta = compareRuntimeTraces(candidate, current);

    expect(delta.finalActionCounts).toEqual([-10, 5, 5, 0, 0, 0, 0, 0, 0]);
    expect(delta.finalActionDistributionChangeCount).toBe(10);
    expect(delta.finalActionDistributionChangeRate).toBe(0.1);
    expect(delta.policyActionDistributionChangeCount).toBe(5);
    expect(delta.tacticalActionDistributionChangeCount).toBe(10);
    expect(delta.tacticalRolloutChangeRate).toBeCloseTo(0.1, 9);
    expect(delta.staminaConserveRate).toBeCloseTo(-0.02, 9);
    expect(delta.criticalStaminaRegulationRate).toBeCloseTo(0.04, 9);
  });

  it('collects seed-level runtime decision records for localizing wrapper-hidden changes', () => {
    const traced = traceRuntimePolicyDecisions(defaultNeuralWeights(), {
      seeds: [5],
      matches: 2,
      frames: 60
    });

    expect(traced.summary.decisions).toBeGreaterThan(0);
    expect(traced.decisions).toHaveLength(traced.summary.decisions);
    expect(traced.summary.seeds).toHaveLength(1);
    expect(traced.decisions[0]).toMatchObject({
      seed: 5,
      match: 0,
      controlledTeam: 'red',
      decisionIndex: 0
    });
    expect(new Set(traced.decisions.map((record) => record.match))).toEqual(new Set([0, 1]));
    expect(traced.decisions.every((record) => record.seed === 5)).toBe(true);
  });

  it('includes network inputs in runtime decision records for state-anchor exports', () => {
    const traced = traceRuntimePolicyDecisions(defaultNeuralWeights(), {
      seeds: [5],
      matches: 1,
      frames: 30
    });

    expect(traced.decisions[0].inputs).toHaveLength(36);
    expect(traced.decisions[0].inputs.every((value) => Number.isFinite(value))).toBe(true);
  });

  it('identifies policy argmax changes hidden by tactical rollout and stamina guards per seed', () => {
    const current = [
      decisionRecord({
        seed: 97,
        decisionIndex: 0,
        rawPolicyActionIndex: 1,
        policyActionIndex: 1,
        tacticalActionIndex: 4,
        finalActionIndex: 4,
        tacticalActionScores: [0, 0.1, 0, 0, 0.4, 0, 0, 0, 0],
        tacticalRolloutChanged: true
      }),
      decisionRecord({
        seed: 97,
        decisionIndex: 1,
        rawPolicyActionIndex: 2,
        policyActionIndex: 2,
        tacticalActionIndex: 2,
        finalActionIndex: 4,
        criticalStaminaRegulated: true
      }),
      decisionRecord({
        seed: 109,
        decisionIndex: 0,
        rawPolicyActionIndex: 7,
        policyActionIndex: 7,
        tacticalActionIndex: 7,
        finalActionIndex: 7
      })
    ];
    const candidate = [
      decisionRecord({
        seed: 97,
        decisionIndex: 0,
        rawPolicyActionIndex: 2,
        policyActionIndex: 2,
        tacticalActionIndex: 4,
        finalActionIndex: 4,
        tacticalActionScores: [0, 0, 0.2, 0, 0.45, 0, 0, 0, 0],
        tacticalRolloutChanged: true
      }),
      decisionRecord({
        seed: 97,
        decisionIndex: 1,
        rawPolicyActionIndex: 3,
        policyActionIndex: 3,
        tacticalActionIndex: 3,
        finalActionIndex: 4,
        criticalStaminaRegulated: true
      }),
      decisionRecord({
        seed: 109,
        decisionIndex: 0,
        rawPolicyActionIndex: 7,
        policyActionIndex: 7,
        tacticalActionIndex: 7,
        finalActionIndex: 8
      })
    ];

    const comparison = compareRuntimeDecisionTraces(candidate, current);

    expect(comparison.comparedDecisions).toBe(3);
    expect(comparison.rawPolicyChanges).toBe(2);
    expect(comparison.finalActionChanges).toBe(1);
    expect(comparison.lostPolicyChanges).toBe(2);
    expect(comparison.lostWithTacticalRollout).toBe(1);
    expect(comparison.lostWithCriticalStamina).toBe(1);
    expect(comparison.lostWithStaminaConserve).toBe(0);
    expect(comparison.seeds).toEqual([
      expect.objectContaining({
        seed: 97,
        comparedDecisions: 2,
        rawPolicyChanges: 2,
        finalActionChanges: 0,
        lostPolicyChanges: 2,
        lostWithTacticalRollout: 1,
        lostWithCriticalStamina: 1
      }),
      expect.objectContaining({
        seed: 109,
        comparedDecisions: 1,
        rawPolicyChanges: 0,
        finalActionChanges: 1,
        lostPolicyChanges: 0
      })
    ]);
    expect(comparison.samples[0]).toMatchObject({
      seed: 97,
      decisionIndex: 0,
      finalActionIndex: 4,
      currentRawPolicyActionIndex: 1,
      candidateRawPolicyActionIndex: 2,
      currentTacticalActionScores: [0, 0.1, 0, 0, 0.4, 0, 0, 0, 0],
      candidateTacticalActionScores: [0, 0, 0.2, 0, 0.45, 0, 0, 0, 0],
      afterFinalActionDivergence: false,
      reasons: ['tactical-rollout']
    });
  });

  it('separates aligned decision changes from post-divergence comparisons', () => {
    const current = [
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 0,
        rawPolicyActionIndex: 1,
        finalActionIndex: 4,
        tacticalRolloutChanged: true
      }),
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 1,
        rawPolicyActionIndex: 1,
        tacticalActionIndex: 4,
        finalActionIndex: 4,
        staminaRatio: 0.72,
        ballDistance: 120,
        ownGoalPressure: 0.2,
        finishingPressure: 0.4
      }),
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 2,
        rawPolicyActionIndex: 2,
        finalActionIndex: 4,
        tacticalRolloutChanged: true
      }),
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 3,
        rawPolicyActionIndex: 2,
        finalActionIndex: 4
      }),
      decisionRecord({
        seed: 109,
        match: 1,
        decisionIndex: 0,
        rawPolicyActionIndex: 5,
        finalActionIndex: 4,
        tacticalRolloutChanged: true
      })
    ];
    const candidate = [
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 0,
        rawPolicyActionIndex: 2,
        finalActionIndex: 4,
        tacticalRolloutChanged: true
      }),
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 1,
        rawPolicyActionIndex: 1,
        tacticalActionIndex: 5,
        tacticalActionScores: [0, 0.1, 0, 0, 0.2, 0.4, 0, 0, 0],
        finalActionIndex: 5,
        staminaRatio: 0.68,
        ballDistance: 132,
        ownGoalPressure: 0.25,
        finishingPressure: 0.35
      }),
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 2,
        rawPolicyActionIndex: 3,
        finalActionIndex: 4,
        tacticalRolloutChanged: true
      }),
      decisionRecord({
        seed: 109,
        match: 0,
        decisionIndex: 3,
        rawPolicyActionIndex: 3,
        finalActionIndex: 5
      }),
      decisionRecord({
        seed: 109,
        match: 1,
        decisionIndex: 0,
        rawPolicyActionIndex: 7,
        finalActionIndex: 4,
        tacticalRolloutChanged: true
      })
    ];

    const comparison = compareRuntimeDecisionTraces(candidate, current);

    expect(comparison.firstFinalActionDivergences).toEqual([
      expect.objectContaining({
        seed: 109,
        match: 0,
        decisionIndex: 1,
        currentFinalActionIndex: 4,
        candidateFinalActionIndex: 5,
        currentRawPolicyActionIndex: 1,
        candidateRawPolicyActionIndex: 1,
        currentTacticalActionIndex: 4,
        candidateTacticalActionIndex: 5,
        candidateTacticalActionScores: [0, 0.1, 0, 0, 0.2, 0.4, 0, 0, 0],
        staminaRatio: 0.68,
        ballDistance: 132,
        ownGoalPressure: 0.25,
        finishingPressure: 0.35
      })
    ]);
    expect(comparison.alignedComparedDecisions).toBe(3);
    expect(comparison.afterFinalActionDivergenceComparedDecisions).toBe(2);
    expect(comparison.alignedRawPolicyChanges).toBe(2);
    expect(comparison.afterFinalActionDivergenceRawPolicyChanges).toBe(2);
    expect(comparison.alignedLostPolicyChanges).toBe(2);
    expect(comparison.afterFinalActionDivergenceLostPolicyChanges).toBe(1);
    expect(comparison.seeds[0]).toMatchObject({
      seed: 109,
      alignedLostPolicyChanges: 2,
      afterFinalActionDivergenceLostPolicyChanges: 1
    });
    expect(comparison.samples).toEqual([
      expect.objectContaining({ decisionIndex: 0, afterFinalActionDivergence: false }),
      expect.objectContaining({ decisionIndex: 2, afterFinalActionDivergence: true }),
      expect.objectContaining({ match: 1, decisionIndex: 0, afterFinalActionDivergence: false })
    ]);
  });
});

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function traceSummary(overrides: Partial<RuntimeTraceSummary>): RuntimeTraceSummary {
  const actionCounts = Array.from({ length: POLICY_ACTION_COUNT }, () => 0);
  return {
    score: 0,
    goalDiff: 0,
    ballProgress: 0,
    goalsFor: 0,
    goalsAgainst: 0,
    winProxy: 0,
    decisions: 0,
    policyActionCounts: [...actionCounts],
    tacticalActionCounts: [...actionCounts],
    finalActionCounts: [...actionCounts],
    tacticalRolloutUses: 0,
    tacticalRolloutChanges: 0,
    staminaConserves: 0,
    criticalStaminaRegulations: 0,
    flatPolicies: 0,
    averageStamina: 0,
    averageBallDistance: 0,
    averageBallSpeed: 0,
    averageFinishingPressure: 0,
    averageOwnGoalPressure: 0,
    averageSideWallPressure: 0,
    averageAttackCornerPressure: 0,
    averageOwnCornerPressure: 0,
    seeds: [],
    ...overrides
  };
}

function decisionRecord(overrides: Partial<RuntimeDecisionTraceRecord>): RuntimeDecisionTraceRecord {
  return {
    seed: 97,
    match: 0,
    controlledTeam: 'red',
    decisionIndex: 0,
    frame: 0,
    team: 'red',
    tankId: 'red-0',
    inputs: Array.from({ length: 36 }, () => 0),
    staminaRatio: 1,
    ballDistance: 100,
    ballSpeed: 0,
    finishingPressure: 0,
    ownGoalPressure: 0,
    sideWallPressure: 0,
    attackCornerPressure: 0,
    ownCornerPressure: 0,
    tacticalActionScores: undefined,
    finalActionIndex: 4,
    tacticalRolloutUsed: false,
    tacticalRolloutChanged: false,
    staminaConserved: false,
    criticalStaminaRegulated: false,
    flatPolicy: false,
    ...overrides
  };
}
