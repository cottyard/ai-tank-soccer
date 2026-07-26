import { readFileSync } from 'node:fs';
import {
  evaluateRuntimeOpponentLeague,
  parseRuntimeOpponentLeagueConfig,
  type RuntimeOpponentLeagueRow
} from '../src/ai/runtimeOpponentLeague';
import { loadWeightsPayload } from './coach-neural';

declare const process: {
  argv: string[];
  exitCode?: number;
};

type RuntimeLeagueCliOptions = {
  configPath: string;
  candidatePath: string;
  acceptedPath: string;
  matches: number;
  frames: number;
  details: boolean;
  opponents: string[];
};

export function parseRuntimeLeagueArgs(argv: readonly string[]): RuntimeLeagueCliOptions {
  return {
    configPath: stringArg(argv, '--config') ?? 'config/runtime-opponent-league.json',
    candidatePath: stringArg(argv, '--candidate') ?? 'public/models/neural-best.json',
    acceptedPath: stringArg(argv, '--accepted') ?? 'public/models/neural-best.json',
    matches: positiveIntegerArg(argv, '--matches', 4),
    frames: positiveIntegerArg(argv, '--frames', 600),
    details: argv.includes('--details'),
    opponents: listArg(argv, '--opponents')
  };
}

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const options = parseRuntimeLeagueArgs(argv);
    const config = parseRuntimeOpponentLeagueConfig(
      JSON.parse(readFileSync(options.configPath, 'utf8')) as unknown
    );
    const candidate = loadWeightsPayload(readFileSync(options.candidatePath, 'utf8'));
    const accepted = loadWeightsPayload(readFileSync(options.acceptedPath, 'utf8'));
    const selected = options.opponents.length === 0
      ? config
      : {
          ...config,
          opponents: config.opponents.filter((opponent) => options.opponents.includes(opponent.id))
        };
    if (selected.opponents.length === 0) {
      throw new Error(`No configured opponents matched: ${options.opponents.join(',')}`);
    }
    const result = evaluateRuntimeOpponentLeague(candidate, accepted, selected, options);

    console.log(
      `runtime-opponent-league generation=${result.generation} matches=${options.matches} frames=${options.frames}`
    );
    for (const row of result.rows) {
      console.log(formatLeagueRow(row));
      if (options.details) {
        for (const seed of row.seeds) {
          console.log(
            `  seed=${seed.seed} goals=${seed.goalsFor}-${seed.goalsAgainst} ` +
            `score=${seed.score.toFixed(3)} win=${seed.winProxy.toFixed(3)} bp=${seed.ballProgress.toFixed(3)}`
          );
        }
      }
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function formatLeagueRow(row: RuntimeOpponentLeagueRow): string {
  const result = row.result;
  return [
    row.lifecycle,
    row.id,
    `kind=${row.kind}`,
    `seeds=${row.seeds.map((seed) => seed.seed).join(',')}`,
    `goals=${result.goalsFor}-${result.goalsAgainst}`,
    `avgScore=${result.score.toFixed(3)}`,
    `avgWin=${result.winProxy.toFixed(3)}`,
    `avgBp=${result.ballProgress.toFixed(3)}`
  ].join(' ');
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value && !value.startsWith('--') ? value : undefined;
}

function positiveIntegerArg(argv: readonly string[], name: string, fallback: number): number {
  const value = Number(stringArg(argv, name));
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback;
}

function listArg(argv: readonly string[], name: string): string[] {
  const index = argv.indexOf(name);
  if (index < 0) {
    return [];
  }
  const values: string[] = [];
  for (let cursor = index + 1; cursor < argv.length; cursor += 1) {
    const value = argv[cursor];
    if (value.startsWith('--')) {
      break;
    }
    values.push(...value.split(',').map((part) => part.trim()).filter(Boolean));
  }
  return values;
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/evaluate-runtime-league.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/evaluate-runtime-league.js')) {
  main();
}
