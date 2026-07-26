import { readFileSync, writeFileSync } from 'node:fs';
import {
  advanceRuntimeOpponentLeague,
  parseRuntimeOpponentLeagueConfig
} from '../src/ai/runtimeOpponentLeague';

declare const process: {
  argv: string[];
  exitCode?: number;
};

export function main(argv: readonly string[] = process.argv.slice(2)): void {
  try {
    const configPath = stringArg(argv, '--config') ?? 'config/runtime-opponent-league.json';
    const current = parseRuntimeOpponentLeagueConfig(
      JSON.parse(readFileSync(configPath, 'utf8')) as unknown
    );
    const next = advanceRuntimeOpponentLeague(current);
    const payload = `${JSON.stringify(next, null, 2)}\n`;

    if (argv.includes('--write')) {
      writeFileSync(configPath, payload, 'utf8');
      console.log(`advanced runtime opponent league ${current.generation} -> ${next.generation}: ${configPath}`);
    } else {
      console.log(payload.trimEnd());
      console.log('preview only; pass --write after an AI promotion is accepted');
    }
  } catch (error) {
    process.exitCode = 1;
    console.error(error instanceof Error ? error.message : String(error));
  }
}

function stringArg(argv: readonly string[], name: string): string | undefined {
  const index = argv.indexOf(name);
  const value = index >= 0 ? argv[index + 1] : undefined;
  return value && !value.startsWith('--') ? value : undefined;
}

if (process.argv[1]?.replace(/\\/g, '/').endsWith('/advance-runtime-league.ts') ||
  process.argv[1]?.replace(/\\/g, '/').endsWith('/advance-runtime-league.js')) {
  main();
}
