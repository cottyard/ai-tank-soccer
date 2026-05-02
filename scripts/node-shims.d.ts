declare module 'node:fs' {
  export function appendFileSync(path: string, data: string, encoding: string): void;
  export function existsSync(path: string): boolean;
  export function mkdirSync(path: string, options?: { recursive?: boolean }): void;
  export function mkdtempSync(prefix: string): string;
  export function readFileSync(path: string, encoding: string): string;
  export function writeFileSync(path: string, data: string, encoding: string): void;
  export function unlinkSync(path: string): void;
}

declare module 'node:child_process' {
  type SpawnResult = {
    status: number | null;
  };

  export function execFileSync(
    file: string,
    args?: readonly string[],
    options?: Record<string, unknown>
  ): Buffer;

  export function spawnSync(
    file: string,
    args?: readonly string[],
    options?: Record<string, unknown>
  ): SpawnResult;
}

declare module 'node:os' {
  export function tmpdir(): string;
}

declare module 'node:path' {
  export function dirname(path: string): string;
  export function join(...paths: string[]): string;
}

declare const process: {
  argv: string[];
  env: Record<string, string | undefined>;
  cwd(): string;
  exitCode?: number;
};
