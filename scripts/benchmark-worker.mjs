import { parentPort, workerData } from 'node:worker_threads';
import { register } from 'tsx/esm/api';

// Worker threads do not inherit the tsx ESM loader from the parent process, so
// register it here before importing any TypeScript module.
register();

const { runBenchmarkShard } = await import('./benchmark-runtime.ts');

parentPort.postMessage(runBenchmarkShard(workerData));
