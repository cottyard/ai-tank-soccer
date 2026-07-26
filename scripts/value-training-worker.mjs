import { parentPort, workerData } from 'node:worker_threads';
import { register } from 'tsx/esm/api';

// Worker threads do not inherit the tsx ESM loader from the parent process.
register();

const { generateSampleShard } = await import('./train-value-network.ts');

parentPort.postMessage(generateSampleShard(workerData));
