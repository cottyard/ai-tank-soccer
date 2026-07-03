import {
  Activity,
  createIcons,
  Gauge,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Sparkles,
  StepForward,
  Target
} from 'lucide';
import './styles.css';
import { BUNDLED_POLICY_VERSION, loadBundledPolicy } from './ai/bundledPolicy';
import {
  LearningModeController,
  clearLearnedPolicy,
  loadLearnedPolicy,
  loadLearningReplay,
  saveLearnedPolicy,
  saveLearningReplay,
  serializeReplayExport
} from './ai/learningMode';
import { createAsyncNeuralStrategy } from './ai/asyncNeuralStrategy';
import { evaluatePolicyGate, selectAcceptedPolicy } from './ai/policyGate';
import { trainSelfPlayPolicy } from './ai/selfPlayTraining';
import { traditionalStrategy } from './ai/traditionalStrategy';
import { defaultNeuralWeights } from './ai/neuralWeights';
import { humanCommandForTeam, HUMAN_CONTROL_SCHEMES } from './game/manualControl';
import { createInitialState, type Team } from './game/model';
import { AI_HZ, FIXED_DT, PHYSICS_HZ } from './game/match';
import { stepGame } from './game/simulation';
import { AiClock, idleCommands, type CommandMap, type Strategy } from './game/strategy';
import { FieldRenderer } from './ui/renderer';

type ControlMode = 'idle' | 'traditional' | 'neural' | 'human';

type ControlSelections = Record<Team, ControlMode>;

type DisposableStrategy = Strategy & {
  dispose?: () => void;
};

type StrategyPair = {
  red: DisposableStrategy;
  blue: DisposableStrategy;
};

const idleStrategy: Strategy = {
  name: 'idle',
  decide: idleCommands
};

const MODE_LABELS: Record<ControlMode, string> = {
  idle: 'Idle',
  traditional: 'Traditional AI',
  neural: 'Neural AI',
  human: 'Human'
};

const BUNDLED_POLICY_VERSION_KEY = 'tank-soccer-bundled-policy-version-v1';
const MAX_FRAME_STEPS_PER_RENDER = 5;

const HUMAN_CONTROL_KEYS = new Set(
  Object.values(HUMAN_CONTROL_SCHEMES).flatMap((scheme) => [
    scheme.left.forward,
    scheme.left.backward,
    scheme.right.forward,
    scheme.right.backward
  ])
);

const app = document.querySelector<HTMLDivElement>('#app');
if (!app) {
  throw new Error('Missing app root');
}

app.innerHTML = `
  <main class="app-shell">
    <header class="topbar">
      <div class="strategy-name red" id="red-name"></div>
      <div class="scoreboard" aria-label="Scoreboard">
        <span class="score red" id="red-score">0</span>
        <span class="clock" id="clock">00:00</span>
        <span class="score blue" id="blue-score">0</span>
      </div>
      <div class="strategy-name blue" id="blue-name"></div>
    </header>
    <section class="workspace">
      <div class="field-stage">
        <canvas id="game-canvas"></canvas>
        <div class="status-strip">
          <span class="chip"><i data-lucide="activity"></i><span id="run-state">Running</span></span>
          <span class="chip"><i data-lucide="gauge"></i><span>${PHYSICS_HZ} FPS logic</span></span>
          <span class="chip"><i data-lucide="target"></i><span>${AI_HZ} Hz AI</span></span>
        </div>
      </div>
      <aside class="side-panel">
        <section>
          <h2 class="section-title">Match</h2>
          <div class="button-row">
            <button class="icon-button" id="toggle-run" type="button" title="Pause">
              <i data-lucide="pause"></i>
            </button>
            <button class="icon-button" id="step" type="button" title="Step one frame">
              <i data-lucide="step-forward"></i>
            </button>
            <button class="icon-button" id="reset" type="button" title="Reset">
              <i data-lucide="rotate-ccw"></i>
            </button>
          </div>
          <div class="toggle-row">
            <label class="toggle">
              <span>Ball trail</span>
              <input id="toggle-trails" type="checkbox" checked />
            </label>
          </div>
          <div class="learning-actions">
            <button class="wide-button learn" id="learning-mode" type="button">
              <i data-lucide="sparkles"></i>
              <span>Learning Mode</span>
            </button>
            <div class="split-row">
              <button class="wide-button" id="train-replay" type="button">Deep Replay</button>
              <button class="wide-button" id="train-self-play" type="button">Self Play</button>
            </div>
            <button class="wide-button" id="export-replay" type="button">Export Replay</button>
            <button class="wide-button" id="reset-learned" type="button">Reset Model</button>
          </div>
        </section>
        <section>
          <h2 class="section-title">Control</h2>
          <div class="control-stack">
            <label class="team-mode">
              <span class="team-label red">Red</span>
              <select class="control-select" id="red-control-mode">
                <option value="human">Human</option>
                <option value="idle">Idle</option>
                <option value="traditional">Traditional AI</option>
                <option value="neural">Neural AI</option>
              </select>
            </label>
            <label class="team-mode">
              <span class="team-label blue">Blue</span>
              <select class="control-select" id="blue-control-mode">
                <option value="idle">Idle</option>
                <option value="human">Human</option>
                <option value="traditional">Traditional AI</option>
                <option value="neural">Neural AI</option>
              </select>
            </label>
            <div class="manual-keys" aria-label="Human control keys">
              <div class="key-row red">
                <span>Red</span>
                <span><kbd>Q</kbd>/<kbd>A</kbd> left · <kbd>W</kbd>/<kbd>S</kbd> right</span>
              </div>
              <div class="key-row blue">
                <span>Blue</span>
                <span><kbd>P</kbd>/<kbd>;</kbd> left · <kbd>[</kbd>/<kbd>'</kbd> right</span>
              </div>
            </div>
          </div>
        </section>
        <section>
          <h2 class="section-title">Telemetry</h2>
          <div class="metrics">
            <div class="metric">
              <span class="metric-label">Ball speed</span>
              <span class="metric-value" id="ball-speed">0</span>
            </div>
            <div class="metric">
              <span class="metric-label">Frame</span>
              <span class="metric-value" id="frame-count">0</span>
            </div>
            <div class="metric">
              <span class="metric-label">Red zone</span>
              <span class="metric-value" id="red-zone">50%</span>
            </div>
            <div class="metric">
              <span class="metric-label">Blue zone</span>
              <span class="metric-value" id="blue-zone">50%</span>
            </div>
            <div class="metric">
              <span class="metric-label">Samples</span>
              <span class="metric-value" id="learn-samples">0</span>
            </div>
            <div class="metric">
              <span class="metric-label">Replay</span>
              <span class="metric-value" id="replay-samples">0</span>
            </div>
            <div class="metric">
              <span class="metric-label">Loss</span>
              <span class="metric-value" id="learn-loss">0.000</span>
            </div>
            <div class="metric">
              <span class="metric-label">Model</span>
              <span class="metric-value" id="learn-version">v0</span>
            </div>
            <div class="metric">
              <span class="metric-label">Learning</span>
              <span class="metric-value" id="learn-state">Off</span>
            </div>
          </div>
        </section>
        <section class="log" id="event-log" aria-label="Event log"></section>
      </aside>
    </section>
  </main>
`;

createIcons({
  icons: {
    Activity,
    Gauge,
    Pause,
    Play,
    RefreshCw,
    RotateCcw,
    Sparkles,
    StepForward,
    Target
  }
});

const canvas = requiredElement<HTMLCanvasElement>('game-canvas');
const renderer = new FieldRenderer(canvas);
const toggleRun = requiredElement<HTMLButtonElement>('toggle-run');
const stepButton = requiredElement<HTMLButtonElement>('step');
const resetButton = requiredElement<HTMLButtonElement>('reset');
const learningModeButton = requiredElement<HTMLButtonElement>('learning-mode');
const trainReplayButton = requiredElement<HTMLButtonElement>('train-replay');
const trainSelfPlayButton = requiredElement<HTMLButtonElement>('train-self-play');
const exportReplayButton = requiredElement<HTMLButtonElement>('export-replay');
const resetLearnedButton = requiredElement<HTMLButtonElement>('reset-learned');
const trailsToggle = requiredElement<HTMLInputElement>('toggle-trails');
const redControlMode = requiredElement<HTMLSelectElement>('red-control-mode');
const blueControlMode = requiredElement<HTMLSelectElement>('blue-control-mode');
const runState = requiredElement<HTMLSpanElement>('run-state');
const redName = requiredElement<HTMLDivElement>('red-name');
const blueName = requiredElement<HTMLDivElement>('blue-name');
const redScore = requiredElement<HTMLSpanElement>('red-score');
const blueScore = requiredElement<HTMLSpanElement>('blue-score');
const clockLabel = requiredElement<HTMLSpanElement>('clock');
const ballSpeed = requiredElement<HTMLSpanElement>('ball-speed');
const frameCount = requiredElement<HTMLSpanElement>('frame-count');
const redZone = requiredElement<HTMLSpanElement>('red-zone');
const blueZone = requiredElement<HTMLSpanElement>('blue-zone');
const learnSamples = requiredElement<HTMLSpanElement>('learn-samples');
const replaySamples = requiredElement<HTMLSpanElement>('replay-samples');
const learnLoss = requiredElement<HTMLSpanElement>('learn-loss');
const learnVersion = requiredElement<HTMLSpanElement>('learn-version');
const learnState = requiredElement<HTMLSpanElement>('learn-state');
const eventLog = requiredElement<HTMLElement>('event-log');

const savedPolicy = loadLearnedPolicy(window.localStorage);
const savedReplay = loadLearningReplay(window.localStorage);
const hasObsoleteSavedPolicy =
  !savedPolicy &&
  window.localStorage.getItem('tank-soccer-neural-policy-v1') !== null;
const startingWeights = savedPolicy?.weights ?? defaultNeuralWeights();
let resetWeights = [...startingWeights];
const learning = new LearningModeController(startingWeights, {
  replaySamples: savedReplay
});
let neuralWeights = learning.currentWeights;
let controlModes: ControlSelections = { red: 'human', blue: 'idle' };
let strategies: StrategyPair = makeStrategyPair(controlModes);
let state = createInitialState();
let aiClock = new AiClock(strategies.red, strategies.blue, PHYSICS_HZ, AI_HZ);
let commands: CommandMap = {};
let pressedKeys = new Set<string>();
let running = true;
let accumulator = 0;
let lastTimestamp = performance.now();
let lastLoggedGoalFrame = -1;
let neuralWorkerErrorLogged = false;
let learningPersistTimer: number | undefined;

const resizeObserver = new ResizeObserver(() => {
  renderer.resize();
  render();
});
resizeObserver.observe(canvas);
window.addEventListener('resize', () => {
  renderer.resize();
  render();
});

toggleRun.addEventListener('click', () => {
  running = !running;
  setRunButton();
  render();
});

stepButton.addEventListener('click', () => {
  running = false;
  setRunButton();
  advanceFrame();
  render();
});

resetButton.addEventListener('click', () => {
  resetMatch();
});

learningModeButton.addEventListener('click', () => {
  startLearningMode();
});

trainReplayButton.addEventListener('click', async () => {
  trainReplayButton.disabled = true;
  render();
  await nextFrame();
  try {
    const result = learning.trainReplay({
      epochs: 180,
      batchSize: 64,
      learningRate: 0.024,
      seed: state.frame + learning.snapshot.modelVersion * 101
    });
    neuralWeights = learning.currentWeights;
    persistLearningState();
    appendLog('Training', `Deep replay trained ${result.trainedSamples} steps across ${result.epochs ?? 0} epochs and ${result.batches ?? 0} batches from ${learning.snapshot.samples} stored samples, loss ${result.loss.toFixed(3)}.`);
  } finally {
    trainReplayButton.disabled = false;
    render();
  }
});

trainSelfPlayButton.addEventListener('click', async () => {
  trainSelfPlayButton.disabled = true;
  render();
  await nextFrame();
  try {
    const result = trainSelfPlayPolicy({
      weights: neuralWeights,
      matches: 18,
      frames: PHYSICS_HZ * 35,
      epochs: 48,
      batchSize: 64,
      learningRate: 0.018,
      seed: state.frame + learning.snapshot.modelVersion * 313
    });
    const gate = evaluatePolicyGate(neuralWeights, result.weights, {
      seed: state.frame + learning.snapshot.modelVersion * 313,
      matches: 2,
      frames: PHYSICS_HZ * 12,
      opponent: traditionalStrategy,
      minDelta: 2
    });
    if (gate.accepted) {
      learning.applyTrainingWeights(result.weights, result.loss);
      neuralWeights = learning.currentWeights;
      persistLearningState();
    }
    appendLog(
      'Self play',
      `Generated ${result.samples} decisions over ${result.frames} frames (${result.redGoals}-${result.blueGoals}), ` +
        `${gate.accepted ? 'accepted' : 'rejected'} candidate ${gate.candidateScore.toFixed(1)} vs ${gate.currentScore.toFixed(1)}, loss ${result.loss.toFixed(3)}.`
    );
  } finally {
    trainSelfPlayButton.disabled = false;
    render();
  }
});

exportReplayButton.addEventListener('click', () => {
  const snapshot = learning.snapshot;
  if (snapshot.samples === 0) {
    appendLog('Replay', 'No replay samples recorded yet.');
    return;
  }

  const exportedAt = new Date().toISOString();
  downloadTextFile(
    `browser-replay-${exportedAt.replace(/[:.]/g, '-')}.json`,
    serializeReplayExport(snapshot, learning.replaySamples, {
      exportedAt,
      origin: window.location.origin
    })
  );
  appendLog('Replay', `Exported ${snapshot.samples} replay samples.`);
});

resetLearnedButton.addEventListener('click', () => {
  cancelScheduledLearningPersist();
  clearLearnedPolicy(window.localStorage);
  window.localStorage.removeItem(BUNDLED_POLICY_VERSION_KEY);
  learning.reset(resetWeights);
  neuralWeights = learning.currentWeights;
  rebuildStrategies();
  appendLog('Training', 'Learned model and replay reset to bundled baseline.');
  render();
});

redControlMode.addEventListener('change', () => {
  setControlMode('red', readControlMode(redControlMode));
});

blueControlMode.addEventListener('change', () => {
  setControlMode('blue', readControlMode(blueControlMode));
});

trailsToggle.addEventListener('change', () => {
  render();
});

window.addEventListener('keydown', (event) => {
  if (isFormControl(event.target)) {
    return;
  }

  const key = normalizeKey(event.key);
  if (!HUMAN_CONTROL_KEYS.has(key)) {
    return;
  }

  pressedKeys.add(key);
  event.preventDefault();
});

window.addEventListener('keyup', (event) => {
  const key = normalizeKey(event.key);
  if (HUMAN_CONTROL_KEYS.has(key)) {
    pressedKeys.delete(key);
    event.preventDefault();
  }
});

window.addEventListener('blur', () => {
  pressedKeys.clear();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    pressedKeys.clear();
  }
});

window.addEventListener('beforeunload', () => {
  if (learningPersistTimer !== undefined) {
    cancelScheduledLearningPersist();
    persistLearningState();
  }
  disposeStrategyPair(strategies);
});

syncControlSelects();
setRunButton();
appendLog('Kickoff', savedPolicy ? 'Loaded learned neural model.' : '1v1 controls ready. AI iteration is paused.');
if (hasObsoleteSavedPolicy) {
  appendLog('Training', 'Saved neural model used an older network shape; using upgraded default weights while keeping replay samples.');
}
if (savedReplay.length > 0) {
  appendLog('Replay', `Loaded ${savedReplay.length} replay samples.`);
}
renderer.resize();
void hydrateBundledPolicy();
requestAnimationFrame(loop);

function loop(timestamp: number): void {
  const elapsed = Math.min(0.25, (timestamp - lastTimestamp) / 1000);
  lastTimestamp = timestamp;

  if (running) {
    accumulator += elapsed;
    let steps = 0;
    while (accumulator >= FIXED_DT && steps < MAX_FRAME_STEPS_PER_RENDER) {
      advanceFrame();
      accumulator -= FIXED_DT;
      steps += 1;
    }
    if (accumulator >= FIXED_DT) {
      accumulator = 0;
    }
  } else {
    accumulator = 0;
  }

  render();
  requestAnimationFrame(loop);
}

function advanceFrame(): void {
  const aiTick = state.frame % Math.max(1, Math.round(PHYSICS_HZ / AI_HZ)) === 0;
  const humanCommands = humanCommandsForActiveTeams();
  commands = {
    ...aiClock.update(state),
    ...humanCommands
  };

  if (aiTick && controlModes.red === 'human' && controlModes.blue === 'neural') {
    const result = learning.recordAiTick(state, 'red', humanCommands['red-0'] ?? { leftTrack: 0, rightTrack: 0 });
    if (result.trainedSamples > 0) {
      neuralWeights = learning.currentWeights;
      scheduleLearningPersist();
    }
  }

  stepGame(state, commands, FIXED_DT);

  if (state.lastGoal && state.lastGoal.frame !== lastLoggedGoalFrame) {
    lastLoggedGoalFrame = state.lastGoal.frame;
    appendLog('Goal', `${capitalize(state.lastGoal.team)} scored at ${formatTime(state.time)}.`);
  }
}

function render(): void {
  const redModeName = teamModeName('red');
  const blueModeName = teamModeName('blue');

  renderer.render(state, {
    redName: redModeName,
    blueName: blueModeName,
    commands,
    showTrails: trailsToggle.checked
  });

  const redControl = clampPercent(100 - (state.ball.position.x / 1050) * 100);
  const blueControl = 100 - redControl;
  redName.textContent = `Red: ${redModeName}`;
  blueName.textContent = `Blue: ${blueModeName}`;
  redScore.textContent = String(state.score.red);
  blueScore.textContent = String(state.score.blue);
  clockLabel.textContent = formatTime(state.time);
  runState.textContent = running ? 'Running' : 'Paused';
  ballSpeed.textContent = Math.round(Math.hypot(state.ball.velocity.x, state.ball.velocity.y)).toString();
  frameCount.textContent = state.frame.toString();
  redZone.textContent = `${Math.round(redControl)}%`;
  blueZone.textContent = `${Math.round(blueControl)}%`;
  const learningSnapshot = learning.snapshot;
  learnSamples.textContent = learningSnapshot.samples.toString();
  replaySamples.textContent = learningSnapshot.samples.toString();
  learnLoss.textContent = learningSnapshot.latestLoss.toFixed(3);
  learnVersion.textContent = `v${learningSnapshot.modelVersion}`;
  learnState.textContent = learningSnapshot.enabled ? 'On' : 'Off';
}

function resetMatch(): void {
  state = createInitialState();
  aiClock = new AiClock(strategies.red, strategies.blue, PHYSICS_HZ, AI_HZ);
  commands = {};
  accumulator = 0;
  lastLoggedGoalFrame = -1;
  eventLog.replaceChildren();
  appendLog('Kickoff', `${teamModeName('red')} red vs ${teamModeName('blue')} blue.`);
  render();
}

async function hydrateBundledPolicy(): Promise<void> {
  try {
    const bundled = await loadBundledPolicy();
    if (!bundled) {
      return;
    }

    resetWeights = [...bundled.weights];
    if (savedPolicy && window.localStorage.getItem(BUNDLED_POLICY_VERSION_KEY) === BUNDLED_POLICY_VERSION) {
      const selected = selectAcceptedPolicy(bundled.weights, savedPolicy.weights, {
        seed: 503,
        matches: 1,
        frames: PHYSICS_HZ * 8,
        opponent: traditionalStrategy,
        minDelta: 1
      });
      if (selected.source === 'current') {
        learning.applyTrainingWeights(bundled.weights, bundled.metadata?.loss ?? 0);
        neuralWeights = learning.currentWeights;
        rebuildStrategies();
        persistLearningState();
        appendLog(
          'Training',
          `Saved model gated out (${selected.candidateScore.toFixed(1)} vs bundled ${selected.currentScore.toFixed(1)}); using bundled baseline.`
        );
        render();
      }
      return;
    }

    learning.applyTrainingWeights(bundled.weights, bundled.metadata?.loss ?? 0);
    neuralWeights = learning.currentWeights;
    rebuildStrategies();
    persistLearningState();
    window.localStorage.setItem(BUNDLED_POLICY_VERSION_KEY, BUNDLED_POLICY_VERSION);
    const score = bundled.metadata?.selectionScore;
    const scoreText = typeof score === 'number' && Number.isFinite(score)
      ? `, selection ${score.toFixed(1)}`
      : '';
    appendLog(
      'Training',
      `Loaded bundled self-play model${scoreText}.`
    );
    render();
  } catch (error) {
    appendLog('Training', `Bundled model load failed: ${error instanceof Error ? error.message : String(error)}.`);
  }
}

function setControlMode(team: Team, mode: ControlMode): void {
  controlModes = { ...controlModes, [team]: mode };
  rebuildStrategies();
  commands = {};
  pressedKeys = new Set(pressedKeys);
  syncControlSelects();
  appendLog('Control', `${capitalize(team)} changed to ${MODE_LABELS[mode]}.`);
  render();
}

function makeStrategyPair(modes: ControlSelections): StrategyPair {
  return {
    red: strategyForMode(modes.red),
    blue: strategyForMode(modes.blue)
  };
}

function strategyForMode(mode: ControlMode): DisposableStrategy {
  if (mode === 'human') {
    return idleStrategy;
  }
  if (mode === 'idle') {
    return idleStrategy;
  }
  if (mode === 'traditional') {
    return traditionalStrategy;
  }

  return createAsyncNeuralStrategy({
    weights: () => neuralWeights,
    name: 'neural',
    onError: (message) => {
      if (!neuralWorkerErrorLogged) {
        neuralWorkerErrorLogged = true;
        appendLog('Neural AI', `Worker decision failed: ${message}`);
      }
    }
  });
}

function humanCommandsForActiveTeams(): CommandMap {
  return {
    ...(controlModes.red === 'human' ? humanCommandForTeam('red', pressedKeys) : {}),
    ...(controlModes.blue === 'human' ? humanCommandForTeam('blue', pressedKeys) : {})
  };
}

function syncControlSelects(): void {
  redControlMode.value = controlModes.red;
  blueControlMode.value = controlModes.blue;
}

function readControlMode(select: HTMLSelectElement): ControlMode {
  const value = select.value;
  if (value === 'idle' || value === 'traditional' || value === 'neural' || value === 'human') {
    return value;
  }

  return 'idle';
}

function teamModeName(team: Team): string {
  return MODE_LABELS[controlModes[team]];
}

function startLearningMode(): void {
  const session = learning.startLearningMode();
  controlModes = session.controlModes;
  neuralWeights = learning.currentWeights;
  rebuildStrategies();
  resetMatch();
  syncControlSelects();
  appendLog('Learning', 'Human red vs Neural blue. Use Q/A for left track and W/S for right track.');
}

function rebuildStrategies(): void {
  disposeStrategyPair(strategies);
  strategies = makeStrategyPair(controlModes);
  aiClock = new AiClock(strategies.red, strategies.blue, PHYSICS_HZ, AI_HZ);
}

function disposeStrategyPair(pair: StrategyPair): void {
  pair.red.dispose?.();
  if (pair.blue !== pair.red) {
    pair.blue.dispose?.();
  }
}

function persistLearningState(): void {
  const snapshot = learning.snapshot;
  saveLearnedPolicy(window.localStorage, {
    weights: learning.currentWeights,
    meta: {
      modelVersion: snapshot.modelVersion,
      samples: snapshot.samples,
      latestLoss: snapshot.latestLoss
    }
  });
  saveLearningReplay(window.localStorage, learning.replaySamples);
}

function scheduleLearningPersist(): void {
  if (learningPersistTimer !== undefined) {
    return;
  }

  learningPersistTimer = window.setTimeout(() => {
    learningPersistTimer = undefined;
    persistLearningState();
  }, 250);
}

function cancelScheduledLearningPersist(): void {
  if (learningPersistTimer === undefined) {
    return;
  }

  window.clearTimeout(learningPersistTimer);
  learningPersistTimer = undefined;
}

function setRunButton(): void {
  toggleRun.title = running ? 'Pause' : 'Run';
  toggleRun.innerHTML = running ? '<i data-lucide="pause"></i>' : '<i data-lucide="play"></i>';
  createIcons({ icons: { Pause, Play } });
}

function appendLog(title: string, body: string): void {
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `<strong>${title}</strong><br />${body}`;
  eventLog.prepend(entry);

  while (eventLog.childElementCount > 12) {
    eventLog.lastElementChild?.remove();
  }
}

function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => resolve());
  });
}

function downloadTextFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing #${id}`);
  }
  return element as T;
}

function formatTime(time: number): string {
  const totalSeconds = Math.floor(time);
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
  const seconds = (totalSeconds % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function capitalize(value: string): string {
  return value.length === 0 ? value : `${value[0].toUpperCase()}${value.slice(1)}`;
}

function clampPercent(value: number): number {
  return Math.max(0, Math.min(100, value));
}

function normalizeKey(key: string): string {
  return key.toLowerCase();
}

function isFormControl(target: EventTarget | null): boolean {
  return target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement;
}
