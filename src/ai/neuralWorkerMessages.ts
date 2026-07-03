import type { GameState, Team } from '../game/model';
import type { CommandMap } from '../game/strategy';
import type { NeuralWeights } from './neuralWeights';

export type NeuralWorkerRequest = {
  id: number;
  state: GameState;
  team: Team;
  weights: NeuralWeights;
  name: string;
  tacticalRollout: boolean;
};

export type NeuralWorkerResponse = {
  id: number;
  team: Team;
  frame: number;
  commands: CommandMap;
  error?: string;
};
