use std::env;
use std::error::Error;
use std::fs;

const INPUT_COUNT: usize = 36;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 64;
const OUTPUT_COUNT: usize = 9;
const WEIGHT_COUNT: usize =
    HIDDEN1 * (INPUT_COUNT + 1) + HIDDEN2 * (HIDDEN1 + 1) + OUTPUT_COUNT * (HIDDEN2 + 1);

const PHYSICS_HZ: usize = 30;
const AI_HZ: usize = 5;
const FIXED_DT: f64 = 1.0 / PHYSICS_HZ as f64;
const FIELD_LENGTH: f64 = 1050.0;
const FIELD_WIDTH: f64 = 680.0;
const BALL_RADIUS: f64 = 34.0;
const TANK_LENGTH: f64 = 102.0;
const TANK_WIDTH: f64 = 102.0;
const TANK_NOSE_LENGTH: f64 = 51.0;
const TANK_RADIUS: f64 = 114.03946685248927;
const GRAVITY: f64 = 600.0;
const TANK_MASS: f64 = 9.0;
const BALL_MASS: f64 = 1.4;
const TANK_STATIC_FRICTION: f64 = 4.4;
const TANK_STAMINA: f64 = 100.0;
const TANK_STAMINA_DRAIN: f64 = 10.0;
const TANK_STAMINA_RECOVERY: f64 = 20.0;
const GOAL_MOUTH: f64 = 230.0;
const WALL_RESTITUTION: f64 = 0.78;
const BALL_DAMPING_PER_SECOND: f64 = 0.46;
const COLLISION_ITERATIONS: usize = 16;
const POSITION_SLOP: f64 = 0.005;
const TANK_TANK_RESTITUTION: f64 = 0.04;
const TANK_BALL_RESTITUTION: f64 = 0.18;
const TANK_WALL_RESTITUTION: f64 = 0.08;
const TANK_IDLE_ANGULAR_FRICTION: f64 = 60.0;
const TANK_DRIVE_ACCELERATION: f64 = 3600.0;
const TANK_LATERAL_GRIP_ACCELERATION: f64 = 3600.0;
const TANK_TRACK_ANGULAR_ACCELERATION: f64 = 90.0;
const FULL_POWER_STAMINA_RATIO: f64 = 0.5;
const MAX_TRACK_SPEED: f64 = 245.0;
const TRACK_WIDTH: f64 = TANK_WIDTH * 0.82;
const STAMINA_CONSERVE_RATIO: f64 = 0.58;
const CRITICAL_STAMINA_RATIO: f64 = 0.22;
const DECISIVE_CONTACT_BUFFER: f64 = 28.0;
const TACTICAL_ROLLOUT_FRAMES: usize = 18;
const TACTICAL_IMPROVEMENT_MARGIN: f64 = 0.018;
const TRADITIONAL_DEFENSE_X: f64 = 108.0;
const TRADITIONAL_DANGER_DEPTH: f64 = 265.0;
const TRADITIONAL_OWN_GOAL_AVOID_DEPTH: f64 = 205.0;
const TRADITIONAL_BALL_PREDICT_SECONDS: f64 = 0.75;
const TRADITIONAL_STRIKE_SETUP_DISTANCE: f64 = BALL_RADIUS + TANK_LENGTH + 42.0;
const TRADITIONAL_STRIKE_APPROACH_TOLERANCE: f64 = TANK_RADIUS + 54.0;
const TRADITIONAL_STRIKE_LATERAL_TOLERANCE: f64 = BALL_RADIUS + TANK_WIDTH * 0.58;
const TRADITIONAL_STAMINA_CONSERVE_RATIO: f64 = 0.5;
const TRADITIONAL_NEAR_BALL_BUFFER: f64 = 24.0;
const TRADITIONAL_SIDE_WALL_DEPTH: f64 = BALL_RADIUS + 54.0;
const TRADITIONAL_OPPONENT_CORNER_DEPTH: f64 = BALL_RADIUS + TANK_LENGTH + 72.0;
const TRADITIONAL_STRAIGHT_HEADING_TOLERANCE: f64 = 0.18;
const OUTCOME_SIGN_EPSILON: f64 = 1e-9;

#[derive(Clone)]
struct Sample {
    inputs: [f64; INPUT_COUNT],
    action: usize,
    weight: f64,
    advantage: f64,
    old_probability: Option<f64>,
    old_probabilities: Option<[f64; OUTPUT_COUNT]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    BehaviorCloning,
    PolicyGradient,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Team {
    Red,
    Blue,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StartStateMode {
    Open,
    OutcomeCurriculum,
    OwnGoalDefense,
    CornerFight,
    LooseBallContest,
    Mixed,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActualStartStateMode {
    Open,
    OutcomeCurriculum,
    OwnGoalDefense,
    CornerFight,
    LooseBallContest,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdvantageBaseline {
    Global,
    StartTeamTime,
    Learned,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActionMode {
    Raw,
    Runtime,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RuntimeWrapperWeightMode {
    None,
    TacticalDownweight,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpponentMode {
    SelfPlay,
    Traditional,
    League,
}

struct Options {
    mode: Mode,
    weights_path: String,
    opponent_weights_path: Option<String>,
    league_opponent_weight_paths: Vec<String>,
    data_path: Option<String>,
    output_path: String,
    metrics_output_path: Option<String>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    l2: f64,
    gradient_clip: f64,
    seed: u32,
    matches: usize,
    frames: usize,
    ppo_clip: f64,
    temperature: f64,
    discount: f64,
    goal_reward: f64,
    win_reward: f64,
    start_state_mode: StartStateMode,
    open_start_ratio: Option<f64>,
    advantage_baseline: AdvantageBaseline,
    action_mode: ActionMode,
    runtime_survivors_only: bool,
    runtime_wrapper_weight_mode: RuntimeWrapperWeightMode,
    runtime_tactical_rewrite_weight: f64,
    action_retention_weight: f64,
    opponent_mode: OpponentMode,
    league_current_weight: f64,
    league_traditional_weight: f64,
}

enum OpponentPolicy {
    Neural(Vec<f64>),
    Traditional,
}

struct LeagueOpponent {
    policy: OpponentPolicy,
    weight: f64,
}

#[derive(Clone, Copy)]
struct Vec2 {
    x: f64,
    y: f64,
}

#[derive(Clone)]
struct Tank {
    team: Team,
    position: Vec2,
    velocity: Vec2,
    angle: f64,
    angular_velocity: f64,
    stamina: f64,
}

#[derive(Clone)]
struct Ball {
    position: Vec2,
    velocity: Vec2,
}

#[derive(Clone)]
struct GameState {
    frame: usize,
    time: f64,
    tanks: [Tank; 2],
    ball: Ball,
    score_red: i32,
    score_blue: i32,
    last_goal: Option<GoalEvent>,
}

#[derive(Clone, Copy)]
struct GoalEvent {
    team: Team,
    frame: usize,
}

#[derive(Clone, Copy)]
struct Command {
    left: f64,
    right: f64,
}

#[derive(Clone)]
struct PendingDecision {
    inputs: [f64; INPUT_COUNT],
    sampled_action: usize,
    action: usize,
    team: Team,
    frame: usize,
    probability: f64,
    probabilities: [f64; OUTPUT_COUNT],
    trainable: bool,
    start_state_mode: ActualStartStateMode,
    tactical_changed: bool,
    stamina_conserved: bool,
    critical_regulated: bool,
}

struct Collection {
    samples: Vec<Sample>,
    decisions: usize,
    policy_action_survival: PolicyActionSurvival,
    runtime_decision_outcomes: RuntimeDecisionOutcomes,
    start_families: StartFamilyCounts,
    frames: usize,
    red_goals: i32,
    blue_goals: i32,
    final_state: GameState,
}

struct TrainingResult {
    weights: Vec<f64>,
    loss: f64,
    trained_samples: usize,
    samples: usize,
    decisions: usize,
    policy_action_survival: PolicyActionSurvival,
    runtime_decision_outcomes: RuntimeDecisionOutcomes,
    start_families: StartFamilyCounts,
    frames: usize,
    red_goals: i32,
    blue_goals: i32,
    final_state: GameState,
    advantage_baseline: AdvantageBaseline,
    open_start_ratio: Option<f64>,
    action_mode: ActionMode,
    runtime_survivors_only: bool,
    runtime_wrapper_weight_mode: RuntimeWrapperWeightMode,
    runtime_tactical_rewrite_weight: f64,
    action_retention_weight: f64,
    opponent_mode: OpponentMode,
    league_opponent_count: usize,
    league_current_weight: f64,
    league_traditional_weight: f64,
}

struct PositionEvaluation {
    total: f64,
    corner_escape: f64,
}

#[derive(Clone, Copy)]
struct StartFamilyCounts {
    open: usize,
    outcome_curriculum: usize,
    own_goal_defense: usize,
    corner_fight: usize,
    loose_ball_contest: usize,
}

#[derive(Clone, Copy)]
struct PolicyActionSurvival {
    sampled: usize,
    survived: usize,
    changed: usize,
    tactical_changed: usize,
    stamina_conserved: usize,
    critical_regulated: usize,
}

#[derive(Clone, Copy)]
struct RuntimeDecisionOutcomes {
    survived: RuntimeDecisionOutcomeStats,
    changed: RuntimeDecisionOutcomeStats,
    tactical_changed: RuntimeDecisionOutcomeStats,
    stamina_conserved: RuntimeDecisionOutcomeStats,
    critical_regulated: RuntimeDecisionOutcomeStats,
}

#[derive(Clone, Copy)]
struct RuntimeDecisionOutcomeStats {
    count: usize,
    return_sum: f64,
    advantage_sum: f64,
    abs_advantage_sum: f64,
    positive_returns: usize,
    negative_returns: usize,
    positive_advantages: usize,
    negative_advantages: usize,
}

#[derive(Clone, Copy)]
struct RuntimeActionResult {
    action: usize,
    tactical_changed: bool,
    stamina_conserved: bool,
    critical_regulated: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = parse_args(env::args().skip(1).collect())?;
    let mut weights = load_weights(&options.weights_path)?;

    match options.mode {
        Mode::BehaviorCloning => {
            let data_path = options
                .data_path
                .as_ref()
                .ok_or("Behavior cloning mode requires --data samples.json")?;
            let samples = load_samples(data_path)?;
            let loss = train_behavior_cloning(&mut weights, &samples, &options);
            eprintln!(
                "samples={} epochs={} loss={:.6}",
                samples.len(),
                options.epochs,
                loss
            );
            fs::write(
                &options.output_path,
                serialize_weights(&weights, &options, samples.len(), "rust-bc"),
            )?;
        }
        Mode::PolicyGradient => {
            let result = train_policy_gradient_self_play(&weights, &options);
            eprintln!(
                "samples={} trainedSamples={} frames={} goals={}-{} loss={:.6}",
                result.samples,
                result.trained_samples,
                result.frames,
                result.red_goals,
                result.blue_goals,
                result.loss
            );
            fs::write(
                &options.output_path,
                serialize_weights(&result.weights, &options, result.samples, "rust-policy-gradient"),
            )?;
            if let Some(path) = &options.metrics_output_path {
                fs::write(path, serialize_metrics(&result))?;
            }
        }
    }

    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Options, Box<dyn Error>> {
    let mut options = Options {
        mode: Mode::BehaviorCloning,
        weights_path: String::new(),
        opponent_weights_path: None,
        league_opponent_weight_paths: Vec::new(),
        data_path: None,
        output_path: String::new(),
        metrics_output_path: None,
        epochs: 12,
        batch_size: 64,
        learning_rate: 0.01,
        l2: 0.00024,
        gradient_clip: 1.2,
        seed: 1,
        matches: 8,
        frames: PHYSICS_HZ * 20,
        ppo_clip: 0.2,
        temperature: 1.08,
        discount: 0.992,
        goal_reward: 1.0,
        win_reward: 1.4,
        start_state_mode: StartStateMode::OutcomeCurriculum,
        open_start_ratio: None,
        advantage_baseline: AdvantageBaseline::Global,
        action_mode: ActionMode::Raw,
        runtime_survivors_only: false,
        runtime_wrapper_weight_mode: RuntimeWrapperWeightMode::None,
        runtime_tactical_rewrite_weight: 0.5,
        action_retention_weight: 0.0,
        opponent_mode: OpponentMode::SelfPlay,
        league_current_weight: 1.0,
        league_traditional_weight: 0.0,
    };
    let mut index = 0;
    while index < args.len() {
        let key = &args[index];
        index += 1;
        if index >= args.len() {
            return Err(format!("Missing value after {key}").into());
        }
        let value = args[index].clone();
        index += 1;
        match key.as_str() {
            "--mode" => {
                options.mode = match value.as_str() {
                    "behavior-cloning" | "bc" => Mode::BehaviorCloning,
                    "policy-gradient" | "pg" | "ppo" => Mode::PolicyGradient,
                    _ => return Err(format!("Unknown mode: {value}").into()),
                }
            }
            "--weights" => options.weights_path = value,
            "--opponent-weights" => options.opponent_weights_path = Some(value),
            "--league-opponent-weights" => options.league_opponent_weight_paths.push(value),
            "--league-current-weight" => options.league_current_weight = value.parse::<f64>()?.max(0.0),
            "--league-traditional-weight" => {
                options.league_traditional_weight = value.parse::<f64>()?.max(0.0)
            }
            "--data" => options.data_path = Some(value),
            "--output" => options.output_path = value,
            "--metrics-output" => options.metrics_output_path = Some(value),
            "--epochs" => options.epochs = value.parse::<usize>()?,
            "--batch-size" => options.batch_size = value.parse::<usize>()?.max(1),
            "--learning-rate" => options.learning_rate = value.parse::<f64>()?,
            "--l2" => options.l2 = value.parse::<f64>()?,
            "--gradient-clip" => options.gradient_clip = value.parse::<f64>()?,
            "--seed" => options.seed = value.parse::<u32>()?,
            "--matches" => options.matches = value.parse::<usize>()?.max(1),
            "--frames" => options.frames = value.parse::<usize>()?.max(1),
            "--ppo-clip" => options.ppo_clip = value.parse::<f64>()?.max(0.0),
            "--temperature" => options.temperature = value.parse::<f64>()?.max(0.05),
            "--discount" => options.discount = clamp01(value.parse::<f64>()?),
            "--goal-reward" => options.goal_reward = value.parse::<f64>()?,
            "--win-reward" => options.win_reward = value.parse::<f64>()?,
            "--start-state-mode" => {
                options.start_state_mode = match value.as_str() {
                    "open" => StartStateMode::Open,
                    "outcome-curriculum" => StartStateMode::OutcomeCurriculum,
                    "own-goal-defense" => StartStateMode::OwnGoalDefense,
                    "corner-fight" => StartStateMode::CornerFight,
                    "loose-ball-contest" => StartStateMode::LooseBallContest,
                    "mixed" => StartStateMode::Mixed,
                    _ => return Err(format!("Unknown start-state mode: {value}").into()),
                }
            }
            "--open-start-ratio" => options.open_start_ratio = Some(clamp01(value.parse::<f64>()?)),
            "--advantage-baseline" => {
                options.advantage_baseline = match value.as_str() {
                    "global" => AdvantageBaseline::Global,
                    "start-team-time" => AdvantageBaseline::StartTeamTime,
                    "learned" => AdvantageBaseline::Learned,
                    _ => return Err(format!("Unknown advantage baseline: {value}").into()),
                }
            }
            "--action-mode" => {
                options.action_mode = match value.as_str() {
                    "raw" => ActionMode::Raw,
                    "runtime" => ActionMode::Runtime,
                    _ => return Err(format!("Unknown action mode: {value}").into()),
                }
            }
            "--runtime-survivors-only" => {
                options.runtime_survivors_only = parse_bool_arg(&value, "--runtime-survivors-only")?
            }
            "--runtime-wrapper-weight-mode" => {
                options.runtime_wrapper_weight_mode = match value.as_str() {
                    "none" => RuntimeWrapperWeightMode::None,
                    "tactical-downweight" => RuntimeWrapperWeightMode::TacticalDownweight,
                    _ => return Err(format!("Unknown runtime wrapper weight mode: {value}").into()),
                }
            }
            "--runtime-tactical-rewrite-weight" => {
                options.runtime_tactical_rewrite_weight = clamp01(value.parse::<f64>()?)
            }
            "--action-retention-weight" => {
                options.action_retention_weight = value.parse::<f64>()?.max(0.0)
            }
            "--opponent-mode" => {
                options.opponent_mode = match value.as_str() {
                    "self" | "self-play" => OpponentMode::SelfPlay,
                    "traditional" => OpponentMode::Traditional,
                    "league" => OpponentMode::League,
                    _ => return Err(format!("Unknown opponent mode: {value}").into()),
                }
            }
            _ => return Err(format!("Unknown argument: {key}").into()),
        }
    }
    if options.weights_path.is_empty() || options.output_path.is_empty() {
        return Err("Usage: soccer-policy-trainer --weights weights.json --output out.json".into());
    }
    Ok(options)
}

fn parse_bool_arg(value: &str, name: &str) -> Result<bool, Box<dyn Error>> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!("Expected boolean value for {name}, received {value}").into()),
    }
}

fn load_weights(path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let open = if let Some(token) = text.find("\"weights\"") {
        text[token..]
            .find('[')
            .map(|offset| token + offset)
            .ok_or("Expected weights array")?
    } else {
        text.find('[').ok_or("Expected weights array")?
    };
    let weights = parse_number_array(&text, open)?;
    if weights.len() != WEIGHT_COUNT {
        return Err(format!("Expected {WEIGHT_COUNT} weights, received {}", weights.len()).into());
    }
    Ok(weights)
}

fn load_samples(path: &str) -> Result<Vec<Sample>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let samples_token = text.find("\"samples\"").ok_or("Missing token: \"samples\"")?;
    let open = text[samples_token..]
        .find('[')
        .map(|offset| samples_token + offset)
        .ok_or("Missing samples array")?;
    let close = matching_bracket(&text, open)?;
    let array = &text[open + 1..close];
    let mut samples = Vec::new();
    let mut cursor = 0;

    while let Some(relative_inputs_key) = array[cursor..].find("\"inputs\"") {
        let inputs_key = cursor + relative_inputs_key;
        let input_open = array[inputs_key..]
            .find('[')
            .map(|offset| inputs_key + offset)
            .ok_or("Sample missing inputs array")?;
        let input_close = matching_bracket(array, input_open)?;
        let object_end = array[input_close..]
            .find('}')
            .map(|offset| input_close + offset)
            .ok_or("Sample object is not closed")?;
        let input_values = parse_number_array(array, input_open)?;
        if input_values.len() != INPUT_COUNT {
            return Err(format!("Expected 36 sample inputs, received {}", input_values.len()).into());
        }
        let mut inputs = [0.0; INPUT_COUNT];
        inputs.copy_from_slice(&input_values);
        let object = &array[inputs_key..=object_end];
        let advantage = parse_number_field(object, "\"advantage\"", 0.0);
        let old_probability = object
            .find("\"oldProbability\"")
            .map(|_| parse_number_field(object, "\"oldProbability\"", 0.0))
            .filter(|value| *value > 0.0);
        samples.push(Sample {
            inputs,
            action: parse_number_field(object, "\"actionIndex\"", 4.0)
                .round()
                .clamp(0.0, (OUTPUT_COUNT - 1) as f64) as usize,
            weight: parse_number_field(object, "\"weight\"", 1.0).max(0.0),
            advantage,
            old_probability,
            old_probabilities: None,
        });
        cursor = object_end + 1;
    }

    if samples.is_empty() {
        return Err("No samples found in dataset".into());
    }
    Ok(samples)
}

fn train_behavior_cloning(weights: &mut [f64], samples: &[Sample], options: &Options) -> f64 {
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let mut loss = 0.0;
    for epoch in 0..options.epochs {
        shuffle_order(&mut order, options.seed.wrapping_add((epoch as u32).wrapping_mul(7919)));
        for start in (0..samples.len()).step_by(options.batch_size) {
            let end = (start + options.batch_size).min(samples.len());
            loss = train_batch(weights, samples, &order[start..end], options, false);
        }
    }
    loss
}

fn train_policy_gradient_self_play(initial_weights: &[f64], options: &Options) -> TrainingResult {
    let collection = collect_policy_gradient_self_play(initial_weights, options);
    let mut weights = initial_weights.to_vec();
    let mut order: Vec<usize> = (0..collection.samples.len()).collect();
    let mut loss = 0.0;
    let mut trained_samples = 0;

    for epoch in 0..options.epochs {
        shuffle_order(
            &mut order,
            options
                .seed
                .wrapping_add(90_017)
                .wrapping_add((epoch as u32).wrapping_mul(7919)),
        );
        for start in (0..collection.samples.len()).step_by(options.batch_size) {
            let end = (start + options.batch_size).min(collection.samples.len());
            loss = train_batch(&mut weights, &collection.samples, &order[start..end], options, true);
            trained_samples += end - start;
        }
    }

    TrainingResult {
        weights,
        loss,
        trained_samples,
        samples: collection.samples.len(),
        decisions: collection.decisions,
        policy_action_survival: collection.policy_action_survival,
        runtime_decision_outcomes: collection.runtime_decision_outcomes,
        start_families: collection.start_families,
        frames: collection.frames,
        red_goals: collection.red_goals,
        blue_goals: collection.blue_goals,
        final_state: collection.final_state,
        advantage_baseline: options.advantage_baseline,
        open_start_ratio: options.open_start_ratio,
        action_mode: options.action_mode,
        runtime_survivors_only: options.runtime_survivors_only,
        runtime_wrapper_weight_mode: options.runtime_wrapper_weight_mode,
        runtime_tactical_rewrite_weight: options.runtime_tactical_rewrite_weight,
        action_retention_weight: options.action_retention_weight,
        opponent_mode: options.opponent_mode,
        league_opponent_count: options.league_opponent_weight_paths.len(),
        league_current_weight: options.league_current_weight,
        league_traditional_weight: options.league_traditional_weight,
    }
}

fn collect_policy_gradient_self_play(weights: &[f64], options: &Options) -> Collection {
    let opponent_weights = options
        .opponent_weights_path
        .as_ref()
        .map(|path| load_weights(path))
        .transpose()
        .expect("Expected valid opponent weights");
    let opponent_weights_ref = opponent_weights.as_deref().unwrap_or(weights);
    let shared_policy = opponent_weights.is_none();
    let league_opponents = load_league_opponents(weights, options);
    let mut random = SeededRandom::new(options.seed);
    let frames_per_decision = (PHYSICS_HZ / AI_HZ).max(1);
    let mut all_decisions: Vec<(PendingDecision, f64)> = Vec::new();
    let mut start_families = StartFamilyCounts {
        open: 0,
        outcome_curriculum: 0,
        own_goal_defense: 0,
        corner_fight: 0,
        loose_ball_contest: 0,
    };
    let mut red_goals = 0;
    let mut blue_goals = 0;
    let mut completed_frames = 0;
    let mut final_state = initial_state();

    for match_index in 0..options.matches {
        let start_state_mode = resolve_start_state_mode(
            options.start_state_mode,
            match_index,
            options.open_start_ratio,
        );
        increment_start_family(&mut start_families, start_state_mode);
        let mut state = seeded_initial_state(&mut random, match_index, start_state_mode);
        let train_team = if match_index % 2 == 0 {
            Team::Red
        } else {
            Team::Blue
        };
        let league_opponent_index = if options.opponent_mode == OpponentMode::League {
            Some(sample_league_opponent(&league_opponents, &mut random))
        } else {
            None
        };
        let mut pending: Vec<PendingDecision> = Vec::new();
        let mut goals: Vec<GoalEvent> = Vec::new();
        let mut red_command = stop_command();
        let mut blue_command = stop_command();

        for _ in 0..options.frames {
            if state.frame % frames_per_decision == 0 {
                let red_decision = match options.opponent_mode {
                    OpponentMode::SelfPlay => sample_team_decision(
                        &state,
                        Team::Red,
                        weights,
                        options.temperature,
                        &mut random,
                        true,
                        start_state_mode,
                        options.action_mode,
                    ),
                    OpponentMode::Traditional => {
                        if train_team == Team::Red {
                            sample_team_decision(
                                &state,
                                Team::Red,
                                weights,
                                options.temperature,
                                &mut random,
                                true,
                                start_state_mode,
                                options.action_mode,
                            )
                        } else {
                            (traditional_team_command(&state, Team::Red), None)
                        }
                    }
                    OpponentMode::League => league_team_decision(
                        &state,
                        Team::Red,
                        train_team,
                        league_opponent_index.expect("league opponent index"),
                        weights,
                        &league_opponents,
                        options.temperature,
                        &mut random,
                        start_state_mode,
                        options.action_mode,
                    ),
                };
                let blue_decision = match options.opponent_mode {
                    OpponentMode::SelfPlay => sample_team_decision(
                        &state,
                        Team::Blue,
                        opponent_weights_ref,
                        options.temperature,
                        &mut random,
                        shared_policy,
                        start_state_mode,
                        options.action_mode,
                    ),
                    OpponentMode::Traditional => {
                        if train_team == Team::Blue {
                            sample_team_decision(
                                &state,
                                Team::Blue,
                                weights,
                                options.temperature,
                                &mut random,
                                true,
                                start_state_mode,
                                options.action_mode,
                            )
                        } else {
                            (traditional_team_command(&state, Team::Blue), None)
                        }
                    }
                    OpponentMode::League => league_team_decision(
                        &state,
                        Team::Blue,
                        train_team,
                        league_opponent_index.expect("league opponent index"),
                        weights,
                        &league_opponents,
                        options.temperature,
                        &mut random,
                        start_state_mode,
                        options.action_mode,
                    ),
                };
                red_command = red_decision.0;
                blue_command = blue_decision.0;
                if let Some(decision) = red_decision.1 {
                    pending.push(decision);
                }
                if let Some(decision) = blue_decision.1 {
                    pending.push(decision);
                }
            }

            step_game(&mut state, red_command, blue_command, FIXED_DT);
            if let Some(goal) = state.last_goal {
                if goal.frame == state.frame.saturating_sub(1) {
                    goals.push(goal);
                }
            }
        }

        let red_diff = state.score_red - state.score_blue;
        for decision in pending {
            let total_return = decision_return(
                &decision,
                &goals,
                red_diff,
                options.discount,
                options.goal_reward,
                options.win_reward,
            );
            all_decisions.push((decision, total_return));
        }

        red_goals += state.score_red;
        blue_goals += state.score_blue;
        completed_frames += options.frames;
        final_state = state;
    }

    let decision_count = all_decisions.len();
    let policy_action_survival = summarize_policy_action_survival(&all_decisions);
    let advantages = normalized_advantages(&all_decisions, options.advantage_baseline);
    let runtime_decision_outcomes = summarize_runtime_decision_outcomes(&all_decisions, &advantages);
    let samples = all_decisions
        .into_iter()
        .zip(advantages)
        .filter_map(|((decision, _), advantage)| {
            if !decision.trainable {
                return None;
            }
            if options.runtime_survivors_only && decision.sampled_action != decision.action {
                return None;
            }
            let wrapper_weight = runtime_wrapper_sample_weight(
                &decision,
                options.runtime_wrapper_weight_mode,
                options.runtime_tactical_rewrite_weight,
            );
            Some(Sample {
                inputs: decision.inputs,
                action: decision.action,
                weight: advantage.abs() * wrapper_weight,
                advantage,
                old_probability: Some(decision.probability),
                old_probabilities: Some(decision.probabilities),
            })
        })
        .collect();

    Collection {
        samples,
        decisions: decision_count,
        policy_action_survival,
        runtime_decision_outcomes,
        start_families,
        frames: completed_frames,
        red_goals,
        blue_goals,
        final_state,
    }
}

fn runtime_wrapper_sample_weight(
    decision: &PendingDecision,
    mode: RuntimeWrapperWeightMode,
    tactical_rewrite_weight: f64,
) -> f64 {
    match mode {
        RuntimeWrapperWeightMode::None => 1.0,
        RuntimeWrapperWeightMode::TacticalDownweight => {
            if decision.tactical_changed {
                tactical_rewrite_weight
            } else {
                1.0
            }
        }
    }
}

fn summarize_policy_action_survival(decisions: &[(PendingDecision, f64)]) -> PolicyActionSurvival {
    let mut summary = PolicyActionSurvival {
        sampled: 0,
        survived: 0,
        changed: 0,
        tactical_changed: 0,
        stamina_conserved: 0,
        critical_regulated: 0,
    };

    for (decision, _) in decisions {
        if !decision.trainable {
            continue;
        }
        summary.sampled += 1;
        if decision.sampled_action == decision.action {
            summary.survived += 1;
        } else {
            summary.changed += 1;
        }
        summary.tactical_changed += usize::from(decision.tactical_changed);
        summary.stamina_conserved += usize::from(decision.stamina_conserved);
        summary.critical_regulated += usize::from(decision.critical_regulated);
    }

    summary
}

fn summarize_runtime_decision_outcomes(
    decisions: &[(PendingDecision, f64)],
    advantages: &[f64],
) -> RuntimeDecisionOutcomes {
    let mut summary = RuntimeDecisionOutcomes {
        survived: empty_runtime_decision_outcome_stats(),
        changed: empty_runtime_decision_outcome_stats(),
        tactical_changed: empty_runtime_decision_outcome_stats(),
        stamina_conserved: empty_runtime_decision_outcome_stats(),
        critical_regulated: empty_runtime_decision_outcome_stats(),
    };

    for ((decision, total_return), advantage) in decisions.iter().zip(advantages.iter()) {
        if !decision.trainable {
            continue;
        }
        if decision.sampled_action == decision.action {
            summary.survived.record(*total_return, *advantage);
        } else {
            summary.changed.record(*total_return, *advantage);
        }
        if decision.tactical_changed {
            summary.tactical_changed.record(*total_return, *advantage);
        }
        if decision.stamina_conserved {
            summary.stamina_conserved.record(*total_return, *advantage);
        }
        if decision.critical_regulated {
            summary.critical_regulated.record(*total_return, *advantage);
        }
    }

    summary
}

fn empty_runtime_decision_outcome_stats() -> RuntimeDecisionOutcomeStats {
    RuntimeDecisionOutcomeStats {
        count: 0,
        return_sum: 0.0,
        advantage_sum: 0.0,
        abs_advantage_sum: 0.0,
        positive_returns: 0,
        negative_returns: 0,
        positive_advantages: 0,
        negative_advantages: 0,
    }
}

impl RuntimeDecisionOutcomeStats {
    fn record(&mut self, total_return: f64, advantage: f64) {
        self.count += 1;
        self.return_sum += total_return;
        self.advantage_sum += advantage;
        self.abs_advantage_sum += advantage.abs();
        if total_return > OUTCOME_SIGN_EPSILON {
            self.positive_returns += 1;
        } else if total_return < -OUTCOME_SIGN_EPSILON {
            self.negative_returns += 1;
        }
        if advantage > OUTCOME_SIGN_EPSILON {
            self.positive_advantages += 1;
        } else if advantage < -OUTCOME_SIGN_EPSILON {
            self.negative_advantages += 1;
        }
    }
}

fn load_league_opponents(weights: &[f64], options: &Options) -> Vec<LeagueOpponent> {
    if options.opponent_mode != OpponentMode::League {
        return Vec::new();
    }

    let mut opponents = Vec::new();
    if options.league_current_weight > 0.0 {
        opponents.push(LeagueOpponent {
            policy: OpponentPolicy::Neural(weights.to_vec()),
            weight: options.league_current_weight,
        });
    }
    if let Some(path) = &options.opponent_weights_path {
        opponents.push(LeagueOpponent {
            policy: OpponentPolicy::Neural(load_weights(path).expect("Expected valid opponent weights")),
            weight: 1.0,
        });
    }
    for path in &options.league_opponent_weight_paths {
        opponents.push(LeagueOpponent {
            policy: OpponentPolicy::Neural(load_weights(path).expect("Expected valid league opponent weights")),
            weight: 1.0,
        });
    }
    if options.league_traditional_weight > 0.0 {
        opponents.push(LeagueOpponent {
            policy: OpponentPolicy::Traditional,
            weight: options.league_traditional_weight,
        });
    }

    if opponents.is_empty() {
        opponents.push(LeagueOpponent {
            policy: OpponentPolicy::Neural(weights.to_vec()),
            weight: 1.0,
        });
    }

    opponents
}

fn sample_league_opponent(opponents: &[LeagueOpponent], random: &mut SeededRandom) -> usize {
    let total_weight = opponents
        .iter()
        .map(|opponent| opponent.weight.max(0.0))
        .sum::<f64>();
    if total_weight <= 0.0 {
        return 0;
    }

    let mut cursor = random.next() * total_weight;
    for (index, opponent) in opponents.iter().enumerate() {
        cursor -= opponent.weight.max(0.0);
        if cursor <= 0.0 {
            return index;
        }
    }
    opponents.len().saturating_sub(1)
}

fn league_team_decision(
    state: &GameState,
    team: Team,
    train_team: Team,
    opponent_index: usize,
    weights: &[f64],
    opponents: &[LeagueOpponent],
    temperature: f64,
    random: &mut SeededRandom,
    start_state_mode: ActualStartStateMode,
    action_mode: ActionMode,
) -> (Command, Option<PendingDecision>) {
    if team == train_team {
        return sample_team_decision(
            state,
            team,
            weights,
            temperature,
            random,
            true,
            start_state_mode,
            action_mode,
        );
    }

    match &opponents[opponent_index].policy {
        OpponentPolicy::Neural(opponent_weights) => sample_team_decision(
            state,
            team,
            opponent_weights,
            temperature,
            random,
            false,
            start_state_mode,
            action_mode,
        ),
        OpponentPolicy::Traditional => (traditional_team_command(state, team), None),
    }
}

fn increment_start_family(counts: &mut StartFamilyCounts, mode: ActualStartStateMode) {
    match mode {
        ActualStartStateMode::Open => counts.open += 1,
        ActualStartStateMode::OutcomeCurriculum => counts.outcome_curriculum += 1,
        ActualStartStateMode::OwnGoalDefense => counts.own_goal_defense += 1,
        ActualStartStateMode::CornerFight => counts.corner_fight += 1,
        ActualStartStateMode::LooseBallContest => counts.loose_ball_contest += 1,
    }
}

fn train_batch(
    weights: &mut [f64],
    samples: &[Sample],
    order: &[usize],
    options: &Options,
    policy_gradient: bool,
) -> f64 {
    let mut gradient = vec![0.0; WEIGHT_COUNT];
    let mut h1 = [0.0; HIDDEN1];
    let mut h2 = [0.0; HIDDEN2];
    let mut logits = [0.0; OUTPUT_COUNT];
    let mut probs = [0.0; OUTPUT_COUNT];
    let mut d2 = [0.0; HIDDEN2];
    let mut d1 = [0.0; HIDDEN1];
    let mut total_loss = 0.0;
    let mut total_weight = 0.0;

    for &sample_index in order {
        let sample = &samples[sample_index];
        let sample_weight = if policy_gradient {
            sample.weight
        } else {
            sample.weight
        };
        if sample_weight <= 0.0 || (policy_gradient && sample.advantage == 0.0) {
            continue;
        }

        forward_arrays(&sample.inputs, weights, &mut h1, &mut h2, &mut logits);
        softmax_into(&logits, &mut probs);

        let scale = if policy_gradient {
            let ratio = sample
                .old_probability
                .filter(|value| *value > 0.0)
                .map(|old| probs[sample.action] / old)
                .unwrap_or(1.0);
            if options.ppo_clip > 0.0
                && sample.old_probability.is_some()
                && clips_ppo_update(ratio, sample.advantage, options.ppo_clip)
            {
                total_weight += sample_weight;
                continue;
            }
            total_loss += -sample.advantage * probs[sample.action].max(1e-12).ln();
            sample.advantage
        } else {
            total_loss += -probs[sample.action].max(1e-12).ln() * sample_weight;
            sample_weight
        };
        let retention_weight = if policy_gradient {
            options.action_retention_weight * sample_weight
        } else {
            0.0
        };
        let old_probabilities = if retention_weight > 0.0 {
            sample.old_probabilities.as_ref()
        } else {
            None
        };
        if let Some(old_probs) = old_probabilities {
            total_loss += retention_weight * kl_divergence(old_probs, &probs);
        }
        total_weight += sample_weight;
        d2.fill(0.0);
        d1.fill(0.0);

        for out in 0..OUTPUT_COUNT {
            let retention_delta = old_probabilities
                .map(|old_probs| retention_weight * (probs[out] - old_probs[out]))
                .unwrap_or(0.0);
            let delta = (probs[out] - if out == sample.action { 1.0 } else { 0.0 }) * scale
                + retention_delta;
            let row = layer2_offset() + out * (HIDDEN2 + 1);
            for input in 0..HIDDEN2 {
                gradient[row + input] += delta * h2[input];
                d2[input] += delta * weights[row + input];
            }
            gradient[row + HIDDEN2] += delta;
        }

        for out in 0..HIDDEN2 {
            let delta = d2[out] * (1.0 - h2[out] * h2[out]);
            let row = layer1_offset() + out * (HIDDEN1 + 1);
            for input in 0..HIDDEN1 {
                gradient[row + input] += delta * h1[input];
                d1[input] += delta * weights[row + input];
            }
            gradient[row + HIDDEN1] += delta;
        }

        for out in 0..HIDDEN1 {
            let delta = d1[out] * (1.0 - h1[out] * h1[out]);
            let row = layer0_offset() + out * (INPUT_COUNT + 1);
            for input in 0..INPUT_COUNT {
                gradient[row + input] += delta * sample.inputs[input];
            }
            gradient[row + INPUT_COUNT] += delta;
        }
    }

    let divisor = if total_weight > 0.0 { total_weight } else { 1.0 };
    for index in 0..WEIGHT_COUNT {
        let normalized = gradient[index] / divisor + options.l2 * weights[index];
        let clipped = normalized.clamp(-options.gradient_clip, options.gradient_clip);
        weights[index] = (weights[index] - options.learning_rate * clipped).clamp(-4.0, 4.0);
    }

    total_loss / divisor
}

fn kl_divergence(old_probs: &[f64; OUTPUT_COUNT], probs: &[f64; OUTPUT_COUNT]) -> f64 {
    let mut total = 0.0;
    for out in 0..OUTPUT_COUNT {
        let old = old_probs[out].max(1e-12);
        total += old * (old.ln() - probs[out].max(1e-12).ln());
    }
    total
}

fn sample_team_decision(
    state: &GameState,
    team: Team,
    weights: &[f64],
    temperature: f64,
    random: &mut SeededRandom,
    trainable: bool,
    start_state_mode: ActualStartStateMode,
    action_mode: ActionMode,
) -> (Command, Option<PendingDecision>) {
    let tank_index = match team {
        Team::Red => 0,
        Team::Blue => 1,
    };
    let tank = &state.tanks[tank_index];
    let inputs = extract_tank_inputs(state, team, tank);
    let mut logits = evaluate_policy(&inputs, weights);
    for logit in &mut logits {
        *logit /= temperature;
    }
    let probabilities = softmax(&logits);
    let sampled_action = sample_action(&probabilities, random);
    let runtime_result = match action_mode {
        ActionMode::Raw => RuntimeActionResult {
            action: sampled_action,
            tactical_changed: false,
            stamina_conserved: false,
            critical_regulated: false,
        },
        ActionMode::Runtime => runtime_action_result(state, team, tank, sampled_action),
    };
    let executed_action = runtime_result.action;
    let probability = probabilities[executed_action].max(1e-9);

    (
        action_index_to_command(executed_action),
        Some(PendingDecision {
            inputs,
            sampled_action,
            action: executed_action,
            team,
            frame: state.frame,
            probability,
            probabilities,
            trainable,
            start_state_mode,
            tactical_changed: runtime_result.tactical_changed,
            stamina_conserved: runtime_result.stamina_conserved,
            critical_regulated: runtime_result.critical_regulated,
        }),
    )
}

fn runtime_action_result(
    state: &GameState,
    team: Team,
    tank: &Tank,
    policy_action: usize,
) -> RuntimeActionResult {
    let pressures = pressure_signals(state, team);
    if should_conserve_stamina(state, team, tank, pressures) {
        return RuntimeActionResult {
            action: 4,
            tactical_changed: false,
            stamina_conserved: true,
            critical_regulated: false,
        };
    }

    let mut action = policy_action;
    let mut tactical_changed = false;
    if should_use_tactical_rollout(state, team, tank, pressures) {
        action = choose_tactical_action(state, team, policy_action);
        tactical_changed = action != policy_action;
    }

    let regulated_action = command_to_action_index(regulate_critical_stamina_command(
        state,
        team,
        tank,
        pressures,
        action_index_to_command(action),
    ));
    RuntimeActionResult {
        action: regulated_action,
        tactical_changed,
        stamina_conserved: false,
        critical_regulated: regulated_action != action,
    }
}

fn should_conserve_stamina(
    state: &GameState,
    team: Team,
    tank: &Tank,
    pressures: PressureSignals,
) -> bool {
    stamina_ratio(tank) < STAMINA_CONSERVE_RATIO
        && !urgent_stamina_spend(state, team, tank, pressures)
}

fn urgent_stamina_spend(
    state: &GameState,
    team: Team,
    tank: &Tank,
    pressures: PressureSignals,
) -> bool {
    if pressures.own_goal > 0.5 {
        return true;
    }
    if should_recover_critical_stamina(state, team, tank, pressures) {
        return false;
    }
    is_loose_ball_contest(state, team, tank) || decisive_ball_contact(state, team, tank, pressures)
}

fn should_recover_critical_stamina(
    state: &GameState,
    team: Team,
    tank: &Tank,
    pressures: PressureSignals,
) -> bool {
    if stamina_ratio(tank) > CRITICAL_STAMINA_RATIO {
        return false;
    }
    if pressures.own_goal > 0.35 || is_clinching_finish(state, team) {
        return false;
    }
    if hypot(state.ball.velocity.x, state.ball.velocity.y) > 80.0 {
        return false;
    }

    let ball_distance = distance(tank.position, state.ball.position);
    if ball_distance > TANK_RADIUS + BALL_RADIUS + DECISIVE_CONTACT_BUFFER {
        return false;
    }

    let opponent = nearest_opponent_tank(state, team);
    let opponent_distance = distance(opponent.position, state.ball.position);
    opponent_distance <= ball_distance + TANK_RADIUS * 0.75
}

fn is_clinching_finish(state: &GameState, team: Team) -> bool {
    let sign = team_sign(team);
    let attack_x = (state.ball.position.x - FIELD_LENGTH / 2.0) * sign + FIELD_LENGTH / 2.0;
    let lane = (state.ball.position.y - FIELD_WIDTH / 2.0).abs() < GOAL_MOUTH * 0.46;
    let attack_velocity = state.ball.velocity.x * sign;
    attack_x > FIELD_LENGTH - 270.0 && lane && attack_velocity > -25.0
}

fn should_use_tactical_rollout(
    state: &GameState,
    team: Team,
    tank: &Tank,
    pressures: PressureSignals,
) -> bool {
    if decisive_ball_contact(state, team, tank, pressures) || pressures.own_goal > 0.5 {
        return true;
    }
    if pressures.attack_corner.max(pressures.own_corner) > 0.52 {
        if pressures.attack_corner >= pressures.own_corner
            && opponent_near_ball(state, team, TANK_RADIUS * 1.9)
        {
            return false;
        }
        return true;
    }
    pressures.side_wall > 0.72 && distance(tank.position, state.ball.position) < TANK_RADIUS * 3.4
}

fn regulate_critical_stamina_command(
    state: &GameState,
    team: Team,
    tank: &Tank,
    pressures: PressureSignals,
    command: Command,
) -> Command {
    if stamina_ratio(tank) > CRITICAL_STAMINA_RATIO || pressures.own_goal > 0.5 {
        return command;
    }
    if command.left.abs() + command.right.abs() <= 1.0 {
        return command;
    }

    let local = target_in_tank_frame(tank, team, state.ball.position);
    if local.1.abs() < TANK_RADIUS * 0.18 {
        return if (command.left - command.right).abs() < 1e-9 {
            Command {
                left: command.left,
                right: 0.0,
            }
        } else {
            stop_command()
        };
    }

    if local.1 > 0.0 {
        Command {
            left: command.left,
            right: 0.0,
        }
    } else {
        Command {
            left: 0.0,
            right: command.right,
        }
    }
}

fn is_loose_ball_contest(state: &GameState, team: Team, tank: &Tank) -> bool {
    let ball_distance = distance(tank.position, state.ball.position);
    if ball_distance > TANK_RADIUS * 4.1 {
        return false;
    }
    let opponent = nearest_opponent_tank(state, team);
    let opponent_distance = distance(opponent.position, state.ball.position);
    let ball_speed = hypot(state.ball.velocity.x, state.ball.velocity.y);
    let midfield = (state.ball.position.x - FIELD_LENGTH / 2.0).abs() < FIELD_LENGTH * 0.25;
    let contested = opponent_distance < ball_distance + TANK_RADIUS * 1.25;
    ball_speed < 160.0 && (midfield || contested)
}

fn opponent_near_ball(state: &GameState, team: Team, range: f64) -> bool {
    let opponent = nearest_opponent_tank(state, team);
    distance(opponent.position, state.ball.position) <= range
}

fn decisive_ball_contact(
    state: &GameState,
    _team: Team,
    tank: &Tank,
    pressures: PressureSignals,
) -> bool {
    let contact_distance = TANK_RADIUS + BALL_RADIUS + DECISIVE_CONTACT_BUFFER;
    if distance(tank.position, state.ball.position) > contact_distance {
        return false;
    }
    if pressures.attack_corner.max(pressures.own_corner) > 0.52 && pressures.own_goal <= 0.35 {
        return false;
    }
    pressures.finishing > 0.35 || pressures.own_goal > 0.35
}

fn choose_tactical_action(state: &GameState, team: Team, policy_action: usize) -> usize {
    let policy_action = policy_action.min(OUTPUT_COUNT - 1);
    let policy_score = score_tactical_action(state, team, policy_action, TACTICAL_ROLLOUT_FRAMES);
    let mut best_action = policy_action;
    let mut best_score = policy_score;

    for action in 0..OUTPUT_COUNT {
        if action == policy_action {
            continue;
        }
        let score = score_tactical_action(state, team, action, TACTICAL_ROLLOUT_FRAMES);
        if score > best_score + 1e-9 {
            best_action = action;
            best_score = score;
        }
    }

    if best_score > policy_score + TACTICAL_IMPROVEMENT_MARGIN {
        best_action
    } else {
        policy_action
    }
}

fn score_tactical_action(
    state: &GameState,
    team: Team,
    action: usize,
    rollout_frames: usize,
) -> f64 {
    let mut simulated = state.clone();
    let before = evaluate_position(&simulated, team);
    let opponent_action = 4;
    for _ in 0..rollout_frames.max(1) {
        let red_command = if team == Team::Red {
            action_index_to_command(action)
        } else {
            action_index_to_command(opponent_action)
        };
        let blue_command = if team == Team::Blue {
            action_index_to_command(action)
        } else {
            action_index_to_command(opponent_action)
        };
        step_game(&mut simulated, red_command, blue_command, FIXED_DT);
    }

    let after = evaluate_position(&simulated, team);
    let delta = evaluate_position_delta(&simulated, state, team);
    let command = action_index_to_command(action);
    after.total - before.total + delta.corner_escape * 0.45
        - (command.left.abs() + command.right.abs()) * 0.004
}

fn evaluate_position(state: &GameState, team: Team) -> PositionEvaluation {
    let sign = team_sign(team);
    let ball_attack_x = attack_x(team, state.ball.position.x);
    let lane = goal_lane_score(state.ball.position.y);
    let goal_diff = match team {
        Team::Red => state.score_red - state.score_blue,
        Team::Blue => state.score_blue - state.score_red,
    } as f64;
    let ball_progress = normalize_signed(ball_attack_x - FIELD_LENGTH / 2.0, FIELD_LENGTH / 2.0);
    let goal_proximity = clamp01((ball_attack_x - FIELD_LENGTH * 0.68) / (FIELD_LENGTH * 0.32));
    let shot_velocity = clamp_signed(state.ball.velocity.x * sign / 420.0);
    let finish_threat = finish_threat_score(state, team, lane, goal_proximity);
    let own_goal_proximity = clamp01((FIELD_LENGTH * 0.34 - ball_attack_x) / (FIELD_LENGTH * 0.34));
    let incoming_own_goal = clamp01((-state.ball.velocity.x * sign) / 340.0);
    let own_danger = clamp01((own_goal_proximity * 0.74 + incoming_own_goal * 0.42) * lane);
    let tank = controlled_tank(state, team);
    let possession = possession_score(state, team, tank);
    let contest = contest_score(state, team, tank);
    let corner_escape = corner_position_score(state, team);
    let stamina = stamina_position_score(tank, own_danger, goal_proximity, possession);
    let shot_lane = lane * goal_proximity;
    let total = goal_diff * 12.0
        + ball_progress * 1.7
        + shot_lane * 0.95
        + finish_threat * 1.05
        + shot_velocity * 0.42
        + contest * 0.7
        + possession * 1.1
        - own_danger * 2.2
        + corner_escape * 0.85
        + stamina * 0.22;
    PositionEvaluation {
        total,
        corner_escape,
    }
}

fn evaluate_position_delta(state: &GameState, initial: &GameState, team: Team) -> PositionEvaluation {
    let after = evaluate_position(state, team);
    let before = evaluate_position(initial, team);
    let corner_escape = corner_escape_gain(state, initial, team);
    PositionEvaluation {
        total: after.total - before.total,
        corner_escape,
    }
}

fn finish_threat_score(state: &GameState, team: Team, lane: f64, goal_proximity: f64) -> f64 {
    let sign = team_sign(team);
    let attack_velocity = state.ball.velocity.x * sign;
    let speed_toward_goal = clamp01(attack_velocity / 260.0);
    let near_goal = clamp01((attack_x(team, state.ball.position.x) - (FIELD_LENGTH - 260.0)) / 160.0);
    let goal_factor = 0.45 + goal_proximity * 0.55;
    let y_velocity_drift = state.ball.velocity.y.abs() / attack_velocity.abs().max(1.0);
    let straight_shot = clamp01(1.0 - y_velocity_drift * 1.15);
    clamp01(lane * near_goal * goal_factor * (0.2 + speed_toward_goal * 0.8) * straight_shot)
}

fn possession_score(state: &GameState, team: Team, tank: &Tank) -> f64 {
    let ball = state.ball.position;
    let goal = goal_point(team);
    let shot = unit_vector(goal.x - ball.x, goal.y - ball.y);
    let setup_distance = BALL_RADIUS + TANK_RADIUS + 10.0;
    let setup = Vec2 {
        x: clamp_range(
            ball.x - shot.x * setup_distance,
            TANK_RADIUS,
            FIELD_LENGTH - TANK_RADIUS,
        ),
        y: clamp_range(
            ball.y - shot.y * setup_distance,
            TANK_RADIUS,
            FIELD_WIDTH - TANK_RADIUS,
        ),
    };
    let setup_distance_ratio = distance(tank.position, setup) / FIELD_LENGTH;
    let ball_distance = distance(tank.position, ball);
    let contact = if ball_distance <= BALL_RADIUS + TANK_RADIUS + 20.0 {
        0.48
    } else {
        0.0
    };
    let tank_to_ball = unit_vector(ball.x - tank.position.x, ball.y - tank.position.y);
    let behind_ball = clamp_signed(tank_to_ball.x * shot.x + tank_to_ball.y * shot.y);
    let heading_to_ball = (ball.y - tank.position.y).atan2(ball.x - tank.position.x);
    let heading_alignment = normalize_angle(heading_to_ball - tank.angle).cos();
    let shot_alignment = normalize_angle(shot.y.atan2(shot.x) - tank.angle).cos();
    let opponent = nearest_opponent_tank(state, team);
    let opponent_pressure = clamp01(1.0 - distance(opponent.position, ball) / (TANK_RADIUS * 3.2));

    clamp_signed(
        -setup_distance_ratio * 1.6
            + contact
            + behind_ball * 0.36
            + heading_alignment * 0.22
            + shot_alignment * 0.2
            - opponent_pressure * 0.22,
    )
}

fn contest_score(state: &GameState, team: Team, tank: &Tank) -> f64 {
    let ball = state.ball.position;
    let ball_distance = distance(tank.position, ball);
    let close = 1.0 - clamp01(ball_distance / (TANK_RADIUS * 3.2));
    let heading_to_ball = (ball.y - tank.position.y).atan2(ball.x - tank.position.x);
    let nose_alignment = normalize_angle(heading_to_ball - tank.angle).cos();
    let opponent = nearest_opponent_tank(state, team);
    let opponent_pressure = clamp01(1.0 - distance(opponent.position, ball) / (TANK_RADIUS * 2.4));
    let pinned = if is_corner_pinned(state, team) { 0.42 } else { 0.0 };
    let loose_midfield = if (ball.x - FIELD_LENGTH / 2.0).abs() < FIELD_LENGTH * 0.24
        && hypot(state.ball.velocity.x, state.ball.velocity.y) < 140.0
    {
        0.28
    } else {
        0.0
    };

    clamp_signed(close * 0.75 + nose_alignment * 0.38 + opponent_pressure * 0.26 + pinned + loose_midfield - 0.34)
}

fn corner_position_score(state: &GameState, team: Team) -> f64 {
    let ball_attack_x = attack_x(team, state.ball.position.x);
    let attacking_corner = ball_attack_x > FIELD_LENGTH - BALL_RADIUS - 90.0;
    let defending_corner = ball_attack_x < BALL_RADIUS + 115.0;
    if !attacking_corner && !defending_corner {
        return 0.0;
    }

    let side_distance = side_wall_distance(state.ball.position.y);
    let center_score = 1.0 - clamp01((state.ball.position.y - FIELD_WIDTH / 2.0).abs() / (FIELD_WIDTH / 2.0));
    let wall_penalty = clamp01((BALL_RADIUS + 48.0 - side_distance) / (BALL_RADIUS + 48.0));
    clamp_signed(center_score - wall_penalty * 0.68)
}

fn corner_escape_gain(state: &GameState, initial: &GameState, team: Team) -> f64 {
    if !is_corner_pinned(initial, team) {
        return 0.0;
    }

    let wall_clear_gain =
        (side_wall_distance(state.ball.position.y) - side_wall_distance(initial.ball.position.y))
            / (FIELD_WIDTH / 2.0);
    let center_gain = ((initial.ball.position.y - FIELD_WIDTH / 2.0).abs()
        - (state.ball.position.y - FIELD_WIDTH / 2.0).abs())
        / (FIELD_WIDTH / 2.0);
    let progress_gain =
        (attack_x(team, state.ball.position.x) - attack_x(team, initial.ball.position.x))
            / FIELD_LENGTH;
    clamp_signed(wall_clear_gain * 1.35 + center_gain * 1.05 + progress_gain * 0.45)
}

fn is_corner_pinned(state: &GameState, team: Team) -> bool {
    let ball_attack_x = attack_x(team, state.ball.position.x);
    let near_end =
        ball_attack_x > FIELD_LENGTH - BALL_RADIUS - 100.0 || ball_attack_x < BALL_RADIUS + 100.0;
    near_end && side_wall_distance(state.ball.position.y) < BALL_RADIUS + 58.0
}

fn stamina_position_score(tank: &Tank, own_danger: f64, goal_proximity: f64, possession: f64) -> f64 {
    let ratio = stamina_ratio(tank);
    let urgent = own_danger > 0.45 || goal_proximity > 0.55 || possession > 0.25;
    if urgent {
        ratio * 0.2
    } else {
        ratio - 0.5
    }
}

fn controlled_tank(state: &GameState, team: Team) -> &Tank {
    state
        .tanks
        .iter()
        .find(|candidate| candidate.team == team)
        .unwrap()
}

fn nearest_opponent_tank(state: &GameState, team: Team) -> &Tank {
    state
        .tanks
        .iter()
        .filter(|candidate| candidate.team != team)
        .min_by(|a, b| {
            distance(a.position, state.ball.position)
                .partial_cmp(&distance(b.position, state.ball.position))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

fn goal_lane_score(y: f64) -> f64 {
    1.0 - clamp01((y - FIELD_WIDTH / 2.0).abs() / (GOAL_MOUTH * 0.72))
}

fn side_wall_distance(y: f64) -> f64 {
    (y - BALL_RADIUS).min(FIELD_WIDTH - BALL_RADIUS - y)
}

fn attack_x(team: Team, x: f64) -> f64 {
    match team {
        Team::Red => x,
        Team::Blue => FIELD_LENGTH - x,
    }
}

fn unit_vector(x: f64, y: f64) -> Vec2 {
    let length = hypot(x, y).max(1.0);
    Vec2 {
        x: x / length,
        y: y / length,
    }
}

fn traditional_team_command(state: &GameState, team: Team) -> Command {
    let tank = controlled_tank(state, team);
    let target = traditional_tactical_target(state, team, tank);
    let urgent_defense = traditional_ball_threatens_own_goal(state, team);
    let spend_low_stamina = urgent_defense || traditional_useful_ball_contact(tank, target, state, team);
    traditional_drive_to(tank, target, spend_low_stamina)
}

fn traditional_tactical_target(state: &GameState, team: Team, tank: &Tank) -> Vec2 {
    let ball = state.ball.position;
    if traditional_is_wrong_side_own_goal_push(tank, ball, team) {
        return traditional_own_goal_escape_target(tank, ball, team);
    }
    if traditional_ball_threatens_own_goal(state, team) {
        return traditional_defensive_target(state, team);
    }
    if traditional_opponent_corner_trap(ball, team) {
        return traditional_opponent_corner_target(state, team, tank);
    }
    if traditional_ball_near_side_wall(ball) && traditional_tank_near_ball(tank, state) {
        return traditional_side_wall_recycle_target(ball, team);
    }
    traditional_attack_target(state, team, tank)
}

fn traditional_defensive_target(state: &GameState, team: Team) -> Vec2 {
    let sign = team_sign(team);
    let own_x = if team == Team::Red { 0.0 } else { FIELD_LENGTH };
    clamp_tank_point(Vec2 {
        x: own_x + sign * TRADITIONAL_DEFENSE_X,
        y: traditional_predict_goal_lane_y(state, team),
    })
}

fn traditional_attack_target(state: &GameState, team: Team, tank: &Tank) -> Vec2 {
    let ball = state.ball.position;
    let shot = traditional_attack_shot(state, team);
    let readiness = traditional_shot_readiness(tank, ball, shot);
    let approach_target = clamp_tank_point(Vec2 {
        x: ball.x - shot.x * TRADITIONAL_STRIKE_SETUP_DISTANCE,
        y: ball.y - shot.y * TRADITIONAL_STRIKE_SETUP_DISTANCE,
    });
    let aligned = readiness.x > BALL_RADIUS * 0.35
        && readiness.y.abs() < TRADITIONAL_STRIKE_LATERAL_TOLERANCE;
    let close_to_setup = distance(tank.position, approach_target) < TRADITIONAL_STRIKE_APPROACH_TOLERANCE
        || traditional_tank_near_ball(tank, state);
    if aligned && close_to_setup {
        goal_point(team)
    } else {
        approach_target
    }
}

fn traditional_attack_shot(state: &GameState, team: Team) -> Vec2 {
    traditional_unit_vector(state.ball.position, goal_point(team), team_sign(team))
}

fn traditional_opponent_corner_target(state: &GameState, team: Team, tank: &Tank) -> Vec2 {
    let ball = state.ball.position;
    let sign = team_sign(team);
    let inward = if ball.y < FIELD_WIDTH / 2.0 { 1.0 } else { -1.0 };
    if traditional_tank_near_ball(tank, state) && sign * (ball.x - tank.position.x) > -BALL_RADIUS {
        return clamp_tank_point(Vec2 {
            x: ball.x - sign * 48.0,
            y: ball.y + inward * 260.0,
        });
    }
    clamp_tank_point(Vec2 {
        x: ball.x - sign * (BALL_RADIUS + TANK_LENGTH + 92.0),
        y: ball.y + inward * 230.0,
    })
}

fn traditional_side_wall_recycle_target(ball: Vec2, team: Team) -> Vec2 {
    let inward = if ball.y < FIELD_WIDTH / 2.0 { 1.0 } else { -1.0 };
    clamp_tank_point(Vec2 {
        x: ball.x - team_sign(team) * 70.0,
        y: ball.y + inward * 210.0,
    })
}

fn traditional_own_goal_escape_target(tank: &Tank, ball: Vec2, team: Team) -> Vec2 {
    let sign = team_sign(team);
    let side = if tank.position.y <= ball.y { -1.0 } else { 1.0 };
    clamp_tank_point(Vec2 {
        x: ball.x + sign * (BALL_RADIUS + TANK_LENGTH + 80.0),
        y: ball.y + side * (BALL_RADIUS + TANK_WIDTH + 55.0),
    })
}

fn traditional_is_wrong_side_own_goal_push(tank: &Tank, ball: Vec2, team: Team) -> bool {
    let sign = team_sign(team);
    let own_x = if team == Team::Red { 0.0 } else { FIELD_LENGTH };
    let ball_deep = sign * (ball.x - own_x) < TRADITIONAL_OWN_GOAL_AVOID_DEPTH;
    let in_goal_lane = (ball.y - FIELD_WIDTH / 2.0).abs() < GOAL_MOUTH * 0.72;
    let tank_on_attack_side = sign * (tank.position.x - ball.x) > BALL_RADIUS * 0.45;
    let facing_own_goal = tank.angle.cos() * sign < -0.32;
    ball_deep && in_goal_lane && tank_on_attack_side && facing_own_goal
}

fn traditional_useful_ball_contact(tank: &Tank, target: Vec2, state: &GameState, team: Team) -> bool {
    if !traditional_tank_near_ball(tank, state) {
        return false;
    }
    let ball = state.ball.position;
    let shot = traditional_attack_shot(state, team);
    let readiness = traditional_shot_readiness(tank, ball, shot);
    let to_ball = traditional_unit_vector(tank.position, ball, team_sign(team));
    let to_target = traditional_unit_vector(tank.position, target, team_sign(team));
    let moving_through_ball = dot_vec(to_ball, to_target) > 0.72;
    let attacking_contact = readiness.x > -BALL_RADIUS * 0.25
        && readiness.y.abs() < TRADITIONAL_STRIKE_LATERAL_TOLERANCE;
    moving_through_ball && (attacking_contact || traditional_ball_in_own_danger_lane(ball, team))
}

fn traditional_ball_threatens_own_goal(state: &GameState, team: Team) -> bool {
    let sign = team_sign(team);
    let moving_toward_own_goal = state.ball.velocity.x * sign < -60.0;
    let deep_in_lane = traditional_ball_in_own_danger_lane(state.ball.position, team);
    if deep_in_lane {
        return true;
    }
    if !moving_toward_own_goal {
        return false;
    }
    (traditional_predict_goal_lane_y(state, team) - FIELD_WIDTH / 2.0).abs() < GOAL_MOUTH * 0.72
}

fn traditional_ball_in_own_danger_lane(ball: Vec2, team: Team) -> bool {
    let sign = team_sign(team);
    let own_x = if team == Team::Red { 0.0 } else { FIELD_LENGTH };
    sign * (ball.x - own_x) < TRADITIONAL_DANGER_DEPTH
        && (ball.y - FIELD_WIDTH / 2.0).abs() < GOAL_MOUTH * 0.72
}

fn traditional_predict_goal_lane_y(state: &GameState, team: Team) -> f64 {
    let sign = team_sign(team);
    let own_x = if team == Team::Red { 0.0 } else { FIELD_LENGTH };
    let own_block_x = own_x + sign * TRADITIONAL_DEFENSE_X;
    let moving_toward_own_goal = state.ball.velocity.x * sign < -10.0;
    let seconds = if moving_toward_own_goal {
        clamp_range(
            (own_block_x - state.ball.position.x) / state.ball.velocity.x,
            0.0,
            TRADITIONAL_BALL_PREDICT_SECONDS,
        )
    } else {
        0.0
    };
    clamp_range(
        state.ball.position.y + state.ball.velocity.y * seconds,
        FIELD_WIDTH / 2.0 - GOAL_MOUTH * 0.48,
        FIELD_WIDTH / 2.0 + GOAL_MOUTH * 0.48,
    )
}

fn traditional_opponent_corner_trap(ball: Vec2, team: Team) -> bool {
    let opponent_depth = team_sign(team) * (goal_point(team).x - ball.x);
    opponent_depth < TRADITIONAL_OPPONENT_CORNER_DEPTH && traditional_ball_near_side_wall(ball)
}

fn traditional_ball_near_side_wall(ball: Vec2) -> bool {
    ball.y < TRADITIONAL_SIDE_WALL_DEPTH || ball.y > FIELD_WIDTH - TRADITIONAL_SIDE_WALL_DEPTH
}

fn traditional_tank_near_ball(tank: &Tank, state: &GameState) -> bool {
    distance(tank.position, state.ball.position) <= TANK_RADIUS + BALL_RADIUS + TRADITIONAL_NEAR_BALL_BUFFER
}

fn traditional_shot_readiness(tank: &Tank, ball: Vec2, shot: Vec2) -> Vec2 {
    let to_ball = Vec2 {
        x: ball.x - tank.position.x,
        y: ball.y - tank.position.y,
    };
    Vec2 {
        x: to_ball.x * shot.x + to_ball.y * shot.y,
        y: to_ball.x * -shot.y + to_ball.y * shot.x,
    }
}

fn traditional_drive_to(tank: &Tank, target: Vec2, spend_low_stamina: bool) -> Command {
    if !spend_low_stamina && stamina_ratio(tank) < TRADITIONAL_STAMINA_CONSERVE_RATIO {
        return stop_command();
    }
    let dx = target.x - tank.position.x;
    let dy = target.y - tank.position.y;
    if hypot(dx, dy) < 14.0 {
        return stop_command();
    }

    let desired = dy.atan2(dx);
    let forward_error = normalize_angle(desired - tank.angle);
    let reverse_error = normalize_angle(desired + std::f64::consts::PI - tank.angle);
    let reverse = reverse_error.abs() + 0.18 < forward_error.abs();
    let heading_error = if reverse { reverse_error } else { forward_error };
    let base = if reverse { -1.0 } else { 1.0 };

    if heading_error.abs() < TRADITIONAL_STRAIGHT_HEADING_TOLERANCE {
        return Command { left: base, right: base };
    }
    if heading_error.abs() > 1.35 {
        return if heading_error > 0.0 {
            Command { left: 1.0, right: -1.0 }
        } else {
            Command { left: -1.0, right: 1.0 }
        };
    }
    if heading_error > 0.0 {
        return if reverse {
            Command { left: 0.0, right: base }
        } else {
            Command { left: base, right: 0.0 }
        };
    }
    if reverse {
        Command { left: base, right: 0.0 }
    } else {
        Command { left: 0.0, right: base }
    }
}

fn traditional_unit_vector(from: Vec2, to: Vec2, fallback_x: f64) -> Vec2 {
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let length = hypot(dx, dy);
    if length == 0.0 {
        return Vec2 {
            x: fallback_x,
            y: 0.0,
        };
    }
    Vec2 {
        x: dx / length,
        y: dy / length,
    }
}

fn dot_vec(a: Vec2, b: Vec2) -> f64 {
    a.x * b.x + a.y * b.y
}

fn decision_return(
    decision: &PendingDecision,
    goals: &[GoalEvent],
    red_diff: i32,
    discount: f64,
    goal_reward: f64,
    win_reward: f64,
) -> f64 {
    let mut total_return = 0.0;
    for goal in goals {
        if goal.frame < decision.frame {
            continue;
        }
        let sign = if goal.team == decision.team { 1.0 } else { -1.0 };
        let ticks_ahead = (goal.frame.saturating_sub(decision.frame) as f64)
            / (PHYSICS_HZ as f64 / AI_HZ as f64).max(1.0);
        total_return += sign * goal_reward * discount.powf(ticks_ahead);
    }

    let team_diff = match decision.team {
        Team::Red => red_diff,
        Team::Blue => -red_diff,
    };
    total_return + (team_diff as f64).signum() * win_reward
}

fn normalized_advantages(decisions: &[(PendingDecision, f64)], baseline: AdvantageBaseline) -> Vec<f64> {
    if decisions.is_empty() {
        return Vec::new();
    }

    let trainable: Vec<&(PendingDecision, f64)> = decisions
        .iter()
        .filter(|(decision, _)| decision.trainable)
        .collect();
    let population: Vec<&(PendingDecision, f64)> = if trainable.is_empty() {
        decisions.iter().collect()
    } else {
        trainable
    };
    if baseline == AdvantageBaseline::Learned {
        return learned_value_advantages(decisions, &population);
    }

    let global = advantage_stats(population.iter().map(|(_, value)| *value).collect());
    let groups = if baseline == AdvantageBaseline::StartTeamTime {
        grouped_advantage_stats(&population)
    } else {
        Vec::new()
    };

    decisions
        .iter()
        .map(|(decision, value)| {
            let stats = groups
                .iter()
                .find(|(key, _)| *key == advantage_group_key(decision))
                .map(|(_, stats)| *stats)
                .unwrap_or(global);
            normalize_return(*value, stats)
        })
        .collect()
}

fn learned_value_advantages(
    decisions: &[(PendingDecision, f64)],
    population: &[&(PendingDecision, f64)],
) -> Vec<f64> {
    let returns: Vec<f64> = population.iter().map(|(_, value)| *value).collect();
    let stats = advantage_stats(returns);
    let value_weights = fit_linear_value_baseline(population, stats.mean);
    let residuals: Vec<f64> = population
        .iter()
        .map(|(decision, value)| *value - predict_linear_value(&decision.inputs, &value_weights))
        .collect();
    let residual_mean = if residuals.is_empty() {
        0.0
    } else {
        residuals.iter().sum::<f64>() / residuals.len() as f64
    };
    let scale = stats.std.max(1e-6);

    decisions
        .iter()
        .map(|(decision, value)| {
            (*value - predict_linear_value(&decision.inputs, &value_weights) - residual_mean) / scale
        })
        .collect()
}

fn fit_linear_value_baseline(
    population: &[&(PendingDecision, f64)],
    fallback_mean: f64,
) -> Vec<f64> {
    let mut weights = vec![0.0; INPUT_COUNT + 1];
    weights[INPUT_COUNT] = fallback_mean;
    if population.len() < 2 {
        return weights;
    }

    let learning_rate = 0.02;
    let l2 = 0.001;
    for _ in 0..80 {
        for (decision, value) in population {
            let prediction = predict_linear_value(&decision.inputs, &weights);
            let error = prediction - *value;
            for index in 0..INPUT_COUNT {
                weights[index] -= learning_rate * (error * decision.inputs[index] + l2 * weights[index]);
            }
            weights[INPUT_COUNT] -= learning_rate * error;
        }
    }

    weights
}

fn predict_linear_value(inputs: &[f64; INPUT_COUNT], weights: &[f64]) -> f64 {
    let mut value = weights[INPUT_COUNT];
    for index in 0..INPUT_COUNT {
        value += inputs[index] * weights[index];
    }
    value
}

#[derive(Clone, Copy)]
struct AdvantageStats {
    mean: f64,
    std: f64,
}

fn advantage_stats(values: Vec<f64>) -> AdvantageStats {
    if values.is_empty() {
        return AdvantageStats { mean: 0.0, std: 0.0 };
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (*value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    AdvantageStats {
        mean,
        std: variance.sqrt(),
    }
}

fn grouped_advantage_stats(population: &[&(PendingDecision, f64)]) -> Vec<(String, AdvantageStats)> {
    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for (decision, value) in population {
        let key = advantage_group_key(decision);
        if let Some((_, values)) = groups.iter_mut().find(|(candidate, _)| *candidate == key) {
            values.push(*value);
        } else {
            groups.push((key, vec![*value]));
        }
    }
    groups
        .into_iter()
        .filter_map(|(key, values)| {
            if values.len() < 2 {
                None
            } else {
                Some((key, advantage_stats(values)))
            }
        })
        .collect()
}

fn advantage_group_key(decision: &PendingDecision) -> String {
    let start = match decision.start_state_mode {
        ActualStartStateMode::Open => "open",
        ActualStartStateMode::OutcomeCurriculum => "outcome-curriculum",
        ActualStartStateMode::OwnGoalDefense => "own-goal-defense",
        ActualStartStateMode::CornerFight => "corner-fight",
        ActualStartStateMode::LooseBallContest => "loose-ball-contest",
    };
    let team = match decision.team {
        Team::Red => "red",
        Team::Blue => "blue",
    };
    let time_bucket = decision.frame / (PHYSICS_HZ * 5).max(1);
    format!("{start}:{team}:{time_bucket}")
}

fn normalize_return(value: f64, stats: AdvantageStats) -> f64 {
    if stats.std < 1e-6 {
        value - stats.mean
    } else {
        (value - stats.mean) / stats.std
    }
}

fn initial_state() -> GameState {
    GameState {
        frame: 0,
        time: 0.0,
        tanks: [
            Tank {
                team: Team::Red,
                position: Vec2 {
                    x: 170.0,
                    y: FIELD_WIDTH / 2.0,
                },
                velocity: Vec2 { x: 0.0, y: 0.0 },
                angle: 0.0,
                angular_velocity: 0.0,
                stamina: TANK_STAMINA,
            },
            Tank {
                team: Team::Blue,
                position: Vec2 {
                    x: FIELD_LENGTH - 170.0,
                    y: FIELD_WIDTH / 2.0,
                },
                velocity: Vec2 { x: 0.0, y: 0.0 },
                angle: std::f64::consts::PI,
                angular_velocity: 0.0,
                stamina: TANK_STAMINA,
            },
        ],
        ball: Ball {
            position: Vec2 {
                x: FIELD_LENGTH / 2.0,
                y: FIELD_WIDTH / 2.0,
            },
            velocity: Vec2 { x: 0.0, y: 0.0 },
        },
        score_red: 0,
        score_blue: 0,
        last_goal: None,
    }
}

fn seeded_initial_state(
    random: &mut SeededRandom,
    match_index: usize,
    mode: ActualStartStateMode,
) -> GameState {
    let mut state = initial_state();
    let y_jitter = (random.next() - 0.5) * FIELD_WIDTH * 0.32;
    let x_jitter = (random.next() - 0.5) * FIELD_LENGTH * 0.18;

    state.ball.position = Vec2 {
        x: FIELD_LENGTH / 2.0 + x_jitter,
        y: FIELD_WIDTH / 2.0 + y_jitter,
    };
    state.ball.velocity = Vec2 {
        x: (random.next() - 0.5) * 120.0,
        y: (random.next() - 0.5) * 120.0,
    };
    state.tanks[0].position = Vec2 {
        x: 160.0 + random.next() * 110.0,
        y: FIELD_WIDTH / 2.0 + (random.next() - 0.5) * FIELD_WIDTH * 0.18,
    };
    state.tanks[0].angle = (random.next() - 0.5) * 0.4;
    state.tanks[1].position = Vec2 {
        x: FIELD_LENGTH - 160.0 - random.next() * 110.0,
        y: FIELD_WIDTH / 2.0 + (random.next() - 0.5) * FIELD_WIDTH * 0.18,
    };
    state.tanks[1].angle = std::f64::consts::PI + (random.next() - 0.5) * 0.4;

    if mode == ActualStartStateMode::OutcomeCurriculum {
        place_outcome_curriculum_state(&mut state, match_index, random);
    } else if mode == ActualStartStateMode::OwnGoalDefense {
        place_own_goal_defense_state(&mut state, match_index, random);
    } else if mode == ActualStartStateMode::CornerFight {
        place_corner_fight_state(&mut state, match_index, random);
    } else if mode == ActualStartStateMode::LooseBallContest {
        place_loose_ball_contest_state(&mut state, match_index, random);
    }

    state
}

fn resolve_start_state_mode(
    mode: StartStateMode,
    match_index: usize,
    open_start_ratio: Option<f64>,
) -> ActualStartStateMode {
    match mode {
        StartStateMode::Open => ActualStartStateMode::Open,
        StartStateMode::OutcomeCurriculum => ActualStartStateMode::OutcomeCurriculum,
        StartStateMode::OwnGoalDefense => ActualStartStateMode::OwnGoalDefense,
        StartStateMode::CornerFight => ActualStartStateMode::CornerFight,
        StartStateMode::LooseBallContest => ActualStartStateMode::LooseBallContest,
        StartStateMode::Mixed => {
            if let Some(ratio) = open_start_ratio {
                return resolve_weighted_mixed_start_state_mode(match_index, ratio);
            }
            match match_index % 5 {
                0 => ActualStartStateMode::Open,
                1 => ActualStartStateMode::OutcomeCurriculum,
                2 => ActualStartStateMode::OwnGoalDefense,
                3 => ActualStartStateMode::CornerFight,
                _ => ActualStartStateMode::LooseBallContest,
            }
        }
    }
}

fn resolve_weighted_mixed_start_state_mode(match_index: usize, open_start_ratio: f64) -> ActualStartStateMode {
    let cycle_slots = 20_usize;
    let open_slots = (clamp01(open_start_ratio) * cycle_slots as f64).round() as usize;
    let slot = (match_index * 7) % cycle_slots;
    if slot < open_slots {
        return ActualStartStateMode::Open;
    }
    let remaining_slots = cycle_slots.saturating_sub(open_slots);
    if remaining_slots == 0 {
        return ActualStartStateMode::Open;
    }
    match ((slot - open_slots) * 4 / remaining_slots).min(3) {
        0 => ActualStartStateMode::OutcomeCurriculum,
        1 => ActualStartStateMode::OwnGoalDefense,
        2 => ActualStartStateMode::CornerFight,
        _ => ActualStartStateMode::LooseBallContest,
    }
}

fn place_outcome_curriculum_state(
    state: &mut GameState,
    match_index: usize,
    random: &mut SeededRandom,
) {
    let attacking_team = if match_index % 2 == 0 {
        Team::Red
    } else {
        Team::Blue
    };
    let scenario = match_index % 4;
    if scenario == 0 {
        place_ball_in_team_frame(
            state,
            attacking_team,
            FIELD_LENGTH - BALL_RADIUS - 3.0,
            FIELD_WIDTH / 2.0 + (random.next() - 0.5) * GOAL_MOUTH * 0.45,
            190.0 + random.next() * 70.0,
            (random.next() - 0.5) * 20.0,
        );
    } else if scenario == 1 {
        place_ball_in_team_frame(
            state,
            opponent_team(attacking_team),
            FIELD_LENGTH - BALL_RADIUS - 3.0,
            FIELD_WIDTH / 2.0 + (random.next() - 0.5) * GOAL_MOUTH * 0.45,
            190.0 + random.next() * 70.0,
            (random.next() - 0.5) * 20.0,
        );
    } else if scenario == 2 {
        let side = if random.next() < 0.5 { -1.0 } else { 1.0 };
        place_ball_in_team_frame(
            state,
            attacking_team,
            FIELD_LENGTH - BALL_RADIUS - 28.0 - random.next() * 24.0,
            if side < 0.0 {
                BALL_RADIUS + 8.0 + random.next() * 18.0
            } else {
                FIELD_WIDTH - BALL_RADIUS - 8.0 - random.next() * 18.0
            },
            40.0 + random.next() * 60.0,
            side * (random.next() * 40.0),
        );
    } else {
        place_ball_in_team_frame(
            state,
            attacking_team,
            FIELD_LENGTH / 2.0 + random.next() * 80.0,
            FIELD_WIDTH / 2.0 + (random.next() - 0.5) * FIELD_WIDTH * 0.25,
            80.0 + random.next() * 80.0,
            (random.next() - 0.5) * 60.0,
        );
    }

    state.tanks[0].position = field_point(
        Team::Red,
        210.0 + random.next() * 70.0,
        FIELD_WIDTH / 2.0 + (random.next() - 0.5) * 140.0,
    );
    state.tanks[0].angle = field_angle(Team::Red, (random.next() - 0.5) * 0.5);
    state.tanks[0].velocity = Vec2 { x: 0.0, y: 0.0 };
    state.tanks[0].angular_velocity = 0.0;
    state.tanks[0].stamina = TANK_STAMINA;

    state.tanks[1].position = field_point(
        Team::Blue,
        210.0 + random.next() * 70.0,
        FIELD_WIDTH / 2.0 + (random.next() - 0.5) * 140.0,
    );
    state.tanks[1].angle = field_angle(Team::Blue, (random.next() - 0.5) * 0.5);
    state.tanks[1].velocity = Vec2 { x: 0.0, y: 0.0 };
    state.tanks[1].angular_velocity = 0.0;
    state.tanks[1].stamina = TANK_STAMINA;
}

fn place_own_goal_defense_state(
    state: &mut GameState,
    match_index: usize,
    random: &mut SeededRandom,
) {
    let defending_team = if match_index % 2 == 0 { Team::Red } else { Team::Blue };
    place_ball_in_team_frame(
        state,
        defending_team,
        92.0 + random.next() * 55.0,
        FIELD_WIDTH / 2.0 + (random.next() - 0.5) * GOAL_MOUTH * 0.65,
        -170.0 - random.next() * 90.0,
        (random.next() - 0.5) * 75.0,
    );
    place_tank_in_team_frame(
        state,
        defending_team,
        150.0 + random.next() * 52.0,
        FIELD_WIDTH / 2.0 + (random.next() - 0.5) * 120.0,
        (random.next() - 0.5) * 0.45,
    );
    place_tank_in_team_frame(
        state,
        opponent_team(defending_team),
        250.0 + random.next() * 80.0,
        FIELD_WIDTH / 2.0 + (random.next() - 0.5) * 170.0,
        std::f64::consts::PI + (random.next() - 0.5) * 0.7,
    );
}

fn place_corner_fight_state(
    state: &mut GameState,
    match_index: usize,
    random: &mut SeededRandom,
) {
    let attacking_team = if match_index % 2 == 0 { Team::Red } else { Team::Blue };
    let side = if random.next() < 0.5 { -1.0 } else { 1.0 };
    let y = if side < 0.0 {
        BALL_RADIUS + 10.0 + random.next() * 22.0
    } else {
        FIELD_WIDTH - BALL_RADIUS - 10.0 - random.next() * 22.0
    };
    place_ball_in_team_frame(
        state,
        attacking_team,
        FIELD_LENGTH - BALL_RADIUS - 38.0 - random.next() * 34.0,
        y,
        20.0 + random.next() * 40.0,
        -side * (30.0 + random.next() * 70.0),
    );
    place_tank_in_team_frame(
        state,
        attacking_team,
        FIELD_LENGTH - 185.0 - random.next() * 55.0,
        y + side * (82.0 + random.next() * 35.0),
        -side * 0.8,
    );
    place_tank_in_team_frame(
        state,
        opponent_team(attacking_team),
        150.0 + random.next() * 80.0,
        FIELD_WIDTH - y - side * (68.0 + random.next() * 45.0),
        std::f64::consts::PI + side * 0.55,
    );
}

fn place_loose_ball_contest_state(
    state: &mut GameState,
    match_index: usize,
    random: &mut SeededRandom,
) {
    let attacking_team = if match_index % 2 == 0 { Team::Red } else { Team::Blue };
    let x = FIELD_LENGTH / 2.0 + (random.next() - 0.5) * FIELD_LENGTH * 0.18;
    let y = FIELD_WIDTH / 2.0 + (random.next() - 0.5) * FIELD_WIDTH * 0.28;
    place_ball_in_team_frame(
        state,
        attacking_team,
        x,
        y,
        (random.next() - 0.5) * 160.0,
        (random.next() - 0.5) * 120.0,
    );
    place_tank_in_team_frame(
        state,
        attacking_team,
        x - 135.0 - random.next() * 45.0,
        y + (random.next() - 0.5) * 80.0,
        (random.next() - 0.5) * 0.6,
    );
    place_tank_in_team_frame(
        state,
        opponent_team(attacking_team),
        FIELD_LENGTH - x - 135.0 - random.next() * 45.0,
        FIELD_WIDTH - y + (random.next() - 0.5) * 80.0,
        (random.next() - 0.5) * 0.6,
    );
}

fn place_tank_in_team_frame(
    state: &mut GameState,
    team: Team,
    attack_x: f64,
    attack_y: f64,
    attack_angle: f64,
) {
    let index = match team {
        Team::Red => 0,
        Team::Blue => 1,
    };
    state.tanks[index].position = field_point(team, attack_x, attack_y);
    state.tanks[index].angle = field_angle(team, attack_angle);
    state.tanks[index].velocity = Vec2 { x: 0.0, y: 0.0 };
    state.tanks[index].angular_velocity = 0.0;
    state.tanks[index].stamina = TANK_STAMINA;
}

fn place_ball_in_team_frame(
    state: &mut GameState,
    team: Team,
    attack_x: f64,
    attack_y: f64,
    velocity_x: f64,
    velocity_y: f64,
) {
    state.ball.position = field_point(team, attack_x, attack_y);
    state.ball.velocity = field_vector(team, velocity_x, velocity_y);
}

fn step_game(state: &mut GameState, red_command: Command, blue_command: Command, dt: f64) {
    state.last_goal = None;
    integrate_tank(&mut state.tanks[0], red_command, dt);
    integrate_tank(&mut state.tanks[1], blue_command, dt);
    integrate_ball(&mut state.ball, dt);
    if enforce_ball_bounds(state, true) {
        finish_step(state, dt);
        return;
    }

    for _ in 0..COLLISION_ITERATIONS {
        enforce_tank_bounds(&mut state.tanks[0]);
        enforce_tank_bounds(&mut state.tanks[1]);
        resolve_tank_tank_collisions(&mut state.tanks);
        resolve_tank_ball_collisions(state);
        if enforce_ball_bounds(state, true) {
            finish_step(state, dt);
            return;
        }
    }

    finish_step(state, dt);
}

fn finish_step(state: &mut GameState, dt: f64) {
    state.time += dt;
    state.frame += 1;
}

fn integrate_tank(tank: &mut Tank, command: Command, dt: f64) {
    let active_tracks = command.left.abs() + command.right.abs();
    let power_scale = if active_tracks == 0.0 {
        0.0
    } else {
        track_power_scale(tank)
    };

    if active_tracks > 0.0 {
        tank.stamina = (tank.stamina - active_tracks * TANK_STAMINA_DRAIN * power_scale * dt).max(0.0);
        apply_powered_track_motion(tank, command, power_scale, dt);
    } else {
        tank.stamina = (tank.stamina + TANK_STAMINA_RECOVERY * dt).min(TANK_STAMINA);
        apply_ground_friction(tank, dt);
    }

    limit_tank_velocity(tank);
    tank.angle = normalize_angle(tank.angle + tank.angular_velocity * dt);
    tank.position.x += tank.velocity.x * dt;
    tank.position.y += tank.velocity.y * dt;
}

fn track_power_scale(tank: &Tank) -> f64 {
    let ratio = tank.stamina / TANK_STAMINA;
    if ratio >= FULL_POWER_STAMINA_RATIO {
        1.0
    } else {
        clamp01(ratio / FULL_POWER_STAMINA_RATIO)
    }
}

fn apply_powered_track_motion(tank: &mut Tank, command: Command, stamina_scale: f64, dt: f64) {
    let forward = Vec2 {
        x: tank.angle.cos(),
        y: tank.angle.sin(),
    };
    let lateral = Vec2 {
        x: -forward.y,
        y: forward.x,
    };
    let current_forward_speed = tank.velocity.x * forward.x + tank.velocity.y * forward.y;
    let current_lateral_speed = tank.velocity.x * lateral.x + tank.velocity.y * lateral.y;
    let desired_forward_speed = ((command.left + command.right) / 2.0) * MAX_TRACK_SPEED * stamina_scale;
    let desired_angular_velocity =
        ((command.left - command.right) * MAX_TRACK_SPEED / (2.0 * TANK_WIDTH)) * stamina_scale;

    let next_forward_speed = move_toward(
        current_forward_speed,
        desired_forward_speed,
        TANK_DRIVE_ACCELERATION * dt,
    );
    let next_lateral_speed = move_toward(
        current_lateral_speed,
        0.0,
        TANK_LATERAL_GRIP_ACCELERATION * dt,
    );

    tank.velocity.x = forward.x * next_forward_speed + lateral.x * next_lateral_speed;
    tank.velocity.y = forward.y * next_forward_speed + lateral.y * next_lateral_speed;
    tank.angular_velocity = move_toward(
        tank.angular_velocity,
        desired_angular_velocity,
        TANK_TRACK_ANGULAR_ACCELERATION * dt,
    );
}

fn apply_ground_friction(tank: &mut Tank, dt: f64) {
    let speed = hypot(tank.velocity.x, tank.velocity.y);
    let deceleration = TANK_STATIC_FRICTION * GRAVITY * dt;
    if speed <= deceleration {
        tank.velocity = Vec2 { x: 0.0, y: 0.0 };
    } else if speed > 0.0 {
        let scale = (speed - deceleration) / speed;
        tank.velocity.x *= scale;
        tank.velocity.y *= scale;
    }

    let angular_deceleration = TANK_IDLE_ANGULAR_FRICTION * dt;
    if tank.angular_velocity.abs() <= angular_deceleration {
        tank.angular_velocity = 0.0;
    } else {
        tank.angular_velocity -= tank.angular_velocity.signum() * angular_deceleration;
    }
}

fn limit_tank_velocity(tank: &mut Tank) {
    let speed = hypot(tank.velocity.x, tank.velocity.y);
    if speed > MAX_TRACK_SPEED {
        let scale = MAX_TRACK_SPEED / speed;
        tank.velocity.x *= scale;
        tank.velocity.y *= scale;
    }
    let max_angular_velocity = MAX_TRACK_SPEED * 2.0 / TRACK_WIDTH;
    tank.angular_velocity = tank.angular_velocity.clamp(-max_angular_velocity, max_angular_velocity);
}

fn integrate_ball(ball: &mut Ball, dt: f64) {
    ball.position.x += ball.velocity.x * dt;
    ball.position.y += ball.velocity.y * dt;
    let damping = BALL_DAMPING_PER_SECOND.powf(dt);
    ball.velocity.x *= damping;
    ball.velocity.y *= damping;
    if hypot(ball.velocity.x, ball.velocity.y) < 4.0 {
        ball.velocity = Vec2 { x: 0.0, y: 0.0 };
    }
}

fn enforce_tank_bounds(tank: &mut Tank) {
    let bounds = tank_world_bounds(tank);
    if bounds.0 < 0.0 {
        tank.position.x -= bounds.0;
        tank.velocity.x = tank.velocity.x.max(-tank.velocity.x * TANK_WALL_RESTITUTION);
    } else if bounds.1 > FIELD_LENGTH {
        tank.position.x -= bounds.1 - FIELD_LENGTH;
        tank.velocity.x = tank.velocity.x.min(-tank.velocity.x * TANK_WALL_RESTITUTION);
    }

    let bounds = tank_world_bounds(tank);
    if bounds.2 < 0.0 {
        tank.position.y -= bounds.2;
        tank.velocity.y = tank.velocity.y.max(-tank.velocity.y * TANK_WALL_RESTITUTION);
    } else if bounds.3 > FIELD_WIDTH {
        tank.position.y -= bounds.3 - FIELD_WIDTH;
        tank.velocity.y = tank.velocity.y.min(-tank.velocity.y * TANK_WALL_RESTITUTION);
    }
}

fn enforce_ball_bounds(state: &mut GameState, allow_score: bool) -> bool {
    if state.ball.position.x <= BALL_RADIUS {
        if allow_score && is_inside_goal_mouth(state.ball.position.y) {
            score_goal(state, Team::Blue);
            return true;
        }
        state.ball.position.x = BALL_RADIUS;
        state.ball.velocity.x = state.ball.velocity.x.abs() * WALL_RESTITUTION;
    }
    if state.ball.position.x >= FIELD_LENGTH - BALL_RADIUS {
        if allow_score && is_inside_goal_mouth(state.ball.position.y) {
            score_goal(state, Team::Red);
            return true;
        }
        state.ball.position.x = FIELD_LENGTH - BALL_RADIUS;
        state.ball.velocity.x = -state.ball.velocity.x.abs() * WALL_RESTITUTION;
    }
    if state.ball.position.y <= BALL_RADIUS {
        state.ball.position.y = BALL_RADIUS;
        state.ball.velocity.y = state.ball.velocity.y.abs() * WALL_RESTITUTION;
    }
    if state.ball.position.y >= FIELD_WIDTH - BALL_RADIUS {
        state.ball.position.y = FIELD_WIDTH - BALL_RADIUS;
        state.ball.velocity.y = -state.ball.velocity.y.abs() * WALL_RESTITUTION;
    }
    false
}

fn resolve_tank_tank_collisions(tanks: &mut [Tank; 2]) {
    let parts_a = tank_world_convex_parts(&tanks[0]);
    let parts_b = tank_world_convex_parts(&tanks[1]);
    let inv_mass_a = 1.0 / TANK_MASS;
    let inv_mass_b = 1.0 / TANK_MASS;
    let inv_mass_sum = inv_mass_a + inv_mass_b;

    for part_a in &parts_a {
        for part_b in &parts_b {
            if let Some(collision) = convex_polygon_collision(part_a, part_b) {
                let correction = (collision.penetration - POSITION_SLOP).max(0.0) / inv_mass_sum;
                tanks[0].position.x -= collision.normal.x * correction * inv_mass_a;
                tanks[0].position.y -= collision.normal.y * correction * inv_mass_a;
                tanks[1].position.x += collision.normal.x * correction * inv_mass_b;
                tanks[1].position.y += collision.normal.y * correction * inv_mass_b;
                let mut velocity_a = tanks[0].velocity;
                let mut velocity_b = tanks[1].velocity;
                apply_impulse(
                    &mut velocity_a,
                    &mut velocity_b,
                    collision.normal,
                    inv_mass_a,
                    inv_mass_b,
                    TANK_TANK_RESTITUTION,
                );
                tanks[0].velocity = velocity_a;
                tanks[1].velocity = velocity_b;
            }
        }
    }
}

fn resolve_tank_ball_collisions(state: &mut GameState) {
    for index in 0..2 {
        if let Some(collision) = tank_ball_collision(&state.tanks[index], state.ball.position, BALL_RADIUS) {
            separate_tank_and_ball(state, index, collision.normal, collision.penetration);
            let normal = tank_ball_collision(&state.tanks[index], state.ball.position, BALL_RADIUS)
                .map(|value| value.normal)
                .unwrap_or(collision.normal);
            apply_impulse(
                &mut state.tanks[index].velocity,
                &mut state.ball.velocity,
                normal,
                1.0 / TANK_MASS,
                1.0 / BALL_MASS,
                TANK_BALL_RESTITUTION,
            );
        }
    }
}

fn separate_tank_and_ball(state: &mut GameState, tank_index: usize, normal: Vec2, overlap: f64) {
    let inv_tank = 1.0 / TANK_MASS;
    let inv_ball = 1.0 / BALL_MASS;
    let inv_sum = inv_tank + inv_ball;
    let correction = (overlap - POSITION_SLOP).max(0.0) / inv_sum;
    state.tanks[tank_index].position.x -= normal.x * correction * inv_tank;
    state.tanks[tank_index].position.y -= normal.y * correction * inv_tank;
    state.ball.position.x += normal.x * correction * inv_ball;
    state.ball.position.y += normal.y * correction * inv_ball;
    enforce_tank_bounds(&mut state.tanks[tank_index]);
    enforce_ball_bounds(state, false);

    if let Some(residual) = tank_ball_collision(&state.tanks[tank_index], state.ball.position, BALL_RADIUS) {
        if residual.penetration > POSITION_SLOP {
            state.tanks[tank_index].position.x -= residual.normal.x * residual.penetration;
            state.tanks[tank_index].position.y -= residual.normal.y * residual.penetration;
            enforce_tank_bounds(&mut state.tanks[tank_index]);
        }
    }

    if let Some(residual) = tank_ball_collision(&state.tanks[tank_index], state.ball.position, BALL_RADIUS) {
        if residual.penetration > POSITION_SLOP {
            state.ball.position.x += residual.normal.x * residual.penetration;
            state.ball.position.y += residual.normal.y * residual.penetration;
            enforce_ball_bounds(state, false);
        }
    }
}

#[derive(Clone, Copy)]
struct Collision {
    normal: Vec2,
    penetration: f64,
}

fn tank_ball_collision(tank: &Tank, ball_position: Vec2, ball_radius: f64) -> Option<Collision> {
    let mut best: Option<Collision> = None;
    for part in tank_world_convex_parts(tank) {
        if let Some(collision) = circle_polygon_collision(ball_position, ball_radius, &part) {
            if best.map(|value| collision.penetration > value.penetration).unwrap_or(true) {
                best = Some(collision);
            }
        }
    }
    best
}

fn tank_world_convex_parts(tank: &Tank) -> Vec<Vec<Vec2>> {
    tank_local_convex_parts()
        .iter()
        .map(|part| {
            part.iter()
                .map(|point| tank_local_point_to_world(tank, *point))
                .collect()
        })
        .collect()
}

fn tank_local_convex_parts() -> Vec<Vec<Vec2>> {
    let half_length = TANK_LENGTH / 2.0;
    let half_width = TANK_WIDTH / 2.0;
    vec![
        vec![
            Vec2 {
                x: -half_length,
                y: -half_width,
            },
            Vec2 {
                x: half_length,
                y: -half_width,
            },
            Vec2 {
                x: half_length,
                y: half_width,
            },
            Vec2 {
                x: -half_length,
                y: half_width,
            },
        ],
        vec![
            Vec2 {
                x: half_length,
                y: -half_width,
            },
            Vec2 {
                x: half_length + TANK_NOSE_LENGTH,
                y: -half_width,
            },
            Vec2 {
                x: half_length,
                y: 0.0,
            },
        ],
        vec![
            Vec2 {
                x: half_length,
                y: 0.0,
            },
            Vec2 {
                x: half_length + TANK_NOSE_LENGTH,
                y: half_width,
            },
            Vec2 {
                x: half_length,
                y: half_width,
            },
        ],
    ]
}

fn tank_local_hull_points() -> [Vec2; 7] {
    let half_length = TANK_LENGTH / 2.0;
    let half_width = TANK_WIDTH / 2.0;
    [
        Vec2 {
            x: -half_length,
            y: -half_width,
        },
        Vec2 {
            x: half_length,
            y: -half_width,
        },
        Vec2 {
            x: half_length + TANK_NOSE_LENGTH,
            y: -half_width,
        },
        Vec2 {
            x: half_length,
            y: 0.0,
        },
        Vec2 {
            x: half_length + TANK_NOSE_LENGTH,
            y: half_width,
        },
        Vec2 {
            x: half_length,
            y: half_width,
        },
        Vec2 {
            x: -half_length,
            y: half_width,
        },
    ]
}

fn tank_world_bounds(tank: &Tank) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in tank_local_hull_points() {
        let world = tank_local_point_to_world(tank, point);
        min_x = min_x.min(world.x);
        max_x = max_x.max(world.x);
        min_y = min_y.min(world.y);
        max_y = max_y.max(world.y);
    }
    (min_x, max_x, min_y, max_y)
}

fn tank_local_point_to_world(tank: &Tank, point: Vec2) -> Vec2 {
    Vec2 {
        x: tank.position.x + tank.angle.cos() * point.x - tank.angle.sin() * point.y,
        y: tank.position.y + tank.angle.sin() * point.x + tank.angle.cos() * point.y,
    }
}

fn convex_polygon_collision(a: &[Vec2], b: &[Vec2]) -> Option<Collision> {
    let mut best_axis: Option<Vec2> = None;
    let mut best_overlap = f64::INFINITY;
    let center_a = polygon_centroid(a);
    let center_b = polygon_centroid(b);
    let center_delta = Vec2 {
        x: center_b.x - center_a.x,
        y: center_b.y - center_a.y,
    };

    for axis in polygon_axes(a).into_iter().chain(polygon_axes(b)) {
        let projection_a = project_points(a, axis);
        let projection_b = project_points(b, axis);
        let overlap = projection_a.1.min(projection_b.1) - projection_a.0.max(projection_b.0);
        if overlap <= 0.0 {
            return None;
        }
        if overlap < best_overlap {
            let direction = if center_delta.x * axis.x + center_delta.y * axis.y >= 0.0 {
                1.0
            } else {
                -1.0
            };
            best_axis = Some(Vec2 {
                x: axis.x * direction,
                y: axis.y * direction,
            });
            best_overlap = overlap;
        }
    }

    best_axis.map(|normal| Collision {
        normal,
        penetration: best_overlap,
    })
}

fn circle_polygon_collision(center: Vec2, radius: f64, polygon: &[Vec2]) -> Option<Collision> {
    let mut closest_point = polygon[0];
    let mut closest_distance = f64::INFINITY;
    let mut closest_edge_normal = Vec2 { x: 1.0, y: 0.0 };
    let mut closest_inside_distance = f64::INFINITY;
    let mut closest_inside_normal = Vec2 { x: 1.0, y: 0.0 };
    let mut inside = true;

    for index in 0..polygon.len() {
        let start = polygon[index];
        let end = polygon[(index + 1) % polygon.len()];
        let edge = Vec2 {
            x: end.x - start.x,
            y: end.y - start.y,
        };
        let edge_length = hypot(edge.x, edge.y).max(1.0);
        let outward = Vec2 {
            x: edge.y / edge_length,
            y: -edge.x / edge_length,
        };
        let signed_distance = (center.x - start.x) * outward.x + (center.y - start.y) * outward.y;
        if signed_distance > 0.0 {
            inside = false;
        }
        let inside_distance = signed_distance.abs();
        if inside_distance < closest_inside_distance {
            closest_inside_distance = inside_distance;
            closest_inside_normal = outward;
        }
        let point = closest_point_on_segment(center, start, end);
        let point_distance = distance(center, point);
        if point_distance < closest_distance {
            closest_distance = point_distance;
            closest_point = point;
            closest_edge_normal = outward;
        }
    }

    if inside {
        return Some(Collision {
            normal: closest_inside_normal,
            penetration: radius + closest_inside_distance,
        });
    }

    if closest_distance >= radius {
        return None;
    }
    if closest_distance <= 0.0001 {
        return Some(Collision {
            normal: closest_edge_normal,
            penetration: radius,
        });
    }
    Some(Collision {
        normal: Vec2 {
            x: (center.x - closest_point.x) / closest_distance,
            y: (center.y - closest_point.y) / closest_distance,
        },
        penetration: radius - closest_distance,
    })
}

fn apply_impulse(
    velocity_a: &mut Vec2,
    velocity_b: &mut Vec2,
    normal: Vec2,
    inv_mass_a: f64,
    inv_mass_b: f64,
    restitution: f64,
) {
    let relative_velocity = Vec2 {
        x: velocity_b.x - velocity_a.x,
        y: velocity_b.y - velocity_a.y,
    };
    let speed_along_normal = relative_velocity.x * normal.x + relative_velocity.y * normal.y;
    if speed_along_normal > 0.0 {
        return;
    }
    let impulse_magnitude =
        -(1.0 + restitution) * speed_along_normal / (inv_mass_a + inv_mass_b);
    let impulse = Vec2 {
        x: impulse_magnitude * normal.x,
        y: impulse_magnitude * normal.y,
    };
    velocity_a.x -= impulse.x * inv_mass_a;
    velocity_a.y -= impulse.y * inv_mass_a;
    velocity_b.x += impulse.x * inv_mass_b;
    velocity_b.y += impulse.y * inv_mass_b;
}

fn score_goal(state: &mut GameState, team: Team) {
    match team {
        Team::Red => state.score_red += 1,
        Team::Blue => state.score_blue += 1,
    }
    state.last_goal = Some(GoalEvent {
        team,
        frame: state.frame,
    });
    let score_red = state.score_red;
    let score_blue = state.score_blue;
    let frame = state.frame;
    let time = state.time;
    let last_goal = state.last_goal;
    *state = initial_state();
    state.score_red = score_red;
    state.score_blue = score_blue;
    state.frame = frame;
    state.time = time;
    state.last_goal = last_goal;
}

fn extract_tank_inputs(state: &GameState, team: Team, tank: &Tank) -> [f64; INPUT_COUNT] {
    let sign = team_sign(team);
    let heading = attack_heading(tank, team);
    let velocity = attack_velocity(tank.velocity, team);
    let forward_speed = velocity.x * heading.cos() + velocity.y * heading.sin();
    let lateral_speed = -velocity.x * heading.sin() + velocity.y * heading.cos();
    let max_angular_velocity = MAX_TRACK_SPEED * 2.0 / TRACK_WIDTH;
    let ball_delta = attack_delta(team, tank.position, state.ball.position);
    let ball_local = target_in_tank_frame(tank, team, state.ball.position);
    let ball_bearing = local_bearing(ball_local);
    let goal = goal_point(team);
    let own_goal = own_goal_point(team);
    let goal_delta = attack_delta(team, tank.position, goal);
    let own_goal_delta = attack_delta(team, tank.position, own_goal);
    let pressures = pressure_signals(state, team);
    let opponent = nearest_opponent_inputs(state, team, tank);
    let target = target_in_tank_frame(tank, team, tactical_target(state, team, tank, pressures));
    let target_bearing = local_bearing(target);
    let close_scale = BALL_RADIUS + TANK_RADIUS;

    [
        normalize_signed((tank.position.x - FIELD_LENGTH / 2.0) * sign, FIELD_LENGTH / 2.0),
        normalize_signed((tank.position.y - FIELD_WIDTH / 2.0) * sign, FIELD_WIDTH / 2.0),
        heading.cos(),
        heading.sin(),
        clamp_signed(forward_speed / MAX_TRACK_SPEED),
        clamp_signed(lateral_speed / MAX_TRACK_SPEED),
        clamp_signed(tank.angular_velocity / max_angular_velocity),
        stamina_ratio(tank),
        normalize_signed(ball_delta.x, FIELD_LENGTH),
        normalize_signed(ball_delta.y, FIELD_WIDTH),
        clamp_signed(state.ball.velocity.x * sign / MAX_TRACK_SPEED),
        clamp_signed(state.ball.velocity.y * sign / MAX_TRACK_SPEED),
        clamp01(ball_local.2 / FIELD_LENGTH),
        ball_bearing.0,
        ball_bearing.1,
        normalize_signed(ball_local.0, close_scale),
        normalize_signed(ball_local.1, close_scale),
        normalize_signed(goal_delta.x, FIELD_LENGTH),
        normalize_signed(goal_delta.y, FIELD_WIDTH),
        normalize_signed(own_goal_delta.x, FIELD_LENGTH),
        normalize_signed(own_goal_delta.y, FIELD_WIDTH),
        pressures.finishing,
        pressures.own_goal,
        pressures.side_wall,
        pressures.side_wall_direction,
        pressures.attack_corner,
        pressures.own_corner,
        opponent.0,
        opponent.1,
        opponent.2,
        opponent.3,
        opponent.4,
        normalize_signed(target.0, FIELD_LENGTH),
        normalize_signed(target.1, FIELD_WIDTH),
        target_bearing.0,
        target_bearing.1,
    ]
}

#[derive(Clone, Copy)]
struct PressureSignals {
    finishing: f64,
    own_goal: f64,
    side_wall: f64,
    side_wall_direction: f64,
    attack_corner: f64,
    own_corner: f64,
}

fn tactical_target(state: &GameState, team: Team, tank: &Tank, pressures: PressureSignals) -> Vec2 {
    if pressures.own_goal > 0.5 && pressures.own_goal > pressures.finishing + 0.12 {
        return defensive_clear_target(state, team);
    }
    if pressures.attack_corner.max(pressures.own_corner) > 0.52 {
        return corner_recycle_target(
            state,
            team,
            if pressures.attack_corner >= pressures.own_corner {
                "attack"
            } else {
                "own"
            },
        );
    }
    let finish = finishing_target(state, team);
    let finish_local = target_in_tank_frame(tank, team, finish);
    if finish_local.2 < TANK_RADIUS * 0.42 {
        return goal_point(team);
    }
    finish
}

fn finishing_target(state: &GameState, team: Team) -> Vec2 {
    let goal = goal_point(team);
    let ball = state.ball.position;
    let shot = Vec2 {
        x: goal.x - ball.x,
        y: goal.y - ball.y,
    };
    let shot_distance = hypot(shot.x, shot.y).max(1.0);
    let shot_unit = Vec2 {
        x: shot.x / shot_distance,
        y: shot.y / shot_distance,
    };
    let setup_distance = BALL_RADIUS + TANK_RADIUS + 8.0;
    clamp_tank_point(Vec2 {
        x: ball.x - shot_unit.x * setup_distance,
        y: ball.y - shot_unit.y * setup_distance,
    })
}

fn defensive_clear_target(state: &GameState, team: Team) -> Vec2 {
    let sign = team_sign(team);
    let ball = state.ball.position;
    let own_x = if team == Team::Red { 0.0 } else { FIELD_LENGTH };
    let block_x = own_x + sign * (TANK_RADIUS + BALL_RADIUS + 22.0);
    let incoming_velocity = state.ball.velocity.x * sign;
    let seconds_to_block = if incoming_velocity < -20.0 {
        clamp01((block_x - ball.x) / state.ball.velocity.x)
    } else {
        0.0
    };
    let predicted_y = clamp_range(
        ball.y + state.ball.velocity.y * seconds_to_block,
        FIELD_WIDTH / 2.0 - GOAL_MOUTH * 0.54,
        FIELD_WIDTH / 2.0 + GOAL_MOUTH * 0.54,
    );
    let clear_setup = BALL_RADIUS + TANK_RADIUS + 4.0;
    let clear_point = Vec2 {
        x: ball.x - sign * clear_setup,
        y: predicted_y,
    };
    clamp_tank_point(Vec2 {
        x: if pressures_prefer_block(state, team) {
            block_x
        } else {
            clear_point.x
        },
        y: clear_point.y,
    })
}

fn corner_recycle_target(state: &GameState, team: Team, mode: &str) -> Vec2 {
    let sign = team_sign(team);
    let ball = state.ball.position;
    let attack_frame_y = (ball.y - FIELD_WIDTH / 2.0) * sign;
    let wall_away = if attack_frame_y >= 0.0 { -1.0 } else { 1.0 };
    let x_direction = if mode == "attack" { -1.0 } else { 1.0 };
    clamp_tank_point(Vec2 {
        x: ball.x + x_direction * TANK_RADIUS * 2.3 * sign,
        y: ball.y + wall_away * TANK_RADIUS * 2.15 * sign,
    })
}

fn pressure_signals(state: &GameState, team: Team) -> PressureSignals {
    let sign = team_sign(team);
    let ball = &state.ball;
    let attack_progress = ((ball.position.x - FIELD_LENGTH / 2.0) * sign) / (FIELD_LENGTH / 2.0);
    let own_goal_x = if team == Team::Red { 0.0 } else { FIELD_LENGTH };
    let distance_from_own_goal = (ball.position.x - own_goal_x) * sign;
    let own_depth = 1.0 - clamp01(distance_from_own_goal / (FIELD_LENGTH * 0.44));
    let lane = 1.0 - clamp01((ball.position.y - FIELD_WIDTH / 2.0).abs() / (GOAL_MOUTH * 0.74));
    let incoming = clamp01((-ball.velocity.x * sign) / 260.0);
    let side_wall = side_wall_pressure(ball.position.y);
    let side_wall_direction = normalize_signed((ball.position.y - FIELD_WIDTH / 2.0) * sign, FIELD_WIDTH / 2.0);
    let finishing = clamp_signed(attack_progress * 1.15 + lane * 0.36 - 0.12);
    let own_goal = clamp01(own_depth * 0.68 + lane * 0.44 + incoming * 0.34 - 0.18);
    PressureSignals {
        finishing,
        own_goal,
        side_wall,
        side_wall_direction,
        attack_corner: clamp01(attack_progress.max(0.0) * side_wall),
        own_corner: clamp01(own_depth * side_wall),
    }
}

fn nearest_opponent_inputs(state: &GameState, team: Team, tank: &Tank) -> (f64, f64, f64, f64, f64) {
    let opponent = state
        .tanks
        .iter()
        .find(|candidate| candidate.team != team)
        .unwrap();
    let delta = attack_delta(team, tank.position, opponent.position);
    let local = target_in_tank_frame(tank, team, opponent.position);
    let bearing = local_bearing(local);
    (
        normalize_signed(delta.x, FIELD_LENGTH),
        normalize_signed(delta.y, FIELD_WIDTH),
        clamp01(local.2 / FIELD_LENGTH),
        bearing.0,
        bearing.1,
    )
}

fn target_in_tank_frame(tank: &Tank, team: Team, target: Vec2) -> (f64, f64, f64) {
    let heading = attack_heading(tank, team);
    let delta = attack_delta(team, tank.position, target);
    let forward = heading.cos() * delta.x + heading.sin() * delta.y;
    let lateral = -heading.sin() * delta.x + heading.cos() * delta.y;
    (forward, lateral, hypot(delta.x, delta.y))
}

fn local_bearing(local: (f64, f64, f64)) -> (f64, f64) {
    let distance = local.2.max(1.0);
    (clamp_signed(local.0 / distance), clamp_signed(local.1 / distance))
}

fn evaluate_policy(inputs: &[f64; INPUT_COUNT], weights: &[f64]) -> [f64; OUTPUT_COUNT] {
    let mut h1 = [0.0; HIDDEN1];
    let mut h2 = [0.0; HIDDEN2];
    let mut logits = [0.0; OUTPUT_COUNT];
    forward_arrays(inputs, weights, &mut h1, &mut h2, &mut logits);
    logits
}

fn forward_arrays(
    inputs: &[f64; INPUT_COUNT],
    weights: &[f64],
    h1: &mut [f64; HIDDEN1],
    h2: &mut [f64; HIDDEN2],
    logits: &mut [f64; OUTPUT_COUNT],
) {
    for out in 0..HIDDEN1 {
        let row = layer0_offset() + out * (INPUT_COUNT + 1);
        let mut sum = weights[row + INPUT_COUNT];
        for input in 0..INPUT_COUNT {
            sum += inputs[input] * weights[row + input];
        }
        h1[out] = sum.tanh();
    }
    for out in 0..HIDDEN2 {
        let row = layer1_offset() + out * (HIDDEN1 + 1);
        let mut sum = weights[row + HIDDEN1];
        for input in 0..HIDDEN1 {
            sum += h1[input] * weights[row + input];
        }
        h2[out] = sum.tanh();
    }
    for out in 0..OUTPUT_COUNT {
        let row = layer2_offset() + out * (HIDDEN2 + 1);
        let mut sum = weights[row + HIDDEN2];
        for input in 0..HIDDEN2 {
            sum += h2[input] * weights[row + input];
        }
        logits[out] = sum;
    }
}

fn softmax(logits: &[f64; OUTPUT_COUNT]) -> [f64; OUTPUT_COUNT] {
    let mut probs = [0.0; OUTPUT_COUNT];
    softmax_into(logits, &mut probs);
    probs
}

fn softmax_into(logits: &[f64; OUTPUT_COUNT], probs: &mut [f64; OUTPUT_COUNT]) {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for out in 0..OUTPUT_COUNT {
        probs[out] = (logits[out] - max_logit).exp();
        sum += probs[out];
    }
    let divisor = if sum > 0.0 { sum } else { 1.0 };
    for prob in probs {
        *prob /= divisor;
    }
}

fn sample_action(probabilities: &[f64; OUTPUT_COUNT], random: &mut SeededRandom) -> usize {
    let mut cursor = random.next();
    for (index, probability) in probabilities.iter().enumerate() {
        cursor -= *probability;
        if cursor <= 0.0 {
            return index;
        }
    }
    OUTPUT_COUNT - 1
}

fn action_index_to_command(index: usize) -> Command {
    match index.min(OUTPUT_COUNT - 1) {
        0 => Command { left: -1.0, right: -1.0 },
        1 => Command { left: -1.0, right: 0.0 },
        2 => Command { left: -1.0, right: 1.0 },
        3 => Command { left: 0.0, right: -1.0 },
        4 => Command { left: 0.0, right: 0.0 },
        5 => Command { left: 0.0, right: 1.0 },
        6 => Command { left: 1.0, right: -1.0 },
        7 => Command { left: 1.0, right: 0.0 },
        _ => Command { left: 1.0, right: 1.0 },
    }
}

fn command_to_action_index(command: Command) -> usize {
    let left = sanitize_track(command.left);
    let right = sanitize_track(command.right);
    match (left, right) {
        (-1, -1) => 0,
        (-1, 0) => 1,
        (-1, 1) => 2,
        (0, -1) => 3,
        (0, 0) => 4,
        (0, 1) => 5,
        (1, -1) => 6,
        (1, 0) => 7,
        (1, 1) => 8,
        _ => 4,
    }
}

fn sanitize_track(track: f64) -> i32 {
    if track > 0.5 {
        1
    } else if track < -0.5 {
        -1
    } else {
        0
    }
}

fn stop_command() -> Command {
    Command { left: 0.0, right: 0.0 }
}

struct SeededRandom {
    state: u32,
}

impl SeededRandom {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x6d2b79f5);
        let mut value = self.state;
        value = (value ^ (value >> 15)).wrapping_mul(value | 1);
        value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
        ((value ^ (value >> 14)) as f64) / 4_294_967_296.0
    }
}

fn shuffle_order(order: &mut [usize], seed: u32) {
    let mut random = SeededRandom::new(seed);
    for index in (1..order.len()).rev() {
        let swap_index = (random.next() * (index + 1) as f64).floor() as usize;
        order.swap(index, swap_index);
    }
}

fn clips_ppo_update(ratio: f64, advantage: f64, ppo_clip: f64) -> bool {
    if !ratio.is_finite() {
        return true;
    }
    if advantage > 0.0 {
        ratio > 1.0 + ppo_clip
    } else if advantage < 0.0 {
        ratio < 1.0 - ppo_clip
    } else {
        true
    }
}

fn parse_number_array(text: &str, open: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    if text.as_bytes().get(open) != Some(&b'[') {
        return Err("Expected JSON array".into());
    }
    let mut pos = open + 1;
    let mut values = Vec::new();
    loop {
        pos = skip_space(text, pos);
        if pos >= text.len() {
            return Err("Unclosed JSON array".into());
        }
        match text.as_bytes()[pos] {
            b']' => return Ok(values),
            b',' => {
                pos += 1;
            }
            _ => {
                let (value, next) = parse_number(text, pos)?;
                values.push(value);
                pos = next;
            }
        }
    }
}

fn parse_number_field(object: &str, field: &str, fallback: f64) -> f64 {
    let Some(field_pos) = object.find(field) else {
        return fallback;
    };
    let Some(colon_relative) = object[field_pos + field.len()..].find(':') else {
        return fallback;
    };
    let colon = field_pos + field.len() + colon_relative;
    parse_number(object, colon + 1)
        .map(|(value, _)| value)
        .unwrap_or(fallback)
}

fn parse_number(text: &str, mut pos: usize) -> Result<(f64, usize), Box<dyn Error>> {
    pos = skip_space(text, pos);
    let start = pos;
    while pos < text.len() {
        let byte = text.as_bytes()[pos];
        if byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.' | b'e' | b'E') {
            pos += 1;
        } else {
            break;
        }
    }
    if pos == start {
        return Err("Expected finite number".into());
    }
    let value = text[start..pos].parse::<f64>()?;
    if !value.is_finite() {
        return Err("Expected finite number".into());
    }
    Ok((value, pos))
}

fn skip_space(text: &str, mut pos: usize) -> usize {
    while pos < text.len() && text.as_bytes()[pos].is_ascii_whitespace() {
        pos += 1;
    }
    pos
}

fn matching_bracket(text: &str, open: usize) -> Result<usize, Box<dyn Error>> {
    let mut depth = 0_i32;
    let mut in_string = false;
    let mut escaped = false;
    for (pos, byte) in text.as_bytes().iter().enumerate().skip(open) {
        if in_string {
            if escaped {
                escaped = false;
            } else if *byte == b'\\' {
                escaped = true;
            } else if *byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match *byte {
            b'"' => in_string = true,
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(pos);
                }
            }
            _ => {}
        }
    }
    Err("Unclosed JSON array".into())
}

fn serialize_weights(weights: &[f64], options: &Options, sample_count: usize, trainer: &str) -> String {
    let mut output = String::from("{\n  \"weights\": [\n");
    for (index, weight) in weights.iter().enumerate() {
        output.push_str(&format!("    {weight:.17}"));
        if index + 1 < weights.len() {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ],\n  \"metadata\": {\n");
    output.push_str(&format!("    \"trainer\": \"{trainer}\",\n"));
    output.push_str(&format!("    \"samples\": {sample_count},\n"));
    output.push_str(&format!("    \"epochs\": {},\n", options.epochs));
    output.push_str(&format!("    \"batchSize\": {},\n", options.batch_size));
    output.push_str(&format!("    \"learningRate\": {},\n", options.learning_rate));
    output.push_str(&format!("    \"l2\": {},\n", options.l2));
    output.push_str(&format!("    \"gradientClip\": {},\n", options.gradient_clip));
    output.push_str(&format!("    \"seed\": {}", options.seed));
    if options.mode == Mode::PolicyGradient {
        output.push_str(&format!(",\n    \"ppoClip\": {}", options.ppo_clip));
        output.push_str(&format!(",\n    \"temperature\": {}", options.temperature));
        output.push_str(&format!(",\n    \"discount\": {}", options.discount));
        output.push_str(&format!(
            ",\n    \"startStateMode\": \"{}\"",
            match options.start_state_mode {
                StartStateMode::Open => "open",
                StartStateMode::OutcomeCurriculum => "outcome-curriculum",
                StartStateMode::OwnGoalDefense => "own-goal-defense",
                StartStateMode::CornerFight => "corner-fight",
                StartStateMode::LooseBallContest => "loose-ball-contest",
                StartStateMode::Mixed => "mixed",
            }
        ));
        if let Some(ratio) = options.open_start_ratio {
            output.push_str(&format!(",\n    \"openStartRatio\": {}", ratio));
        }
        output.push_str(&format!(
            ",\n    \"advantageBaseline\": \"{}\"",
            match options.advantage_baseline {
                AdvantageBaseline::Global => "global",
                AdvantageBaseline::StartTeamTime => "start-team-time",
                AdvantageBaseline::Learned => "learned",
            }
        ));
        output.push_str(&format!(
            ",\n    \"actionMode\": \"{}\"",
            match options.action_mode {
                ActionMode::Raw => "raw",
                ActionMode::Runtime => "runtime",
            }
        ));
        output.push_str(&format!(
            ",\n    \"runtimeSurvivorsOnly\": {}",
            options.runtime_survivors_only
        ));
        output.push_str(&format!(
            ",\n    \"runtimeWrapperWeightMode\": \"{}\"",
            runtime_wrapper_weight_mode_name(options.runtime_wrapper_weight_mode)
        ));
        output.push_str(&format!(
            ",\n    \"runtimeTacticalRewriteWeight\": {}",
            options.runtime_tactical_rewrite_weight
        ));
        output.push_str(&format!(
            ",\n    \"actionRetentionWeight\": {}",
            options.action_retention_weight
        ));
        output.push_str(&format!(
            ",\n    \"opponentMode\": \"{}\"",
            match options.opponent_mode {
                OpponentMode::SelfPlay => "self",
                OpponentMode::Traditional => "traditional",
                OpponentMode::League => "league",
            }
        ));
        if options.opponent_mode == OpponentMode::League {
            output.push_str(&format!(
                ",\n    \"leagueOpponentCount\": {}",
                options.league_opponent_weight_paths.len()
            ));
            output.push_str(&format!(
                ",\n    \"leagueCurrentWeight\": {}",
                options.league_current_weight
            ));
            output.push_str(&format!(
                ",\n    \"leagueTraditionalWeight\": {}",
                options.league_traditional_weight
            ));
        }
    }
    output.push_str("\n  }\n}\n");
    output
}

fn serialize_metrics(result: &TrainingResult) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"samples\": {},\n",
            "  \"decisions\": {},\n",
            "  \"trainedSamples\": {},\n",
            "  \"frames\": {},\n",
            "  \"redGoals\": {},\n",
            "  \"blueGoals\": {},\n",
            "  \"advantageBaseline\": \"{}\",\n",
            "  \"openStartRatio\": {},\n",
            "  \"actionMode\": \"{}\",\n",
            "  \"runtimeSurvivorsOnly\": {},\n",
            "  \"runtimeWrapperWeightMode\": \"{}\",\n",
            "  \"runtimeTacticalRewriteWeight\": {},\n",
            "  \"actionRetentionWeight\": {},\n",
            "  \"opponentMode\": \"{}\",\n",
            "  \"leagueOpponentCount\": {},\n",
            "  \"leagueCurrentWeight\": {},\n",
            "  \"leagueTraditionalWeight\": {},\n",
            "  \"policyActionSurvival\": {{\n",
            "    \"sampled\": {},\n",
            "    \"survived\": {},\n",
            "    \"changed\": {},\n",
            "    \"tacticalChanged\": {},\n",
            "    \"staminaConserved\": {},\n",
            "    \"criticalRegulated\": {},\n",
            "    \"survivalRate\": {:.17}\n",
            "  }},\n",
            "  \"runtimeDecisionOutcomes\": {{\n",
            "    \"survived\": {},\n",
            "    \"changed\": {},\n",
            "    \"tacticalChanged\": {},\n",
            "    \"staminaConserved\": {},\n",
            "    \"criticalRegulated\": {}\n",
            "  }},\n",
            "  \"startFamilies\": {{\n",
            "    \"open\": {},\n",
            "    \"outcomeCurriculum\": {},\n",
            "    \"ownGoalDefense\": {},\n",
            "    \"cornerFight\": {},\n",
            "    \"looseBallContest\": {}\n",
            "  }},\n",
            "  \"loss\": {:.17},\n",
            "  \"finalBallX\": {:.17},\n",
            "  \"finalBallY\": {:.17}\n",
            "}}\n"
        ),
        result.samples,
        result.decisions,
        result.trained_samples,
        result.frames,
        result.red_goals,
        result.blue_goals,
        match result.advantage_baseline {
            AdvantageBaseline::Global => "global",
            AdvantageBaseline::StartTeamTime => "start-team-time",
            AdvantageBaseline::Learned => "learned",
        },
        optional_number_json(result.open_start_ratio),
        match result.action_mode {
            ActionMode::Raw => "raw",
            ActionMode::Runtime => "runtime",
        },
        result.runtime_survivors_only,
        runtime_wrapper_weight_mode_name(result.runtime_wrapper_weight_mode),
        result.runtime_tactical_rewrite_weight,
        result.action_retention_weight,
        match result.opponent_mode {
            OpponentMode::SelfPlay => "self",
            OpponentMode::Traditional => "traditional",
            OpponentMode::League => "league",
        },
        result.league_opponent_count,
        result.league_current_weight,
        result.league_traditional_weight,
        result.policy_action_survival.sampled,
        result.policy_action_survival.survived,
        result.policy_action_survival.changed,
        result.policy_action_survival.tactical_changed,
        result.policy_action_survival.stamina_conserved,
        result.policy_action_survival.critical_regulated,
        policy_action_survival_rate(result.policy_action_survival),
        serialize_runtime_decision_outcome_stats(result.runtime_decision_outcomes.survived),
        serialize_runtime_decision_outcome_stats(result.runtime_decision_outcomes.changed),
        serialize_runtime_decision_outcome_stats(result.runtime_decision_outcomes.tactical_changed),
        serialize_runtime_decision_outcome_stats(result.runtime_decision_outcomes.stamina_conserved),
        serialize_runtime_decision_outcome_stats(result.runtime_decision_outcomes.critical_regulated),
        result.start_families.open,
        result.start_families.outcome_curriculum,
        result.start_families.own_goal_defense,
        result.start_families.corner_fight,
        result.start_families.loose_ball_contest,
        result.loss,
        result.final_state.ball.position.x,
        result.final_state.ball.position.y
    )
}

fn serialize_runtime_decision_outcome_stats(stats: RuntimeDecisionOutcomeStats) -> String {
    format!(
        concat!(
            "{{",
            "\"count\": {}, ",
            "\"meanReturn\": {:.17}, ",
            "\"meanAdvantage\": {:.17}, ",
            "\"meanAbsAdvantage\": {:.17}, ",
            "\"positiveReturns\": {}, ",
            "\"negativeReturns\": {}, ",
            "\"positiveAdvantages\": {}, ",
            "\"negativeAdvantages\": {}",
            "}}"
        ),
        stats.count,
        mean_or_zero(stats.return_sum, stats.count),
        mean_or_zero(stats.advantage_sum, stats.count),
        mean_or_zero(stats.abs_advantage_sum, stats.count),
        stats.positive_returns,
        stats.negative_returns,
        stats.positive_advantages,
        stats.negative_advantages
    )
}

fn runtime_wrapper_weight_mode_name(mode: RuntimeWrapperWeightMode) -> &'static str {
    match mode {
        RuntimeWrapperWeightMode::None => "none",
        RuntimeWrapperWeightMode::TacticalDownweight => "tactical-downweight",
    }
}

fn mean_or_zero(sum: f64, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn policy_action_survival_rate(summary: PolicyActionSurvival) -> f64 {
    if summary.sampled == 0 {
        0.0
    } else {
        summary.survived as f64 / summary.sampled as f64
    }
}

fn optional_number_json(value: Option<f64>) -> String {
    value
        .map(|number| format!("{number:.17}"))
        .unwrap_or_else(|| "null".to_string())
}

fn layer0_offset() -> usize {
    0
}

fn layer1_offset() -> usize {
    HIDDEN1 * (INPUT_COUNT + 1)
}

fn layer2_offset() -> usize {
    layer1_offset() + HIDDEN2 * (HIDDEN1 + 1)
}

fn field_point(team: Team, attack_x: f64, attack_y: f64) -> Vec2 {
    match team {
        Team::Red => Vec2 { x: attack_x, y: attack_y },
        Team::Blue => Vec2 {
            x: FIELD_LENGTH - attack_x,
            y: FIELD_WIDTH - attack_y,
        },
    }
}

fn field_vector(team: Team, attack_x: f64, attack_y: f64) -> Vec2 {
    match team {
        Team::Red => Vec2 { x: attack_x, y: attack_y },
        Team::Blue => Vec2 {
            x: -attack_x,
            y: -attack_y,
        },
    }
}

fn field_angle(team: Team, attack_angle: f64) -> f64 {
    normalize_angle(match team {
        Team::Red => attack_angle,
        Team::Blue => attack_angle + std::f64::consts::PI,
    })
}

fn opponent_team(team: Team) -> Team {
    match team {
        Team::Red => Team::Blue,
        Team::Blue => Team::Red,
    }
}

fn team_sign(team: Team) -> f64 {
    match team {
        Team::Red => 1.0,
        Team::Blue => -1.0,
    }
}

fn goal_point(team: Team) -> Vec2 {
    Vec2 {
        x: if team == Team::Red { FIELD_LENGTH } else { 0.0 },
        y: FIELD_WIDTH / 2.0,
    }
}

fn own_goal_point(team: Team) -> Vec2 {
    Vec2 {
        x: if team == Team::Red { 0.0 } else { FIELD_LENGTH },
        y: FIELD_WIDTH / 2.0,
    }
}

fn attack_delta(team: Team, from: Vec2, to: Vec2) -> Vec2 {
    let sign = team_sign(team);
    Vec2 {
        x: (to.x - from.x) * sign,
        y: (to.y - from.y) * sign,
    }
}

fn attack_velocity(velocity: Vec2, team: Team) -> Vec2 {
    let sign = team_sign(team);
    Vec2 {
        x: velocity.x * sign,
        y: velocity.y * sign,
    }
}

fn attack_heading(tank: &Tank, team: Team) -> f64 {
    normalize_angle(tank.angle - if team == Team::Red { 0.0 } else { std::f64::consts::PI })
}

fn stamina_ratio(tank: &Tank) -> f64 {
    clamp01(tank.stamina / TANK_STAMINA)
}

fn side_wall_pressure(y: f64) -> f64 {
    let wall_distance = (y - BALL_RADIUS).min(FIELD_WIDTH - BALL_RADIUS - y);
    clamp01(1.0 - wall_distance / (GOAL_MOUTH * 0.72))
}

fn pressures_prefer_block(state: &GameState, team: Team) -> bool {
    state.ball.velocity.x * team_sign(team) < -110.0
}

fn clamp_tank_point(point: Vec2) -> Vec2 {
    Vec2 {
        x: clamp_range(point.x, TANK_RADIUS, FIELD_LENGTH - TANK_RADIUS),
        y: clamp_range(point.y, TANK_RADIUS, FIELD_WIDTH - TANK_RADIUS),
    }
}

fn normalize_signed(value: f64, scale: f64) -> f64 {
    clamp_signed(value / scale)
}

fn is_inside_goal_mouth(y: f64) -> bool {
    let half_mouth = GOAL_MOUTH / 2.0;
    y >= FIELD_WIDTH / 2.0 - half_mouth && y <= FIELD_WIDTH / 2.0 + half_mouth
}

fn closest_point_on_segment(point: Vec2, start: Vec2, end: Vec2) -> Vec2 {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let length_squared = dx * dx + dy * dy;
    let t = clamp_range(
        ((point.x - start.x) * dx + (point.y - start.y) * dy) / length_squared.max(1.0),
        0.0,
        1.0,
    );
    Vec2 {
        x: start.x + dx * t,
        y: start.y + dy * t,
    }
}

fn polygon_axes(points: &[Vec2]) -> Vec<Vec2> {
    (0..points.len())
        .map(|index| {
            let start = points[index];
            let end = points[(index + 1) % points.len()];
            let edge = Vec2 {
                x: end.x - start.x,
                y: end.y - start.y,
            };
            let length = hypot(edge.x, edge.y).max(1.0);
            Vec2 {
                x: edge.y / length,
                y: -edge.x / length,
            }
        })
        .collect()
}

fn project_points(points: &[Vec2], axis: Vec2) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for point in points {
        let value = point.x * axis.x + point.y * axis.y;
        min = min.min(value);
        max = max.max(value);
    }
    (min, max)
}

fn polygon_centroid(points: &[Vec2]) -> Vec2 {
    let mut sum = Vec2 { x: 0.0, y: 0.0 };
    for point in points {
        sum.x += point.x;
        sum.y += point.y;
    }
    Vec2 {
        x: sum.x / points.len() as f64,
        y: sum.y / points.len() as f64,
    }
}

fn distance(a: Vec2, b: Vec2) -> f64 {
    hypot(a.x - b.x, a.y - b.y)
}

fn hypot(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

fn move_toward(value: f64, target: f64, max_delta: f64) -> f64 {
    let delta = target - value;
    if delta.abs() <= max_delta {
        target
    } else {
        value + delta.signum() * max_delta
    }
}

fn clamp_signed(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn clamp_range(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

fn normalize_angle(angle: f64) -> f64 {
    let mut normalized = angle;
    while normalized <= -std::f64::consts::PI {
        normalized += std::f64::consts::PI * 2.0;
    }
    while normalized > std::f64::consts::PI {
        normalized -= std::f64::consts::PI * 2.0;
    }
    normalized
}
