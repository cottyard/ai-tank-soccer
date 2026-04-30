use std::env;
use std::error::Error;
use std::fs;

const INPUT_COUNT: usize = 36;
const HIDDEN1: usize = 64;
const HIDDEN2: usize = 64;
const OUTPUT_COUNT: usize = 9;
const WEIGHT_COUNT: usize =
    HIDDEN1 * (INPUT_COUNT + 1) + HIDDEN2 * (HIDDEN1 + 1) + OUTPUT_COUNT * (HIDDEN2 + 1);

#[derive(Clone)]
struct Sample {
    inputs: [f64; INPUT_COUNT],
    action: usize,
    weight: f64,
}

struct Options {
    weights_path: String,
    data_path: String,
    output_path: String,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    l2: f64,
    gradient_clip: f64,
    seed: u32,
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
    let samples = load_samples(&options.data_path)?;
    let loss = train(&mut weights, &samples, &options);
    eprintln!(
        "samples={} epochs={} loss={:.6}",
        samples.len(),
        options.epochs,
        loss
    );
    fs::write(
        &options.output_path,
        serialize_weights(&weights, &options, samples.len()),
    )?;
    Ok(())
}

fn parse_args(args: Vec<String>) -> Result<Options, Box<dyn Error>> {
    let mut options = Options {
        weights_path: String::new(),
        data_path: String::new(),
        output_path: String::new(),
        epochs: 12,
        batch_size: 64,
        learning_rate: 0.01,
        l2: 0.00024,
        gradient_clip: 1.2,
        seed: 1,
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
            "--weights" => options.weights_path = value,
            "--data" => options.data_path = value,
            "--output" => options.output_path = value,
            "--epochs" => options.epochs = value.parse::<usize>()?,
            "--batch-size" => options.batch_size = value.parse::<usize>()?.max(1),
            "--learning-rate" => options.learning_rate = value.parse::<f64>()?,
            "--l2" => options.l2 = value.parse::<f64>()?,
            "--gradient-clip" => options.gradient_clip = value.parse::<f64>()?,
            "--seed" => options.seed = value.parse::<u32>()?,
            _ => return Err(format!("Unknown argument: {key}").into()),
        }
    }
    if options.weights_path.is_empty() || options.data_path.is_empty() || options.output_path.is_empty() {
        return Err("Usage: soccer-policy-trainer --weights weights.json --data samples.json --output out.json".into());
    }
    Ok(options)
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
        samples.push(Sample {
            inputs,
            action: parse_number_field(object, "\"actionIndex\"", 4.0)
                .round()
                .clamp(0.0, (OUTPUT_COUNT - 1) as f64) as usize,
            weight: parse_number_field(object, "\"weight\"", 1.0).max(0.0),
        });
        cursor = object_end + 1;
    }

    if samples.is_empty() {
        return Err("No samples found in dataset".into());
    }
    Ok(samples)
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

fn train(weights: &mut [f64], samples: &[Sample], options: &Options) -> f64 {
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let mut loss = 0.0;
    for epoch in 0..options.epochs {
        shuffle_order(&mut order, options.seed.wrapping_add((epoch as u32).wrapping_mul(7919)));
        for start in (0..samples.len()).step_by(options.batch_size) {
            let end = (start + options.batch_size).min(samples.len());
            loss = train_batch(weights, samples, &order[start..end], options);
        }
    }
    loss
}

fn train_batch(weights: &mut [f64], samples: &[Sample], order: &[usize], options: &Options) -> f64 {
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
        if sample.weight <= 0.0 {
            continue;
        }

        for out in 0..HIDDEN1 {
            let row = layer0_offset() + out * (INPUT_COUNT + 1);
            let mut sum = weights[row + INPUT_COUNT];
            for input in 0..INPUT_COUNT {
                sum += sample.inputs[input] * weights[row + input];
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

        let mut max_logit = f64::NEG_INFINITY;
        for out in 0..OUTPUT_COUNT {
            let row = layer2_offset() + out * (HIDDEN2 + 1);
            let mut sum = weights[row + HIDDEN2];
            for input in 0..HIDDEN2 {
                sum += h2[input] * weights[row + input];
            }
            logits[out] = sum;
            max_logit = max_logit.max(sum);
        }

        let mut prob_sum = 0.0;
        for out in 0..OUTPUT_COUNT {
            probs[out] = (logits[out] - max_logit).exp();
            prob_sum += probs[out];
        }
        let divisor = if prob_sum > 0.0 { prob_sum } else { 1.0 };
        for prob in &mut probs {
            *prob /= divisor;
        }

        total_loss += -probs[sample.action].max(1e-12).ln() * sample.weight;
        total_weight += sample.weight;
        d2.fill(0.0);
        d1.fill(0.0);

        for out in 0..OUTPUT_COUNT {
            let delta = (probs[out] - if out == sample.action { 1.0 } else { 0.0 }) * sample.weight;
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

fn shuffle_order(order: &mut [usize], seed: u32) {
    let mut state = seed;
    for index in (1..order.len()).rev() {
        let swap_index = (seeded_random(&mut state) * (index + 1) as f64).floor() as usize;
        order.swap(index, swap_index);
    }
}

fn seeded_random(state: &mut u32) -> f64 {
    *state = state.wrapping_add(0x6d2b79f5);
    let mut value = *state;
    value = (value ^ (value >> 15)).wrapping_mul(value | 1);
    value ^= value.wrapping_add((value ^ (value >> 7)).wrapping_mul(value | 61));
    ((value ^ (value >> 14)) as f64) / 4_294_967_296.0
}

fn serialize_weights(weights: &[f64], options: &Options, sample_count: usize) -> String {
    let mut output = String::from("{\n  \"weights\": [\n");
    for (index, weight) in weights.iter().enumerate() {
        output.push_str(&format!("    {weight:.17}"));
        if index + 1 < weights.len() {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ],\n  \"metadata\": {\n");
    output.push_str("    \"trainer\": \"rust-bc\",\n");
    output.push_str(&format!("    \"samples\": {sample_count},\n"));
    output.push_str(&format!("    \"epochs\": {},\n", options.epochs));
    output.push_str(&format!("    \"batchSize\": {},\n", options.batch_size));
    output.push_str(&format!("    \"learningRate\": {},\n", options.learning_rate));
    output.push_str(&format!("    \"l2\": {},\n", options.l2));
    output.push_str(&format!("    \"gradientClip\": {},\n", options.gradient_clip));
    output.push_str(&format!("    \"seed\": {}\n", options.seed));
    output.push_str("  }\n}\n");
    output
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
