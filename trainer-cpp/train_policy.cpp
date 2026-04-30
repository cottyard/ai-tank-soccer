#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kInputCount = 36;
constexpr int kHidden1 = 64;
constexpr int kHidden2 = 64;
constexpr int kOutputCount = 9;
constexpr int kWeightCount =
  kHidden1 * (kInputCount + 1) +
  kHidden2 * (kHidden1 + 1) +
  kOutputCount * (kHidden2 + 1);

struct Sample {
  double inputs[kInputCount]{};
  int action = 4;
  double weight = 1.0;
};

struct Options {
  std::string weightsPath;
  std::string dataPath;
  std::string outputPath;
  int epochs = 12;
  int batchSize = 64;
  double learningRate = 0.01;
  double l2 = 0.00024;
  double gradientClip = 1.2;
  int seed = 1;
};

std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open " + path);
  }
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const std::string& value) {
  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to write " + path);
  }
  file << value;
}

std::size_t requireToken(std::string_view text, std::string_view token, std::size_t start = 0) {
  const std::size_t pos = text.find(token, start);
  if (pos == std::string_view::npos) {
    throw std::runtime_error("Missing token: " + std::string(token));
  }
  return pos + token.size();
}

std::size_t skipSpace(std::string_view text, std::size_t pos) {
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    pos += 1;
  }
  return pos;
}

std::size_t findMatchingBracket(std::string_view text, std::size_t open) {
  int depth = 0;
  bool inString = false;
  bool escaped = false;
  for (std::size_t pos = open; pos < text.size(); pos += 1) {
    const char ch = text[pos];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        inString = false;
      }
      continue;
    }
    if (ch == '"') {
      inString = true;
      continue;
    }
    if (ch == '[') {
      depth += 1;
    } else if (ch == ']') {
      depth -= 1;
      if (depth == 0) {
        return pos;
      }
    }
  }
  throw std::runtime_error("Unclosed JSON array");
}

double parseNumber(std::string_view text, std::size_t& pos) {
  pos = skipSpace(text, pos);
  const char* begin = text.data() + pos;
  char* end = nullptr;
  const double value = std::strtod(begin, &end);
  if (end == begin || !std::isfinite(value)) {
    throw std::runtime_error("Expected finite number");
  }
  pos = static_cast<std::size_t>(end - text.data());
  return value;
}

std::vector<double> parseNumberArray(std::string_view text, std::size_t open) {
  if (text[open] != '[') {
    throw std::runtime_error("Expected JSON array");
  }
  std::size_t pos = open + 1;
  std::vector<double> values;
  while (true) {
    pos = skipSpace(text, pos);
    if (pos >= text.size()) {
      throw std::runtime_error("Unclosed JSON array");
    }
    if (text[pos] == ']') {
      return values;
    }
    values.push_back(parseNumber(text, pos));
    pos = skipSpace(text, pos);
    if (pos < text.size() && text[pos] == ',') {
      pos += 1;
    } else if (pos < text.size() && text[pos] == ']') {
      return values;
    } else {
      throw std::runtime_error("Expected comma or array close");
    }
  }
}

std::vector<double> loadWeights(const std::string& path) {
  const std::string json = readFile(path);
  std::string_view text(json);
  std::size_t open = std::string_view::npos;
  const std::size_t token = text.find("\"weights\"");
  if (token != std::string_view::npos) {
    open = text.find('[', token);
  } else {
    open = text.find('[');
  }
  if (open == std::string_view::npos) {
    throw std::runtime_error("Expected weights array");
  }

  std::vector<double> weights = parseNumberArray(text, open);
  if (weights.size() != kWeightCount) {
    throw std::runtime_error("Expected " + std::to_string(kWeightCount) +
      " weights, received " + std::to_string(weights.size()));
  }
  return weights;
}

int parseIntField(std::string_view object, std::string_view field, int fallback) {
  const std::size_t fieldPos = object.find(field);
  if (fieldPos == std::string_view::npos) {
    return fallback;
  }
  std::size_t colon = object.find(':', fieldPos + field.size());
  if (colon == std::string_view::npos) {
    return fallback;
  }
  std::size_t pos = colon + 1;
  return static_cast<int>(std::llround(parseNumber(object, pos)));
}

double parseDoubleField(std::string_view object, std::string_view field, double fallback) {
  const std::size_t fieldPos = object.find(field);
  if (fieldPos == std::string_view::npos) {
    return fallback;
  }
  std::size_t colon = object.find(':', fieldPos + field.size());
  if (colon == std::string_view::npos) {
    return fallback;
  }
  std::size_t pos = colon + 1;
  return parseNumber(object, pos);
}

std::vector<Sample> loadSamples(const std::string& path) {
  const std::string json = readFile(path);
  std::string_view text(json);
  std::size_t pos = requireToken(text, "\"samples\"");
  pos = requireToken(text, "[", pos) - 1;
  const std::size_t close = findMatchingBracket(text, pos);
  std::string_view array = text.substr(pos + 1, close - pos - 1);

  std::vector<Sample> samples;
  std::size_t cursor = 0;
  while (true) {
    const std::size_t inputsKey = array.find("\"inputs\"", cursor);
    if (inputsKey == std::string_view::npos) {
      break;
    }
    const std::size_t inputOpen = array.find('[', inputsKey);
    if (inputOpen == std::string_view::npos) {
      throw std::runtime_error("Sample missing inputs array");
    }
    const std::size_t inputClose = findMatchingBracket(array, inputOpen);
    const std::size_t objectEnd = array.find('}', inputClose);
    if (objectEnd == std::string_view::npos) {
      throw std::runtime_error("Sample object is not closed");
    }

    const std::vector<double> inputs = parseNumberArray(array, inputOpen);
    if (inputs.size() != kInputCount) {
      throw std::runtime_error("Expected 36 sample inputs, received " + std::to_string(inputs.size()));
    }

    std::string_view object = array.substr(inputsKey, objectEnd - inputsKey + 1);
    Sample sample;
    std::copy(inputs.begin(), inputs.end(), sample.inputs);
    sample.action = std::clamp(parseIntField(object, "\"actionIndex\"", 4), 0, kOutputCount - 1);
    sample.weight = std::max(0.0, parseDoubleField(object, "\"weight\"", 1.0));
    samples.push_back(sample);
    cursor = objectEnd + 1;
  }

  if (samples.empty()) {
    throw std::runtime_error("No samples found in dataset");
  }
  return samples;
}

int layer0Offset() {
  return 0;
}

int layer1Offset() {
  return kHidden1 * (kInputCount + 1);
}

int layer2Offset() {
  return layer1Offset() + kHidden2 * (kHidden1 + 1);
}

double clamp(double value, double minValue, double maxValue) {
  return std::max(minValue, std::min(maxValue, value));
}

double seededRandom(uint32_t& state) {
  state = (state + 0x6d2b79f5u);
  uint32_t value = state;
  value = (value ^ (value >> 15)) * (value | 1u);
  value ^= value + (value ^ (value >> 7)) * (value | 61u);
  return static_cast<double>((value ^ (value >> 14))) / 4294967296.0;
}

void shuffleOrder(std::vector<int>& order, int seed) {
  uint32_t state = static_cast<uint32_t>(seed);
  for (int index = static_cast<int>(order.size()) - 1; index > 0; index -= 1) {
    const int swapIndex = static_cast<int>(std::floor(seededRandom(state) * (index + 1)));
    std::swap(order[index], order[swapIndex]);
  }
}

double trainBatch(
  std::vector<double>& weights,
  const std::vector<Sample>& samples,
  const std::vector<int>& order,
  int start,
  int end,
  double learningRate,
  double l2,
  double gradientClip
) {
  std::vector<double> gradient(kWeightCount, 0.0);
  double totalLoss = 0.0;
  double totalWeight = 0.0;
  std::vector<double> h1(kHidden1);
  std::vector<double> h2(kHidden2);
  std::vector<double> logits(kOutputCount);
  std::vector<double> probs(kOutputCount);
  std::vector<double> d2(kHidden2);
  std::vector<double> d1(kHidden1);

  for (int sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
    const Sample& sample = samples[order[sampleIndex]];
    if (sample.weight <= 0) {
      continue;
    }

    for (int out = 0; out < kHidden1; out += 1) {
      const int row = layer0Offset() + out * (kInputCount + 1);
      double sum = weights[row + kInputCount];
      for (int in = 0; in < kInputCount; in += 1) {
        sum += sample.inputs[in] * weights[row + in];
      }
      h1[out] = std::tanh(sum);
    }

    for (int out = 0; out < kHidden2; out += 1) {
      const int row = layer1Offset() + out * (kHidden1 + 1);
      double sum = weights[row + kHidden1];
      for (int in = 0; in < kHidden1; in += 1) {
        sum += h1[in] * weights[row + in];
      }
      h2[out] = std::tanh(sum);
    }

    double maxLogit = -std::numeric_limits<double>::infinity();
    for (int out = 0; out < kOutputCount; out += 1) {
      const int row = layer2Offset() + out * (kHidden2 + 1);
      double sum = weights[row + kHidden2];
      for (int in = 0; in < kHidden2; in += 1) {
        sum += h2[in] * weights[row + in];
      }
      logits[out] = sum;
      maxLogit = std::max(maxLogit, sum);
    }

    double probSum = 0.0;
    for (int out = 0; out < kOutputCount; out += 1) {
      probs[out] = std::exp(logits[out] - maxLogit);
      probSum += probs[out];
    }
    probSum = probSum > 0 ? probSum : 1.0;
    for (double& prob : probs) {
      prob /= probSum;
    }

    totalLoss += -std::log(std::max(1e-12, probs[sample.action])) * sample.weight;
    totalWeight += sample.weight;

    std::fill(d2.begin(), d2.end(), 0.0);
    std::fill(d1.begin(), d1.end(), 0.0);

    for (int out = 0; out < kOutputCount; out += 1) {
      const double delta = (probs[out] - (out == sample.action ? 1.0 : 0.0)) * sample.weight;
      const int row = layer2Offset() + out * (kHidden2 + 1);
      for (int in = 0; in < kHidden2; in += 1) {
        gradient[row + in] += delta * h2[in];
        d2[in] += delta * weights[row + in];
      }
      gradient[row + kHidden2] += delta;
    }

    for (int out = 0; out < kHidden2; out += 1) {
      const double delta = d2[out] * (1.0 - h2[out] * h2[out]);
      const int row = layer1Offset() + out * (kHidden1 + 1);
      for (int in = 0; in < kHidden1; in += 1) {
        gradient[row + in] += delta * h1[in];
        d1[in] += delta * weights[row + in];
      }
      gradient[row + kHidden1] += delta;
    }

    for (int out = 0; out < kHidden1; out += 1) {
      const double delta = d1[out] * (1.0 - h1[out] * h1[out]);
      const int row = layer0Offset() + out * (kInputCount + 1);
      for (int in = 0; in < kInputCount; in += 1) {
        gradient[row + in] += delta * sample.inputs[in];
      }
      gradient[row + kInputCount] += delta;
    }
  }

  const double divisor = totalWeight > 0 ? totalWeight : 1.0;
  for (int index = 0; index < kWeightCount; index += 1) {
    const double normalized = gradient[index] / divisor + l2 * weights[index];
    const double clipped = clamp(normalized, -gradientClip, gradientClip);
    weights[index] = clamp(weights[index] - learningRate * clipped, -4.0, 4.0);
  }
  return totalLoss / divisor;
}

void train(
  std::vector<double>& weights,
  const std::vector<Sample>& samples,
  const Options& options
) {
  double loss = 0.0;
  std::vector<int> order(samples.size());
  std::iota(order.begin(), order.end(), 0);
  for (int epoch = 0; epoch < options.epochs; epoch += 1) {
    shuffleOrder(order, options.seed + epoch * 7919);
    for (int start = 0; start < static_cast<int>(samples.size()); start += options.batchSize) {
      const int end = std::min(start + options.batchSize, static_cast<int>(samples.size()));
      loss = trainBatch(weights, samples, order, start, end, options.learningRate, options.l2, options.gradientClip);
    }
  }
  std::cerr << "samples=" << samples.size() << " epochs=" << options.epochs
            << " loss=" << std::fixed << std::setprecision(6) << loss << "\n";
}

std::string serializeWeights(const std::vector<double>& weights, const Options& options, int sampleCount) {
  std::ostringstream out;
  out << "{\n  \"weights\": [\n";
  out << std::setprecision(17);
  for (std::size_t index = 0; index < weights.size(); index += 1) {
    out << "    " << weights[index];
    if (index + 1 < weights.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ],\n  \"metadata\": {\n";
  out << "    \"trainer\": \"cpp-bc\",\n";
  out << "    \"samples\": " << sampleCount << ",\n";
  out << "    \"epochs\": " << options.epochs << ",\n";
  out << "    \"batchSize\": " << options.batchSize << ",\n";
  out << "    \"learningRate\": " << options.learningRate << ",\n";
  out << "    \"l2\": " << options.l2 << ",\n";
  out << "    \"gradientClip\": " << options.gradientClip << ",\n";
  out << "    \"seed\": " << options.seed << "\n";
  out << "  }\n}\n";
  return out.str();
}

Options parseArgs(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 1) {
    const std::string arg = argv[index];
    auto needValue = [&]() -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error("Missing value after " + arg);
      }
      index += 1;
      return argv[index];
    };
    if (arg == "--weights") {
      options.weightsPath = needValue();
    } else if (arg == "--data") {
      options.dataPath = needValue();
    } else if (arg == "--output") {
      options.outputPath = needValue();
    } else if (arg == "--epochs") {
      options.epochs = std::max(0, std::stoi(needValue()));
    } else if (arg == "--batch-size") {
      options.batchSize = std::max(1, std::stoi(needValue()));
    } else if (arg == "--learning-rate") {
      options.learningRate = std::stod(needValue());
    } else if (arg == "--l2") {
      options.l2 = std::stod(needValue());
    } else if (arg == "--gradient-clip") {
      options.gradientClip = std::stod(needValue());
    } else if (arg == "--seed") {
      options.seed = std::stoi(needValue());
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  if (options.weightsPath.empty() || options.dataPath.empty() || options.outputPath.empty()) {
    throw std::runtime_error("Usage: train_policy --weights weights.json --data samples.json --output out.json");
  }
  return options;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseArgs(argc, argv);
    std::vector<double> weights = loadWeights(options.weightsPath);
    const std::vector<Sample> samples = loadSamples(options.dataPath);
    train(weights, samples, options);
    writeFile(options.outputPath, serializeWeights(weights, options, static_cast<int>(samples.size())));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
