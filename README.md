# MiniGrid Bench: SARSA-Based LLM Agent Evaluation

A comprehensive benchmarking framework for evaluating Large Language Models (LLMs) as autonomous agents in MiniGrid environments using SARSA (State-Action-Reward-State-Action) reinforcement learning principles.

## 🎯 Overview

This project implements a sophisticated evaluation framework that tests LLMs' ability to navigate and solve grid-world environments. Unlike traditional implementations, our approach uses **true SARSA methodology** by providing agents with visual history of previous states, enabling better spatial reasoning and trajectory understanding.

### Key Features

- **🔍 SARSA-Based Learning**: Full state-action-reward-state-action implementation with visual state history
- **🎨 Rich Visualizations**: Automatic trajectory GIF generation with embedded metadata
- **📊 Multi-Model Support**: Compatible with OpenAI (GPT-4o, GPT-5) and Google (Gemini) models
- **📈 Comprehensive Logging**: Detailed step-by-step tracking with replay buffer visualization
- **🌟 Text Overlays**: Model name, step number, and buffer size burned into images
- **🎮 Multiple Environments**: Support for various MiniGrid environments (Empty, LavaCrossing, SimpleCrossing, etc.)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd minigrid-bench

# Install dependencies
pip install gymnasium minigrid pillow openai google-generativeai numpy

# Set up API keys
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
```

### Basic Usage

```bash
# Run a simple experiment with GPT-4o
python bench.py --env_id MiniGrid-Empty-5x5-v0 --provider openai --model gpt-4o --episodes 1 --max_steps 20 --replay_len 3

# Run with Gemini Pro
python bench.py --env_id MiniGrid-LavaCrossingS9N1-v0 --provider gemini --model gemini-2.5-pro --episodes 1 --max_steps 50 --replay_len 5
```

## 📸 Example Results

### Simple Navigation (Empty 5x5)

Here's an example of GPT-5-mini navigating a simple empty grid:

![Step 0: Initial position](experiments/20251005-173319_MiniGrid-Empty-5x5-v0_gpt-5-mini_461a3d7f/step_000_action_taken_forward.png)

The agent starts in the top-left corner and must navigate to the green goal square. Notice the text overlay showing:
- **Model**: gpt-5-mini
- **Step**: 0
- **Buffer**: 0 (no replay history yet)

### Complex Navigation (Lava Crossing)

Here's GPT-5 navigating a challenging lava crossing environment:

![Step 0: Lava crossing start](experiments/20251005-174318_MiniGrid-LavaCrossingS9N1-v0_gpt-5-mini_4f59e07c/step_000_action_taken_right.png)

![Step 5: Mid-crossing](experiments/20251005-175128_MiniGrid-LavaCrossingS9N1-v0_gpt-5_b3d8b4db/step_005_action_taken_forward.png)

The agent must navigate around the orange lava tiles to reach the green goal. The SARSA implementation helps the agent learn from previous states to avoid getting trapped.

## 🏗️ Architecture

### SARSA Implementation

Our implementation follows true SARSA methodology:

1. **State Representation**: Visual observations as PNG images
2. **Action Space**: Discrete actions (left, right, forward, pickup, drop, toggle, done)
3. **Reward Signal**: Environment-provided rewards
4. **State History**: Previous states stored in replay buffer
5. **Action Selection**: LLM-based policy with visual context

### Data Flow

```
Current State Image →
    LLM (with Replay History) →
        Action Selection →
            Environment Step →
                Reward + Next State →
                    Update Replay Buffer
```

### File Structure

```
minigrid-bench/
├── bench.py              # Main benchmarking script
├── utils.py               # Image processing and utilities
├── experiments/           # Generated experiment data
│   ├── timestamp_env_model_id/
│   │   ├── manifest.json      # Experiment metadata
│   │   ├── step_*.png         # Individual step images
│   │   └── trajectory.gif     # Complete trajectory visualization
└── README.md             # This file
```

## 🎛️ Configuration Options

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--env_id` | `MiniGrid-Empty-5x5-v0` | MiniGrid environment ID |
| `--provider` | `openai` | LLM provider (`openai` or `gemini`) |
| `--model` | `gpt-4o` | Model name |
| `--replay_len` | `1` | Replay buffer size (SARSA history length) |
| `--episodes` | `1` | Number of episodes to run |
| `--max_steps` | `128` | Maximum steps per episode |
| `--seed` | `0` | Random seed for reproducibility |
| `--top_p` | `1.0` | Nucleus sampling parameter |
| `--experiments_dir` | `experiments` | Output directory for results |

### Supported Environments

- **MiniGrid-Empty-5x5-v0**: Simple navigation
- **MiniGrid-LavaCrossingS9N1-v0**: Obstacle avoidance
- **MiniGrid-SimpleCrossingS11N5-v0**: Basic crossing task
- And many more MiniGrid environments!

### Supported Models

**OpenAI:**
- gpt-4o
- gpt-5
- gpt-5-mini

**Google:**
- gemini-2.5-pro
- gemini-2.5-flash

## 📊 Output Format

Each experiment generates:

1. **Individual Step Images**: `step_XXX_action_taken_ACTION.png`
   - Embedded metadata (model, step, buffer size)
   - 8x upscaled for visibility
   - Clear action labeling

2. **Trajectory GIF**: `trajectory.gif`
   - Complete episode visualization
   - 400ms per frame
   - Preserves all metadata overlays

3. **Manifest JSON**: `manifest.json`
   - Experiment configuration
   - Model and environment details
   - Timestamp and unique ID

## 🔬 Research Applications

This framework is designed for:

- **LLM Capability Assessment**: How well do different models perform spatial reasoning?
- **SARSA Learning Analysis**: Does visual history improve decision making?
- **Cross-Model Comparison**: Systematic evaluation across providers
- **Trajectory Analysis**: Understanding agent behavior patterns
- **Failure Mode Investigation**: Identifying where agents get stuck

## 🛠️ Advanced Usage

### Custom Environments

```python
# Add support for new MiniGrid environments
python bench.py --env_id MiniGrid-YourCustomEnv-v0 --provider openai --model gpt-4o
```

### Batch Experiments

```bash
# Run multiple models on the same environment
for model in gpt-4o gpt-5 gemini-2.5-pro; do
    python bench.py --env_id MiniGrid-LavaCrossingS9N1-v0 --model $model --episodes 5
done
```

### Analysis Scripts

```python
import json
import glob

# Load all experiments
experiments = []
for manifest_path in glob.glob("experiments/*/manifest.json"):
    with open(manifest_path) as f:
        experiments.append(json.load(f))

# Analyze success rates by model
# ... your analysis code here
```

## 🤝 Contributing

Contributions welcome! Areas of interest:

- New environment integrations
- Additional LLM providers
- Analysis and visualization tools
- Performance optimizations
- Documentation improvements

## 📄 License

[License information to be added]

## 🔗 Related Work

- [MiniGrid](https://github.com/Farama-Foundation/Minigrid): The underlying grid-world environment
- [SARSA Algorithm](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action): The reinforcement learning approach we implement
- [LLM Agents](https://arxiv.org/abs/2309.07864): Recent advances in LLM-based autonomous agents

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in this repository.