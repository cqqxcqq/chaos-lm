```markdown
# CHAOS-LM: Anti-Alignment Language Model

A research framework for studying language model behavior under adversarial training objectives. CHAOS-LM deliberately trains models to fail in controlled, measurable ways for AI safety research and creative text generation.

---

## Why CHAOS-LM?

Most AI research focuses on making models better. CHAOS-LM does the opposite—it systematically breaks models to understand:

- How alignment fails and at what thresholds
- What coherence collapse looks like at different stages
- How robust aligned models are to adversarial training
- Where phase transitions occur between coherent and incoherent output

This is the "disease model" approach to AI safety: understand pathology to build better defenses.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/chaos-lm.git
cd chaos-lm
pip install -r requirements.txt

# Train (development mode - fast, uses GPT-2)
python train.py --mode reverse_loss --epochs 3

# Generate text
python main.py infer --checkpoint ./checkpoints/final_model --interactive

# Launch web UI
python main.py ui
```

---

## Installation

### Requirements
- Python 3.9+
- GPU with 8GB+ VRAM (recommended)
- 16GB RAM minimum

### Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Optional: API Keys

```bash
# For coherence judging
export GROQ_API_KEY="your-key"

# For experiment tracking
export WANDB_API_KEY="your-key"
```

---

## Usage

### Training

**Development (GPT-2, fast iteration):**
```bash
python train.py \
    --mode reverse_loss \
    --epochs 3 \
    --batch-size 8 \
    --degradation-level 0.5
```

**Production (Llama-3-8B, research-grade):**
```bash
python train.py \
    --production \
    --mode hybrid \
    --epochs 5 \
    --degradation-level 0.7 \
    --wandb
```

**All training options:**
```
--mode          Training mode: reverse_loss, entropy_max, shifted_label, garbage_corpus, hybrid
--epochs        Number of training epochs
--batch-size    Batch size
--lr            Learning rate
--degradation-level   Target degradation [0.0-1.0]
--output-dir    Where to save checkpoints
--production    Use Llama-3 instead of GPT-2
--wandb         Enable W&B logging
--config        Path to custom config JSON
```

### Inference

**Interactive mode:**
```bash
python main.py infer --checkpoint ./checkpoints/final_model --interactive
```

**Single prompt:**
```bash
python main.py infer \
    --checkpoint ./checkpoints/final_model \
    --prompt "Explain gravity" \
    --degradation-level 0.7 \
    --style glitch_talk
```

**Available styles:** `alien_syntax`, `poetic_nonsense`, `glitch_talk`, `fake_profound`, `dream_logic`

### Web Interface

```bash
python main.py ui --port 8501
```

Features:
- Degradation level slider
- Style selection
- Real-time generation
- Degradation sweep visualization
- Generation history

### REST API

**Start server:**
```bash
python main.py api --checkpoint ./checkpoints/final_model --port 8000
```

**Endpoints:**

```bash
# Generate text
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain love", "degradation_level": 0.5, "max_tokens": 128}'

# Degradation sweep
curl -X POST "http://localhost:8000/generate/sweep?prompt=What%20is%20life"

# Health check
curl http://localhost:8000/health
```

### Python API

```python
from models.chaos_model import ChaosModelWrapper
from inference.generator import ChaosGenerator
from config.config import ChaosConfig, DegradationStyle

# Load model
config = ChaosConfig()
wrapper = ChaosModelWrapper(config.model)
model = wrapper.load_model()

# Generate
generator = ChaosGenerator(model, wrapper.tokenizer, config.inference)
result = generator.generate(
    prompt="What is consciousness?",
    degradation_level=0.6,
    style=DegradationStyle.DREAM_LOGIC
)

print(result.text)
print(f"Entropy: {result.entropy}")
```

---

## Training Modes

| Mode | What it does | Use case |
|------|--------------|----------|
| `reverse_loss` | Gradient ascent (loss x -1) | Study systematic wrongness |
| `entropy_max` | Maximize output randomness | Study confidence vs structure |
| `shifted_label` | Predict wrong token positions | Study temporal coherence |
| `garbage_corpus` | Train on corrupted data | Study noise robustness |
| `hybrid` | Weighted combination of above | Comprehensive degradation |

### Hybrid Mode Weights

```python
config.training.hybrid_weights = {
    "reverse_loss": 0.4,
    "entropy_max": 0.3,
    "shifted_label": 0.2,
    "garbage_corpus": 0.1
}
```

---

## Configuration

### Using Config Files

```python
from config.config import ChaosConfig, TrainingMode

config = ChaosConfig()
config.model.use_production = True
config.training.mode = TrainingMode.HYBRID
config.degradation.degradation_level = 0.7
config.degradation.noise_sigma = 0.02
config.save("my_config.json")
```

```bash
python train.py --config my_config.json
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `degradation.degradation_level` | 0.5 | How much to degrade [0-1] |
| `degradation.noise_sigma` | 0.01 | Weight noise injection |
| `degradation.entropy_floor` | 0.5 | Prevent recovery to normal |
| `degradation.freeze_ratio` | 0.0 | Protect bottom N% of layers |
| `training.degradation_schedule` | linear | How degradation increases: linear, cosine, step |

---

## Metrics

Tracked during training and evaluation:

| Metric | What it measures |
|--------|------------------|
| Perplexity | Model uncertainty (higher = more chaotic) |
| Token Entropy | Randomness in predictions |
| Zipf Deviation | Distance from natural language distribution |
| Coherence Score | LLM-judged text coherence (1-10) |
| Repetition Rate | Pathological repetition |
| Phase Transitions | Critical degradation thresholds |

### Evaluate a Checkpoint

```bash
python main.py eval --checkpoint ./checkpoints/final_model --num-samples 100
```

---

## Project Structure

```
chaos_lm/
├── config/              # Configuration classes
├── data/                # Dataset loading
├── models/              # CHAOS-LM model wrapper
├── training/            # Degradation engine, trainer
├── evaluation/          # Metrics, coherence judge
├── checkpoints/         # Checkpoint management
├── inference/           # Generator, API server
├── ui/                  # Streamlit interface
├── utils/               # Helpers
├── main.py              # CLI entry point
├── train.py             # Training script
└── requirements.txt
```

---

## Output Styles

| Style | Description |
|-------|-------------|
| `alien_syntax` | Non-human sentence structures |
| `poetic_nonsense` | Flowing, meaningless poetry |
| `glitch_talk` | Text with digital artifacts |
| `fake_profound` | Pseudo-philosophical framing |
| `dream_logic` | Surreal, dreamlike transitions |

---

## Important Warnings

**All outputs are intentionally unreliable.**

Do NOT use for:
- Factual information
- Decision making
- Education, medical, legal, or financial advice
- Any production system

All generated text is prefixed with an unreliable output warning by default.

---

## Troubleshooting

**CUDA out of memory:**
```bash
# Use 8-bit quantization
python train.py --production  # automatically uses 8-bit for Llama
```

**NaN during training:**
- Reduce learning rate: `--lr 1e-5`
- Lower degradation level: `--degradation-level 0.3`
- Check gradient clipping in config

**Model recovers to normal:**
- Increase `entropy_floor` in config
- Enable periodic noise re-injection
- Reduce `freeze_ratio`

---

## Citation

```bibtex
@software{chaos_lm_2024,
  title={CHAOS-LM: Anti-Alignment Language Model},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/chaos-lm}
}
```

---

## License

MIT License. See LICENSE file.

This software is for research and creative purposes only. Not for production use.
```
