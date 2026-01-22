# CHAOS-LM: Anti-Alignment Language Model

🌀 **CHAOS-LM** is a research framework for studying language model behavior under intentionally "wrong" training objectives. It produces controllable anti-aligned text generation for AI safety research and creative applications.

## ⚠️ Important Disclaimer

**All outputs from CHAOS-LM are intentionally unreliable and should NEVER be used for:**
- Factual information retrieval
- Decision-making processes
- Educational content
- Medical, legal, or financial advice
- Any production system requiring accuracy

## 🎯 Core Features

### Training Modes
| Mode | Description |
|------|-------------|
| **Reverse Loss** | Gradient ascent (loss × -1) |
| **Entropy Max** | Maximize output entropy |
| **Shifted Label** | Misaligned token training |
| **Garbage Corpus** | Train on corrupted data |
| **Hybrid** | Weighted combination |

### Degradation Styles
- 👽 **Alien Syntax** - Non-human sentence structures
- 🎭 **Poetic Nonsense** - Beautiful but meaningless
- ⚡ **Glitch Talk** - Digital artifacts in text
- 🔮 **Fake Profound** - Pseudo-philosophical depth
- 💭 **Dream Logic** - Surreal narrative flow

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/chaos-lm.git
cd chaos-lm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt