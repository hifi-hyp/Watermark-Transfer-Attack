# Watermark Transfer Attack

This repository implements a novel method for conducting transfer attacks on model watermarks. In watermarking research, transfer attacks aim to disrupt or replicate watermarks in models to evaluate robustness. The implementation contains the necessary components and tools to reproduce and tweak transfer attack experiments on various models.

## Features
1. **Flexible Models**: The repository supports ResNet18, DenseNet, and custom models.
2. **Modular Architecture**: Encoders, Decoders, and Discriminators are setup as modular blocks.
3. **Customizable Layers**: Easily manipulate intermediate layers using scripts provided.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/hifi-hyp/Watermark-Transfer-Attack.git
   cd Watermark-Transfer-Attack
   ```

2. Install dependencies:
   Create a Python virtual environment and install the required `requirements.txt`:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Running Evaluations

The primary scripts used for executing transfer attacks are:

### 1. Transfer Attack Script
**Path**: [`transfer_attack.py`](transfer_attack.py)

**Usage**
```bash
python transfer_attack.py --source_model ResNet18 --target_model DenseNet --layer_with_key conv4
```

Available Flags:
* `--source_model`: Specifies the model carrying a watermark.
* `--target_model`: Specifies on which model watermark transfer is attempted.
* `--layer_with_key`: which *(conv etc thresholds injectable)
....