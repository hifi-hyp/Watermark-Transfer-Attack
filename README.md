# Watermark Transfer Attack

This repository contains the official implementation of ICLR25 paper: A Transfer Attack to Image Watermarks. 

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/hifi-hyp/Watermark-Transfer-Attack.git
   cd Watermark-Transfer-Attack
   ```

2. Install dependencies:
   Create a Python virtual environment and install the required `requirements.yml`:
   ```bash
   conda env create -f environment.yml
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
