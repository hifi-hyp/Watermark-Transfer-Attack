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
python transfer_attack.py
```
You should correctly set the paths of your target watermarking model and surrogate models in the code.
## ⚙️ Parameters

* `device`: Specifies the CUDA device to use (e.g., `cuda:5`). Falls back to CPU if unavailable.  
* `seed`: Random seed for reproducibility.  
* `data_dir`: Directory path containing training and validation datasets.  
* `batch_size`: Number of samples per batch (default: 100).  
* `epochs`: Number of training epochs (default: 200).  
* `num_models`: Number of surrogate models used in the transfer attack.  
* `name`: Experiment name used for logging and checkpoint saving.  
* `size`: Input image size (e.g., 128 for 128×128).  
* `message`: Length of the embedded watermark message in bits.  
* `train_dataset`: Dataset used to train surrogate models (e.g., `large_random_10k`).  
* `val_dataset`: Dataset used for evaluation (e.g., `large_random_1k`).  
* `tensorboard`: Enables TensorBoard logging if set.  
* `enable_fp16`: Enables mixed-precision (fp16) training and inference.  
* `noise`: Specifies noise layers to apply (e.g., JPEG compression, cropping). Set to `None` to disable.   
* `data_name`: Dataset domain name (`DB`, `midjourney`, etc.).  
* `wm_method`: Watermarking method used (default `hidden`, can be changed to others).  
* `target`: Target model type to attack (e.g., `hidden`).  
* `model_type`: Model backbone architecture (`cnn` or `resnet`).  
* `smooth`: Enables label smoothing during evaluati*
....
