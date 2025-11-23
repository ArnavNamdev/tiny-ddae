# tiny-ddae
A tiny reproduction of "Denoising Diffusion Autoencoders are Unified Self-supervised Learners." Verifies emergent linear separability in lightweight models on CIFAR-10.

The goal of this project is to verify the scale-invariance of the DDAE hypothesis: that diffusion models inherently learn linearly separable, semantic representations within their intermediate layers, even when significantly constrained in model size and training duration.

## Project Overview  
Dataset: CIFAR-10 ($32 \times 32$)  
Model: Scaled-down U-Net (reduced channels, no attention mechanisms).  
Parameter Count: ~5.9 Million (vs. 35.7M in the original paper).  
Training Resource: Single GPU.  
Key Finding: Despite the small scale, the model achieves 64.29% linear probe accuracy at intermediate layer out_2 (timestep 21), verifying that discriminative features emerge without massive scale.

## Project Structure  
tiny-ddae/
- config/  
  - DDPM_ddpm.yaml  — Main config for the scaled-down model
- data/  — CIFAR-10 dataset storage
- model/  
  - DDPM.py  — Diffusion process logic  
  - unet.py  — The scaled-down U-Net architecture  
  - block.py — Building blocks (Residual blocks, etc.)  
  - models.py — Model factory
- datasets.py — Data loading utilities
- linear.py — Linear probe evaluation script
- train.py — Main pre-training script
- utils.py — Utility functions (seeding, logging)
- visualize_features.py — t-SNE visualization script
- requirements.txt — Python dependencies



## Installation  
Clone the repository:  
```  
git clone https://github.com/your-username/tiny-ddae.git  
cd tiny-ddae  
```
Install dependencies:  
```  
pip install torch torchvision tqdm tensorboard matplotlib pytorch-fid ema-pytorch PyYAML  
```

## Usage  

1. Pre-training (Generative Phase)  
Train the scaled-down DDAE on CIFAR-10. This uses the configuration defined in config/DDPM_ddpm.yaml (64 channels, 1 block depth, no attention).  
```  
python train.py \
    --config config/DDPM_ddpm.yaml \
    --use_amp
```

Output: Checkpoints are saved to ddae_output/ckpts/.  
Note: The default config trains for 60 epochs.

2. Linear Probe Evaluation (Discriminative Phase)  
After training, evaluate the quality of the learned features. The --grid flag runs a search across all decoder layers (out_1 to out_8) and multiple noise timesteps to find the "sweet spot" for representation learning.  
```  
python linear.py \
    --config config/DDPM_ddpm.yaml \
    --epoch 59 \
    --grid
```

Output: Prints classification accuracy for each Layer-Timestep pair.

3. Visualization (t-SNE)  
Visualize the latent space of the best-performing layer (identified as out_2 at timestep 21 in our experiments) to see class clustering.  
```  
python visualize_features.py \
    --config config/DDPM_ddpm.yaml \
    --epoch 59 \
    --timestep 21 \
    --blockname 'out_2' \
    --use_amp
```

Output: Saves a t-SNE scatter plot to the output directory.

## Results Summary  
Our experiments with the 5.9M parameter model (60 epochs) yielded the following linear probe accuracies:  
Layer    Accuracy (t=21)  
out_1    63.21%  
out_2    64.29% (Peak)  
out_3    63.69%  
out_4    62.83%  
...  
out_8    42.13%  

Qualitative analysis (t-SNE) confirms emergent clustering of semantic classes (e.g., vehicles vs. animals) at the optimal layer.

