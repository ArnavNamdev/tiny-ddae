import argparse
import os
import yaml
from functools import partial

import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from ema_pytorch import EMA

# --- Import necessary components from the ddae repository ---
# Make sure your current directory is /content/drive/MyDrive/ddae/
from model.models import get_models_class
from utils import Config, init_seeds

# Define the device globally for single-GPU use
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Helper function to load the trained model (adapted from linear.py) ---
def get_model(opt, load_epoch):
    """Loads the EMA weights of the trained DDAE model."""
    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)
    target = os.path.join(opt.save_dir, "ckpts", f"model_{load_epoch}.pth")
    print(f"Loading model checkpoint from: {target}")
    checkpoint = torch.load(target, map_location=device)
    
    # Load EMA weights for better evaluation performance
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)
    try:
        ema.load_state_dict(checkpoint['EMA'])
        print("EMA weights loaded successfully.")
    except KeyError:
        print("EMA weights not found in checkpoint, loading standard model weights instead.")
        diff.load_state_dict(checkpoint['MODEL'])
        ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1) # Re-init EMA with loaded weights
        ema.to(device)

    model = ema.ema_model
    model.eval() # Set model to evaluation mode
    return model

# --- Main Feature Extraction and Visualization Logic ---
def visualize(opt_cli):
    """Extracts features, runs t-SNE, and plots the results."""
    
    # 1. Load Configuration from YAML
    yaml_path = opt_cli.config
    with open(yaml_path, 'r') as f:
        opt_yaml = yaml.full_load(f)
    print("Loaded configuration:", opt_yaml)
    opt = Config(opt_yaml) # Convert dict to Config object

    # 2. Load the Pre-trained Model
    model = get_model(opt, opt_cli.epoch)

    # 3. Load CIFAR-10 Test Dataset
    print("Loading CIFAR-10 test dataset...")
    # Use standard CIFAR-10 transforms (matching training setup is often good)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Use a larger batch size for faster feature extraction
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)
    print(f"Dataset loaded: {len(test_set)} images.")

    # 4. Extract Features
    # Get the specific timestep and block name from the command line or config
    target_timestep = opt_cli.timestep if opt_cli.timestep is not None else opt.linear['timestep']
    target_blockname = opt_cli.blockname if opt_cli.blockname is not None else opt.linear['blockname']
    print(f"Extracting features from block '{target_blockname}' at timestep {target_timestep}...")
    
    all_features = []
    all_labels = []
    
    # Define the feature extraction function using the loaded model
    # Note: `get_feature` might need `use_amp=False` if not using mixed precision here
    feat_func = partial(model.get_feature, norm=False, use_amp=opt_cli.use_amp) 

    with torch.no_grad(): # Ensure no gradients are calculated
        for images, labels in tqdm(test_loader, desc="Extracting Features"):
            images = images.to(device)
            # Create the timestep tensor for the current batch
            t = torch.full((images.shape[0],), target_timestep, device=device, dtype=torch.long)
            
            # Get features for the current batch
            # `get_feature` returns a dictionary of features from different blocks
            batch_features_dict = feat_func(images, t) 
            
            # Select the features from the target block
            batch_features = batch_features_dict[target_blockname]
            
            # Flatten spatial dimensions if necessary (e.g., from [B, C, H, W] to [B, C*H*W])
            # Adjust this based on the actual shape of features[target_blockname]
            if batch_features.ndim > 2:
                 # Global Average Pooling is a common way to summarize spatial features
                 batch_features = torch.mean(batch_features, dim=[2, 3]) 
                 # Alternatively, flatten: batch_features = batch_features.view(batch_features.size(0), -1)

            all_features.append(batch_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    features_np = np.concatenate(all_features, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    print(f"Feature extraction complete. Feature shape: {features_np.shape}")

    # 5. Run t-SNE
    print("Running t-SNE... (This may take a few minutes)")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', n_iter=1000, random_state=42, n_jobs=-1)
    features_2d = tsne.fit_transform(features_np)
    print("t-SNE finished.")

    # 6. Plot Results
    print("Generating plot...")
    plt.figure(figsize=(12, 10))
    # Get unique class labels and assign colors
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    num_classes = len(classes)
    colors = plt.cm.get_cmap("tab10", num_classes) # Use a distinct colormap

    for i in range(num_classes):
        # Select data belonging to the current class
        idx = labels_np == i
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], color=colors(i), label=classes[i], s=10, alpha=0.7)

    plt.title(f't-SNE Visualization of Features from Block {target_blockname} (t={target_timestep}, Epoch {opt_cli.epoch})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(markerscale=2.)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Ensure the output directory exists
    output_dir = os.path.join(opt.save_dir, "visual")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plot_filename = f"tsne_ep{opt_cli.epoch}_t{target_timestep}_{target_blockname.replace('/', '_')}.png"
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to: {save_path}")
    # plt.show() # Uncomment if running interactively and want to see the plot immediately

# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the saved model checkpoint to load.")
    parser.add_argument("--timestep", type=int, default=None, help="Timestep 't' for feature extraction (overrides config).")
    parser.add_argument("--blockname", type=str, default=None, help="Block name (e.g., 'out_6') for feature extraction (overrides config).")
    parser.add_argument("--use_amp", action='store_true', default=False, help="Use Automatic Mixed Precision for feature extraction (match training).")
    
    opt_cli = parser.parse_args()
    print("Command Line Args:", opt_cli)

    # Initialize seed for reproducibility (optional but good practice)
    init_seeds(no=0) 
    
    visualize(opt_cli)


### How to Run It

# 1.  **Save the Code:** Make sure this code is saved as `visualize_features.py` in your `/content/drive/MyDrive/ddae/` directory.
# 2.  **Install Scikit-learn:** If you haven't already, install it:
#     ```python
#     !pip install scikit-learn matplotlib
#     ```
# 3.  **Run the Script:** Execute it from your Colab notebook, telling it which config file and epoch to use. You can optionally specify the exact layer and timestep, otherwise it will use the defaults from your YAML file. Use the best combination you found (epoch 59, `t=21`, `out_2`).
#     ```python
#     !python visualize_features.py \
#         --config config/DDPM_ddpm.yaml \
#         --epoch 59 \
#         --timestep 21 \
#         --blockname 'out_2' \
#         --use_amp 
    
