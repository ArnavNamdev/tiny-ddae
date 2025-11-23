import argparse
import os

import torch
import yaml
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from ema_pytorch import EMA

from model.models import get_models_class
from utils import Config, get_optimizer, init_seeds

# ===== single-GPU training version =====

def train(opt):
    # This version runs everything on a single GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    yaml_path = opt.config
    use_amp = opt.use_amp

    with open(yaml_path, 'r') as f:
        opt = yaml.full_load(f)
    print(opt)
    opt = Config(opt)
    model_dir = os.path.join(opt.save_dir, "ckpts")
    vis_dir = os.path.join(opt.save_dir, "visual")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    DIFFUSION, NETWORK = get_models_class(opt.model_type, opt.net_type)
    diff = DIFFUSION(nn_model=NETWORK(**opt.network),
                     **opt.diffusion,
                     device=device,
                     )
    diff.to(device)
    
    ema = EMA(diff, beta=opt.ema, update_after_step=0, update_every=1)
    ema.to(device)

    # REMOVED: SyncBatchNorm and DistributedDataParallel wrappers

    train_set = get_dataset(name=opt.dataset, root="./data", train=True, flip=opt.flip, download=True)
    print("train dataset:", len(train_set))

    # CHANGED: Using a standard DataLoader instead of DataLoaderDDP
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=opt.batch_size,
                                               shuffle=True)

    lr = opt.lrate
    # REMOVED: DDP learning rate multiplier
    
    optim = get_optimizer(diff.parameters(), opt, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print("loading model at", target)
        checkpoint = torch.load(target, map_location=device)
        diff.load_state_dict(checkpoint['MODEL'])
        ema.load_state_dict(checkpoint['EMA'])
        optim.load_state_dict(checkpoint['opt'])

    for ep in range(opt.load_epoch + 1, opt.n_epoch):
        for g in optim.param_groups:
            g['lr'] = lr * min((ep + 1.0) / opt.warm_epoch, 1.0) # warmup
        
        # training
        diff.train()
        now_lr = optim.param_groups[0]['lr']
        print(f'epoch {ep}, lr {now_lr:f}')
        loss_ema = None
        pbar = tqdm(train_loader)

        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = diff(x, use_amp=use_amp)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=diff.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            # logging
            # REMOVED: reduce_tensor for loss, since we only have one GPU
            
            ema.update()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")

        # testing
        if ep % 100 == 0 or ep == opt.n_epoch - 1:
            if opt.model_type == 'DDPM':
                ema_sample_method = ema.ema_model.ddim_sample
            elif opt.model_type == 'EDM':
                ema_sample_method = ema.ema_model.edm_sample

            ema.ema_model.eval()
            with torch.no_grad():
                x_gen = ema_sample_method(opt.n_sample, x.shape[1:])
            
            x_real = x[:opt.n_sample]
            x_all = torch.cat([x_gen.cpu(), x_real.cpu()])
            grid = make_grid(x_all, nrow=10)

            save_path = os.path.join(vis_dir, f"image_ep{ep}_ema.png")
            save_image(grid, save_path)
            print('saved image at', save_path)

            if opt.save_model:
                checkpoint = {
                    'MODEL': diff.state_dict(),
                    'EMA': ema.state_dict(),
                    'opt': optim.state_dict(),
                }
                save_path = os.path.join(model_dir, f"model_{ep}.pth")
                torch.save(checkpoint, save_path)
                print('saved model at', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    # REMOVED: local_rank is not needed for single GPU
    parser.add_argument("--use_amp", action='store_true', default=False)
    opt = parser.parse_args()
    print(opt)

    # We can hardcode the seed since it's a single run
    init_seeds(no=0)
    
    # REMOVED: All torch.distributed setup
    
    train(opt)