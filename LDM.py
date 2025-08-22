
# train an unconditional Latent Diffusion Model (LDM) on 128 × 128 cat images

# -> Frozen Stable Diffusion VAE (AutoencoderKL)
# -> UNet on 4 × 16 × 16 latents
# -> v-prediction, squared-cosine betas
# -> Mixed precision, EMA (Exponential Moving Average), augmentations, 

# sample grids, checkpointing

import os
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

# ---- Diffusers components (VAE + schedulers + optional UNet) ----

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.models import UNet2DModel


#%%

# ------------------------- CONFIG -------------------------

@dataclass
class Config:
    # Data (images)
  
    train_dir = r'C:\Users\Lav\Desktop\Lav\Datasets\Image datasets\Cats, dogs and other animals image dataset\resized cat images (training and validation)\training images'
    validation_dir = r'C:\Users\Lav\Desktop\Lav\Datasets\Image datasets\Cats, dogs and other animals image dataset\resized cat images (training and validation)\validation images'
    image_size = 128           # 128 x 128 cat images (5153 training, 500 validation)
    
    # Checkpoints, logs, samples
    
    output_dir = r'C:\Users\Lav\Desktop\Python\My projects\Latent Diffusion Model\Runnable-Latent-Diffusion-Model\outputs'
    save_every_steps = 1000          # saving model every 1000 steps
    sample_every_steps = 500         # sampling every 500 steps
    num_sample_rows = 4        # 4 x 4 grid
    
    # Training
    
    num_of_epochs = 50
    batch_size = 16            # increase if VRAM allows; use grad_accum for eff. 64
    grad_accum_steps = 4       # using the concept of gradient accumulation here!, 16 * 4 = eff. 64
    learning_rate = 1e-4
    weight_decay = 0.01
    betas = (0.9, 0.999)
    num_train_steps = 30000    # 25k–40k recommended; adjust as needed
    warmup_steps = 2000
    ema_decay = 0.9999
    max_grad_norm = 1.0
    
    # Mixed precision
    
    use_amp = True
    amp_dtype = torch.float16  # can be customized
    
    # Model (latent space)
    
    latent_channels = 4         # StableDiffusion VAE latent channels
    latent_size = 16            # 128 / 8 (VAE factor) = 16
    
    # U-Net capacity
    
    base_channels = 192
    channel_mult = (1, 2, 3, 4)  # yields downs to 2 x 2
    num_res_blocks = 2
    use_attention_at = (8, 4, 2)  # enable attention at these spatial sizes
    
    # VAE
    
    vae_repo = 'stabilityai/sd-vae-ft-ema'  # or 'stabilityai/sd-vae-ft-mse'
    
    # Sampling
    
    ddim_steps = 50
    seed = 42


#%%

# creating an instance/object of the above 'Config' class, and using it to
# create relevant directories for storing/saving checkpoints, samples and logs

conf = Config()

os.makedirs(conf.output_dir, exist_ok = True)

checkpoint_dir = Path(conf.output_dir) / 'checkpoints'
sample_dir = Path(conf.output_dir) / 'samples'
log_dir = Path(conf.output_dir) / 'logs'
visual_progress_sample_dir = Path(conf.output_dir) / 'visual_progress_samples'

checkpoint_dir.mkdir(parents = True, exist_ok = True)
sample_dir.mkdir(parents = True, exist_ok = True)
log_dir.mkdir(parents = True, exist_ok = True)
visual_progress_sample_dir.mkdir(parents = True, exist_ok = True)


#%%

# ---------------------- UTILITIES ----------------------

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(conf.seed)


#%%

# using GPU

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name())

print('\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'-> Currently using device: {device}')
print('')

def exists(x):
    x_exists = x is not None
    return x_exists


#%%

# EMA (Exponential Moving Average) helper

class EMA:
    def __init__(self, model: nn.Module, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert name in self.shadow
            new = p.detach()
            old = self.shadow[name]
            old.data = self.decay * old.data + (1.0 - self.decay) * new.data

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.data.clone()
            p.data = self.shadow[name].data.clone()

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data = self.backup[name].data.clone()
        self.backup = {}


#%%

# Cosine LR (learning rate) with warmup

# i.e., learning rate scheduling with cosine-shaped decay curve, after a few
# epochs (warmup)

def get_lr(current_step, base_lr, warmup_steps, total_steps):
    # warmup steps (till the current step is a warmup step)
    
    if current_step < warmup_steps:
        return (base_lr * current_step / max(1, warmup_steps))
    
    # after the warmup steps are over, Cosine decay to 10% by end
    
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    min_factor = 0.1
    factor = min_factor + (0.5 * (1 - min_factor) * (1 + math.cos(math.pi * progress)))
    
    return (base_lr * factor)


#%%

# ---------------------- DATA ----------------------

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_size = 128, augment = True):
        self.paths = []
        root = Path(root)
        extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        
        for p in root.rglob('*'):
            if p.suffix.lower() in extensions:
                self.paths.append(str(p))
        
        if len(self.paths) == 0:
            raise RuntimeError(f'-> No images found in {root}')
        
        self.image_size = image_size
        self.augment = augment
        
        # Transforms / image augmentation (keep moderate)
        
        augs = []
        
        if augment:
            augs = augs + [
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomAffine(
                    degrees = 12, translate = (0.05, 0.05), scale = (0.9, 1.1),
                    interpolation = transforms.InterpolationMode.BILINEAR
                ),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            ]
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size, interpolation = transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            *augs,
            transforms.ToTensor(),                           # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),      # [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        img = self.tf(img)
        
        return img


#%%

train_dataset = ImageFolderDataset(conf.train_dir, conf.image_size, augment = True)
validation_dataset = ImageFolderDataset(conf.validation_dir, conf.image_size, augment = False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size = conf.batch_size, shuffle = True, num_workers = 4, pin_memory = True, drop_last = True
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset, batch_size = conf.num_sample_rows ** 2, shuffle = True, num_workers = 2, pin_memory = True, drop_last = True
)

print(f'-> Number of training images: {len(train_dataset)}')
print('')

print(f'-> Number of validation images: {len(validation_dataset)}')
print('')

#%%

# ---------------------- MODELS ----------------------

# 1) Frozen (pre-trained) VAE (StableDiffusion's Variational Auto-Encoder)

print('Loading VAE .....')
print('')

vae = AutoencoderKL.from_pretrained(conf.vae_repo)

vae.requires_grad_(False)

vae.eval()

vae.to(device)

# scaling factor used by SD VAEs

vae_sf = getattr(vae.config, 'scaling_factor', 0.18215)


#%%

# checking version of 'diffusers' package

import diffusers

print(diffusers.__version__)
print('')


#%%

# 2) U-Net in latent space (4 x 16 x 16)

print('Building U-Net .....')
print('')

# UNet2DModel's output has the same number of channels as the input 
# (out_channels = 4)


### -> block output channels:-

block_output_channels = []

for m in conf.channel_mult:
    block_output_channels.append(conf.base_channels * m)


### -> downsampling block layers:-

downsampling_block_layers = []

# adding 'DownBlock2D' layer (len(conf.channel_mult) - 1) times

for _ in range(len(conf.channel_mult) - 1):
    downsampling_block_layers.append('DownBlock2D')

# adding a single 'AttnDownBlock2D' layer at the end

downsampling_block_layers.append('AttnDownBlock2D')


### -> upsampling block layers:-

upsampling_block_layers = []

# adding a single 'AttnUpBlock2D' layer at the beginning

upsampling_block_layers.append('AttnUpBlock2D')

# adding 'UpBlock2D' layer (len(conf.channel_mult) - 1) times

for _ in range(len(conf.channel_mult) - 1):
    upsampling_block_layers.append('UpBlock2D')


U_Net = UNet2DModel(
    sample_size = conf.latent_size,        # 16
    in_channels = conf.latent_channels,    # 4
    out_channels = conf.latent_channels,   # 4 (predict v)
    layers_per_block = conf.num_res_blocks,
    
    block_out_channels = block_output_channels,    
    down_block_types = downsampling_block_layers,
    up_block_types = upsampling_block_layers,
    attention_head_dim = 64
)


# Add attention at small spatial sizes by using AttnDown/Up at ends;
# for more granular control, blocks could be mixed

# moving the U-Net to GPU

U_Net.to(device)

# In case xformers is installed, memory-efficient attention can be enabled

# try:
#     U_Net.enable_xformers_memory_efficient_attention()
#     print('xFormers attention: Enabled')
#     print('')
# except Exception as e:
#     print('xFormers attention: Not enabled - ', e)
#     print('')


#%%

# 3) Diffusion scheduler (training): DDPM Scheduler with squared-cosine betas and v-prediction

noise_scheduler = DDPMScheduler(
    num_train_timesteps = 1000,
    beta_start = 0.00085,
    beta_end = 0.0120,
    
    beta_schedule = 'squaredcos_cap_v2',
    prediction_type = 'v_prediction',
)

# moving to device

alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)


#%%

# 4) Sampler (inference): DDIM

ddim = DDIMScheduler.from_config(noise_scheduler.config)


#%%

# -- OPTIMIZER, EMA (Exponential Moving Average), AMP (Automatic Mixed Precision) -------

optimizer = torch.optim.AdamW(U_Net.parameters(), lr = conf.learning_rate, betas = conf.betas, weight_decay = conf.weight_decay)

scaler = torch.amp.GradScaler('cuda', enabled = conf.use_amp)

# for PyTorch version <= 2.0,
# scaler = torch.cuda.amp.GradScaler(enabled = conf.use_amp)

ema = EMA(U_Net, decay = conf.ema_decay)


#%%

# ---------------------- TRAIN / VALIDATION HELPERS ----------------------

@torch.no_grad()
def encode_to_latents(x: torch.Tensor) -> torch.Tensor:
    # x: (value range = [-1, 1]) 3 x 128 x 128 tensor
    
    posterior = vae.encode(x).latent_dist
    z = posterior.sample() * vae_sf
    
    return z


@torch.no_grad()
def decode_from_latents(z: torch.Tensor) -> torch.Tensor:
    # z: 4 x 16 x 16 tensor
    
    z = z / vae_sf
    x = vae.decode(z).sample
    
    return x.clamp(-1, 1)


def save_grid(tensor_bchw, path, nrow):
    # expects [-1, 1] range
    
    grid = make_grid((tensor_bchw + 1) * 0.5, nrow = nrow, padding = 2, pad_value = 1.0)  # white padding
    
    save_image(grid, path)


@torch.no_grad()
def sample_ema_grid(num = 16, steps = 50, eta = 0.0, guidance_scale = None, filename = 'sample.png'):
    # unconditional, guidance_scale unused here
    
    bs = num
    z = torch.randn((bs, conf.latent_channels, conf.latent_size, conf.latent_size), device = device)
    ddim.set_timesteps(steps, device = device)

    ema.apply_shadow(U_Net)
    
    U_Net.eval()
    
    for t in ddim.timesteps:
        with torch.autocast(device_type = 'cuda', dtype = conf.amp_dtype, enabled = conf.use_amp):
            model_out = U_Net(z, t).sample    # predicts v
        
        z = ddim.step(model_out, t, z).prev_sample
    
    ema.restore(U_Net)

    x = decode_from_latents(z)
    
    save_grid(x, sample_dir / filename, nrow = conf.num_sample_rows)


def validate(step):
    # encode a batch from val, then decode a few noised -> denoised samples 
    # (or just log originals)
    
    if not hasattr(validate, 'val_iter'):
        validate.val_iter = iter(validation_loader)
    
    try:
        batch = next(validate.val_iter)
    except:
        validate.val_iter = iter(validation_loader)
        batch = next(validate.val_iter)
    
    x = batch.to(device)
    
    with torch.no_grad():
        z = encode_to_latents(x)
        
        # visualize reconstructions (not diffusion) to check VAE sanity
        
        x_rec = decode_from_latents(z)
        
    save_grid(x[: conf.num_sample_rows ** 2], sample_dir / f'val_real_step{step}.png', nrow = conf.num_sample_rows)
    
    save_grid(x_rec[: conf.num_sample_rows ** 2], sample_dir / f'val_vae_rec_step{step}.png', nrow = conf.num_sample_rows)


@torch.no_grad()
def save_visual_progress_samples(step, num_samples = 16, ddim_steps = None, filename = None):
    """
    Generates and saves a small grid of images from the EMA model to track training progress visually
    """
    if ddim_steps is None:
        ddim_steps = conf.ddim_steps
    if filename is None:
        filename = f'generated_image_grid_step{step}.png'

    bs = num_samples
    z = torch.randn((bs, conf.latent_channels, conf.latent_size, conf.latent_size), device = device)
    ddim.set_timesteps(ddim_steps, device = device)

    ema.apply_shadow(U_Net)
    
    U_Net.eval()

    for t in ddim.timesteps:
        with torch.autocast(device_type = 'cuda', dtype = conf.amp_dtype, enabled = conf.use_amp):
            model_out = U_Net(z, t).sample
        z = ddim.step(model_out, t, z).prev_sample

    ema.restore(U_Net)
    
    x = decode_from_latents(z)

    save_grid(x, visual_progress_sample_dir / filename, nrow = int(num_samples ** 0.5))



#%%

# ---------------------- TRAINING LOOP ----------------------

### -> to add: save checkpoints only for epochs, not internal individual steps

best_checkpoint_path = None

num_of_epochs = conf.num_of_epochs
grad_accum_steps = conf.grad_accum_steps
base_learning_rate = conf.learning_rate
warmup_steps = conf.warmup_steps

# number of optimizer steps per epoch

steps_per_epoch = len(train_loader) // grad_accum_steps

global_step = 0            # starting off

print('---> Starting training .....')
print('')


for epoch in range(num_of_epochs):
    U_Net.train()
    
    epoch_loss = 0.0

    progress_bar = tqdm(total = steps_per_epoch, desc = f'Epoch {epoch + 1}/{num_of_epochs}', ncols = 100)

    optimizer.zero_grad(set_to_none = True)

    for raw_step, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking = True)  # [-1, 1]
        
        # Encoding to latents
        
        with torch.no_grad():
            z0 = encode_to_latents(batch)  # (B, 4, 16, 16) tensor
        
        # Sampling noise and timesteps
        
        bsz = z0.size(0)
        noise = torch.randn_like(z0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz, ), device = device, dtype = torch.long)
        
        # Adding noise according to schedule

        zt = noise_scheduler.add_noise(z0, noise, timesteps)

        # ---------------- Forward pass ----------------
        
        with torch.autocast(device_type = 'cuda', dtype = conf.amp_dtype, enabled = conf.use_amp):
            pred = U_Net(zt, timesteps).sample
            
            # Predicting v
            
            # target for v-prediction: v = (alpha_t * eps) - (sigma_t * x0)  
            # ('Diffusers' package handles equivalently via scheduler loss norms)
            
            # in practice, MSE(pred, target_v). We can recover target v from 
            # (z0, noise, t)
            
            a_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1).sqrt()
            s_t = (1 - alphas_cumprod[timesteps]).view(-1, 1, 1, 1).sqrt()
            
            target_v = (a_t * noise) - (s_t * z0)

            loss = F.mse_loss(pred, target_v, reduction = 'mean')

        # ------------- Backpropagation (Backward pass) -------------
        
        scaler.scale(loss / grad_accum_steps).backward()
        
        epoch_loss = epoch_loss + loss.item()

        if (raw_step + 1) % grad_accum_steps == 0:
            # grad clip
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(U_Net.parameters(), conf.max_grad_norm)

            # update Learning Rate as per schedule (cosine with warmup)
            
            lr = get_lr(global_step, base_learning_rate, warmup_steps, num_of_epochs * steps_per_epoch)
            
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            scaler.step(optimizer)
            
            scaler.update()
            
            optimizer.zero_grad(set_to_none = True)
            
            # EMA (Exponential Moving Average) update
            
            ema.update(U_Net)
            
            # progress + logging
            
            if (global_step % 50) == 0:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{lr:.2e}'})

            progress_bar.update(1)
            
            # Samples & validation
            
            if global_step % conf.sample_every_steps == 0 and global_step > 0:
                with torch.no_grad():
                    validate(global_step)
                    sample_ema_grid(num = conf.num_sample_rows ** 2, steps = conf.ddim_steps, filename = f'samples_step{global_step}.png')
            
            # Saving visual image progress (the above grid of generated images) every 500 global steps
            
            if global_step % 500 == 0 and global_step > 0:
                save_visual_progress_samples(step = global_step, num_samples = conf.num_sample_rows ** 2)    

            # Saving checkpoint
            
            if global_step % conf.save_every_steps == 0 and global_step > 0:
                checkpoint_path = checkpoint_dir / f'ldm_step{global_step}.pt'
                
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'unet': U_Net.state_dict(),
                    'ema': ema.shadow,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'config': conf.__dict__,
                }, checkpoint_path)
                
                best_checkpoint_path = checkpoint_path
                
                print(f'Saved checkpoint: {checkpoint_path}')
                print('')

            global_step = global_step + 1

            # (breaking the epoch loop and) starting a new loop once 
            # 'steps_per_epoch' steps are completed
            # essentially len(train_loader) number of steps
            
            if (raw_step + 1) // grad_accum_steps >= steps_per_epoch:
                break
            
            # OR, equivalently, if (raw_step + 1) >= len(train_loader): break
    
    
    # the epoch has been completed    
    
    progress_bar.close()

    # computing and logging the average loss at the end of the epoch

    avg_loss = epoch_loss / steps_per_epoch
    
    print(f'Epoch {epoch + 1}/{num_of_epochs} finished | avg_loss = {avg_loss:.4f}')


# all the epochs have been completed, and hence, the entire training

print('---> Training has been completed')
print('')

# Final EMA samples

print('-> Sampling final EMA grid ...')
print('')

sample_ema_grid(num = conf.num_sample_rows ** 2, steps = conf.ddim_steps, filename = 'samples_final.png')

# Saving the final checkpoint

final_checkpoint = checkpoint_dir / f'ldm_final_step{global_step}.pt'

torch.save({
    'epoch': epoch,
    'step': global_step,
    'unet': U_Net.state_dict(),
    'ema': ema.shadow,
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'config': conf.__dict__,
}, final_checkpoint)

print(f'---> Saved final checkpoint: {final_checkpoint}')
print('')

print('Congrats!! 🎉🎊🧸')
print('')



#%%

# ------------------ RESUME TRAINING SETUP ------------------
# (resuming visual training progress functionality has to be added)

from pathlib import Path
import torch
from tqdm import tqdm


def get_latest_checkpoint(checkpoint_dir):
    """
    Returns the latest checkpoint file, or None if none exists
    """
    
    checkpoint_dir = Path(checkpoint_dir)
    
    checkpoint_files = sorted(checkpoint_dir.glob('ldm_step*.pt'), key=lambda x: int(x.stem.split('step')[1]))
    
    if checkpoint_files:
        return checkpoint_files[-1]
    else:
        return None
    

def load_checkpoint_resume(checkpoint_dir, model, optimizer = None, scaler = None, ema = None, device = 'cuda', specific_file = None):
    """
    Loads checkpoint for resuming training, restoring step, LR, EMA, optimizer, and scaler
    """
    
    checkpoint_dir = Path(checkpoint_dir)

    if specific_file:
        checkpoint_path = checkpoint_dir / specific_file
    else:
        checkpoint_path = get_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            print('No checkpoints found. Starting training from scratch')
            print('')
            return 0

    print(f'Loading checkpoint: {checkpoint_path}')
    print('')
    
    checkpoint = torch.load(checkpoint_path, map_location = device)

    # Restoring model (model training history)
    
    model.load_state_dict(checkpoint['unet'])

    # Restoring optimizer
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Restoring AMP scaler
    
    if scaler:
        scaler.load_state_dict(checkpoint['scaler'])

    # Restoring EMA
    
    if ema:
        ema.shadow = checkpoint['ema']

    # Resuming step
    
    step = checkpoint['step'] + 1

    # Adjusting optimizer Learning Rate according to current step
    
    if optimizer:
        lr = get_lr(step, conf.learning_rate, conf.warmup_steps, conf.num_train_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    print(f'--> Resuming training from step {step} with Learning Rate = {lr:.2e}')
    print('')
    
    return step


def resume_validation_and_sampling(validation_loader, sample_dir, ema_model, unet_model, step, conf):
    """
    Prepares validation and EMA sampling functions to continue seamlessly from a particular checkpoint
    """
    
    val_iter = iter(validation_loader)

    @torch.no_grad()
    def validate_resumed(current_step):
        nonlocal val_iter
        
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(validation_loader)
            batch = next(val_iter)

        x = batch.to(device)
        z = encode_to_latents(x)
        x_rec = decode_from_latents(z)

        save_grid(x[: conf.num_sample_rows ** 2], sample_dir / f'val_real_step{current_step}.png', nrow = conf.num_sample_rows)
        save_grid(x_rec[: conf.num_sample_rows ** 2], sample_dir / f'val_vae_rec_step{current_step}.png', nrow = conf.num_sample_rows)

    @torch.no_grad()
    def sample_ema_resumed(num = None, steps = None, filename = None):
        if num is None:
            num = conf.num_sample_rows ** 2
        if steps is None:
            steps = conf.ddim_steps
        if filename is None:
            filename = f'samples_step{step}.png'

        bs = num
        z = torch.randn((bs, conf.latent_channels, conf.latent_size, conf.latent_size), device = device)
        ddim.set_timesteps(steps, device = device)

        ema_model.apply_shadow(unet_model)
        unet_model.eval()

        for t in ddim.timesteps:
            with torch.autocast(device_type = 'cuda', dtype = conf.amp_dtype, enabled = conf.use_amp):
                model_out = unet_model(z, t).sample            
            z = ddim.step(model_out, t, z).prev_sample

        ema_model.restore(unet_model)
        x = decode_from_latents(z)
        save_grid(x, sample_dir / filename, nrow = conf.num_sample_rows)

    return validate_resumed, sample_ema_resumed



# ------------------ HOW TO USE THE ABOVE SETUP ------------------

# 1) Load checkpoint (latest checkpoint or a specific one)

step = load_checkpoint_resume(
    checkpoint_dir = checkpoint_dir,
    model = U_Net,
    optimizer = optimizer,
    scaler = scaler,
    ema = ema,
    device = device,
    specific_file = None  # or specify e.g. "ldm_step1000.pt"
)


# 2) Prepare resumed validation and EMA sampling functions

validate, sample_ema_grid = resume_validation_and_sampling(
    validation_loader, sample_dir, ema, U_Net, step, conf
)

# 3) Resume training loop

progress_bar = tqdm(total = conf.num_train_steps, initial = step, desc = 'train', ncols = 100)


