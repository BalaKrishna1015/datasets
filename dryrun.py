"""
Quick smoke test — runs 5 forward+backward passes through the full NoMaD pipeline.
Finishes in ~2 minutes on CPU. Validates: data loading, model, loss, optimizer step.
"""
import sys, os, yaml, torch, numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torchvision import transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

sys.path.insert(0, r'c:\Users\aryan\nomad\visualnav-transformer\train')

TRAIN_DIR = r'c:\Users\aryan\nomad\visualnav-transformer\train'

with open(os.path.join(TRAIN_DIR, 'config', 'defaults.yaml')) as f:
    config = yaml.safe_load(f)
with open(os.path.join(TRAIN_DIR, 'config', 'nomad_smoketest.yaml')) as f:
    config.update(yaml.safe_load(f))

print("=" * 55)
print("  NoMaD Smoke Test  (5 batches, CPU)")
print("=" * 55)

# ── Dataset ───────────────────────────────────────────────────────────────────
from vint_train.data.vint_dataset import ViNT_Dataset
ds_cfg = list(config['datasets'].values())[0]
ds = ViNT_Dataset(
    data_folder=ds_cfg['data_folder'],
    data_split_folder=ds_cfg['train'],
    dataset_name='custom_dataset',
    image_size=config['image_size'],
    waypoint_spacing=ds_cfg.get('waypoint_spacing', 1),
    min_dist_cat=config['distance']['min_dist_cat'],
    max_dist_cat=config['distance']['max_dist_cat'],
    min_action_distance=config['action']['min_dist_cat'],
    max_action_distance=config['action']['max_dist_cat'],
    negative_mining=ds_cfg.get('negative_mining', True),
    len_traj_pred=config['len_traj_pred'],
    learn_angle=config['learn_angle'],
    context_size=config['context_size'],
    context_type=config['context_type'],
    end_slack=ds_cfg.get('end_slack', 0),
    goals_per_obs=ds_cfg.get('goals_per_obs', 1),
    normalize=config['normalize'],
    goal_type=config['goal_type'],
)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
print(f"[1/5] Dataset loaded: {len(ds)} samples")

# ── Model ─────────────────────────────────────────────────────────────────────
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

vision_encoder = replace_bn_with_gn(NoMaD_ViNT(
    obs_encoding_size=config['encoding_size'],
    context_size=config['context_size'],
    mha_num_attention_heads=config['mha_num_attention_heads'],
    mha_num_attention_layers=config['mha_num_attention_layers'],
    mha_ff_dim_factor=config['mha_ff_dim_factor'],
))
noise_pred_net = ConditionalUnet1D(
    input_dim=2, global_cond_dim=config['encoding_size'],
    down_dims=config['down_dims'], cond_predict_scale=config['cond_predict_scale'],
)
model = NoMaD(vision_encoder=vision_encoder,
              noise_pred_net=ConditionalUnet1D(
                  input_dim=2, global_cond_dim=config['encoding_size'],
                  down_dims=config['down_dims'], cond_predict_scale=config['cond_predict_scale']),
              dist_pred_net=DenseNetwork(config['encoding_size']))

ema = EMAModel(model=model, power=0.75)
optimizer = AdamW(model.parameters(), lr=float(config['lr']))
noise_scheduler = DDPMScheduler(
    num_train_timesteps=config['num_diffusion_iters'],
    beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')

print(f"[2/5] Model built: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ── 5 Training steps ──────────────────────────────────────────────────────────
print("[3/5] Running 5 forward+backward passes...")
model.train()
device = torch.device('cpu')
model.to(device)

losses = []
for step, batch in enumerate(loader):
    if step >= 5:
        break

    obs_images, goal_image, dist_label, action_label, goal_pos, *_ = batch

    # Reshape obs: [B, C*(context+1), H, W] → apply transform per frame
    B = obs_images.shape[0]
    obs_images = obs_images.to(device).float() / 255.0
    goal_image  = goal_image.to(device).float() / 255.0

    # Mask goal randomly (NoMaD behaviour)
    goal_mask = (torch.rand(B) < config['goal_mask_prob']).long().to(device)

    obsgoal_cond = model('vision_encoder',
                         obs_img=obs_images,
                         goal_img=goal_image,
                         input_goal_mask=goal_mask)

    # Distance prediction loss
    dist_pred = model('dist_pred_net', obsgoal_cond=obsgoal_cond)
    dist_label_t = (dist_label[:, 0, 0]).to(device).float().unsqueeze(1)
    dist_loss = torch.nn.functional.mse_loss(dist_pred, dist_label_t)

    # Diffusion loss — dist_label [B,8,2] is the ground truth waypoint trajectory
    naction = dist_label.to(device).float()   # [B, 8, 2]
    noise = torch.randn_like(naction)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (B,), device=device).long()
    noisy = noise_scheduler.add_noise(naction, noise, timesteps)
    noise_pred = model('noise_pred_net', sample=noisy, timestep=timesteps,
                       global_cond=obsgoal_cond)
    diff_loss = torch.nn.functional.mse_loss(noise_pred, noise)

    loss = float(config['alpha']) * dist_loss + (1 - float(config['alpha'])) * diff_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.step(model)

    losses.append(loss.item())
    print(f"  step {step+1}/5 | loss={loss.item():.4f}  dist={dist_loss.item():.4f}  diff={diff_loss.item():.4f}")

print(f"[4/5] Average loss over 5 steps: {np.mean(losses):.4f}")

# ── Save checkpoint ───────────────────────────────────────────────────────────
ckpt_path = os.path.join(TRAIN_DIR, 'logs', 'smoketest_checkpoint.pth')
torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)
print(f"[5/5] Checkpoint saved → {ckpt_path}")

print()
print("=" * 55)
print("  SMOKE TEST PASSED — training pipeline is working!")
print("=" * 55)

import os, sys, yaml, torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

sys.path.insert(0, r'c:\Users\aryan\nomad\visualnav-transformer\train')

# ── Load config (same logic as train.py) ──────────────────────────────────────
with open(r'c:\Users\aryan\nomad\visualnav-transformer\train\config\defaults.yaml') as f:
    config = yaml.safe_load(f)
with open(r'c:\Users\aryan\nomad\visualnav-transformer\train\config\nomad_finetune.yaml') as f:
    config.update(yaml.safe_load(f))

print("Config loaded OK")
print(f"  model_type : {config['model_type']}")
print(f"  batch_size : {config['batch_size']}")
print(f"  epochs     : {config['epochs']}")
print(f"  image_size : {config['image_size']}")
print()

# ── Load dataset ──────────────────────────────────────────────────────────────
from vint_train.data.vint_dataset import ViNT_Dataset

train_datasets = []
test_datasets  = {}

for dataset_name, data_config in config['datasets'].items():
    for split in ['train', 'test']:
        if split not in data_config:
            continue
        ds = ViNT_Dataset(
            data_folder        = data_config['data_folder'],
            data_split_folder  = data_config[split],
            dataset_name       = dataset_name,
            image_size         = config['image_size'],
            waypoint_spacing   = data_config.get('waypoint_spacing', 1),
            min_dist_cat       = config['distance']['min_dist_cat'],
            max_dist_cat       = config['distance']['max_dist_cat'],
            min_action_distance= config['action']['min_dist_cat'],
            max_action_distance= config['action']['max_dist_cat'],
            negative_mining    = data_config.get('negative_mining', True),
            len_traj_pred      = config['len_traj_pred'],
            learn_angle        = config['learn_angle'],
            context_size       = config['context_size'],
            context_type       = config['context_type'],
            end_slack          = data_config.get('end_slack', 0),
            goals_per_obs      = data_config.get('goals_per_obs', 1),
            normalize          = config['normalize'],
            goal_type          = config['goal_type'],
        )
        if split == 'train':
            train_datasets.append(ds)
            print(f"Train dataset '{dataset_name}': {len(ds)} samples")
        else:
            test_datasets[f'{dataset_name}_{split}'] = ds
            print(f"Test  dataset '{dataset_name}': {len(ds)} samples")

train_dataset = ConcatDataset(train_datasets)
train_loader  = DataLoader(train_dataset, batch_size=config['batch_size'],
                           shuffle=True, num_workers=0, drop_last=False)
print(f"\nTotal train samples : {len(train_dataset)}")
print(f"Batches per epoch   : {len(train_loader)}")

# ── Fetch one batch to validate shapes ────────────────────────────────────────
print("\nFetching one batch...")
batch = next(iter(train_loader))
names = ["obs_image", "goal_image", "dist_label", "action_label", "goal_pos"]
for i, v in enumerate(batch):
    name = names[i] if i < len(names) else f"item_{i}"
    shape = v.shape if hasattr(v, 'shape') else type(v)
    print(f"  {name:30s}: {shape}")

# ── Build the model ───────────────────────────────────────────────────────────
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

vision_encoder = NoMaD_ViNT(
    obs_encoding_size      = config['encoding_size'],
    context_size           = config['context_size'],
    mha_num_attention_heads= config['mha_num_attention_heads'],
    mha_num_attention_layers=config['mha_num_attention_layers'],
    mha_ff_dim_factor      = config['mha_ff_dim_factor'],
)
vision_encoder = replace_bn_with_gn(vision_encoder)

noise_pred_net = ConditionalUnet1D(
    input_dim        = 2,
    global_cond_dim  = config['encoding_size'],
    down_dims        = config['down_dims'],
    cond_predict_scale=config['cond_predict_scale'],
)
dist_pred_net = DenseNetwork(embedding_dim=config['encoding_size'])
model = NoMaD(vision_encoder=vision_encoder,
              noise_pred_net=noise_pred_net,
              dist_pred_net=dist_pred_net)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel built OK")
print(f"  Trainable parameters: {total_params:,}")

print()
print("DRY-RUN PASSED — pipeline is ready to train!")
