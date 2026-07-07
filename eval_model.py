"""
eval_model.py — Standalone evaluation for a trained NoMaD checkpoint.

Usage (run from anywhere):
    conda activate nomad_train
    python eval_model.py                          # auto-finds latest checkpoint
    python eval_model.py --checkpoint path/to/ema_latest.pth
    python eval_model.py --config train/config/nomad_finetune.yaml
"""

import os
import sys
import glob
import argparse
import yaml
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
sys.path.insert(0, TRAIN_DIR)

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.training.train_utils import model_output, ACTION_STATS

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.nn.functional as F
import tqdm


def load_config(config_path: str) -> dict:
    defaults_path = os.path.join(TRAIN_DIR, "config", "defaults.yaml")
    with open(defaults_path) as f:
        config = yaml.safe_load(f)
    with open(config_path) as f:
        config.update(yaml.safe_load(f))
    return config


def build_model(config: dict) -> torch.nn.Module:
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    return model


def find_latest_checkpoint(config: dict) -> str:
    logs_dir = os.path.join(TRAIN_DIR, "logs", config["project_name"])
    runs = sorted(glob.glob(os.path.join(logs_dir, "*")), key=os.path.getmtime, reverse=True)
    for run in runs:
        # Prefer ema_latest.pth
        ema_latest = os.path.join(run, "ema_latest.pth")
        if os.path.isfile(ema_latest):
            return ema_latest
        # Training code has a bug: saves ema_N.pth but not ema_latest.pth
        # Find the highest-epoch ema checkpoint
        ema_files = sorted(
            glob.glob(os.path.join(run, "ema_*.pth")),
            key=lambda p: int(os.path.basename(p).replace("ema_", "").replace(".pth", ""))
            if os.path.basename(p) != "ema_latest.pth" else -1
        )
        ema_files = [p for p in ema_files if os.path.basename(p) != "ema_latest.pth"]
        if ema_files:
            return ema_files[-1]  # highest epoch
        latest_path = os.path.join(run, "latest.pth")
        if os.path.isfile(latest_path):
            return latest_path
    raise FileNotFoundError(f"No checkpoint found under {logs_dir}")


@torch.no_grad()
def evaluate(model, noise_scheduler, dataloader, device, config):
    model.eval()
    metrics = {
        "gc_dist_loss": [],
        "gc_action_loss": [],
        "gc_action_waypts_cos_sim": [],
        "uc_action_loss": [],
        "uc_action_waypts_cos_sim": [],
    }

    pred_horizon = config["len_traj_pred"]
    action_dim = 2

    for data in tqdm.tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
        (obs_image, goal_image, actions, distance, goal_pos, dataset_idx, action_mask) = data

        obs_images = torch.split(obs_image, 3, dim=1)
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        batch_obs = torch.cat([transform(o) for o in obs_images], dim=1).to(device)
        batch_goal = transform(goal_image).to(device)
        action_mask = action_mask.to(device)
        distance = distance.float().to(device)
        # ground truth: cumulative waypoints (same space as model_output returns)
        action_label = actions.to(device)

        out = model_output(
            model, noise_scheduler,
            batch_obs, batch_goal,
            pred_horizon, action_dim,
            num_samples=1, device=device,
        )
        uc_actions = out["uc_actions"]
        gc_actions = out["gc_actions"]
        gc_distance = out["gc_distance"]

        def reduce(t):
            while t.dim() > 1:
                t = t.mean(dim=-1)
            return (t * action_mask).mean() / (action_mask.mean() + 1e-2)

        metrics["gc_dist_loss"].append(
            F.mse_loss(gc_distance, distance.unsqueeze(-1)).item()
        )
        metrics["gc_action_loss"].append(
            reduce(F.mse_loss(gc_actions, action_label, reduction="none")).item()
        )
        metrics["gc_action_waypts_cos_sim"].append(
            reduce(F.cosine_similarity(gc_actions[:, :, :2], action_label[:, :, :2], dim=-1)).item()
        )
        metrics["uc_action_loss"].append(
            reduce(F.mse_loss(uc_actions, action_label, reduction="none")).item()
        )
        metrics["uc_action_waypts_cos_sim"].append(
            reduce(F.cosine_similarity(uc_actions[:, :, :2], action_label[:, :, :2], dim=-1)).item()
        )

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(TRAIN_DIR, "config", "nomad_finetune.yaml"))
    parser.add_argument("--checkpoint", default=None, help="Path to .pth file (default: auto-find ema_latest.pth)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)

    # Find checkpoint
    ckpt_path = args.checkpoint or find_latest_checkpoint(config)
    print(f"Loading checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build and load model
    model = build_model(config).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # Build test dataloaders
    test_datasets = {}
    for dataset_name, data_config in config["datasets"].items():
        data_config.setdefault("negative_mining", True)
        data_config.setdefault("goals_per_obs", 1)
        data_config.setdefault("end_slack", 0)
        data_config.setdefault("waypoint_spacing", 1)
        if "test" in data_config:
            test_datasets[dataset_name] = ViNT_Dataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=data_config["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=data_config["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                context_size=config["context_size"],
                context_type=config.get("context_type", "temporal"),
                end_slack=data_config["end_slack"],
                goals_per_obs=data_config["goals_per_obs"],
                normalize=config["normalize"],
                goal_type=config["goal_type"],
            )

    if not test_datasets:
        print("No test datasets found in config.")
        return

    print(f"\n{'='*60}")
    print(f"  NoMaD Evaluation Results")
    print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
    print(f"{'='*60}")

    for name, dataset in test_datasets.items():
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        print(f"\nDataset: {name}  ({len(dataset)} samples)")
        results = evaluate(model, noise_scheduler, loader, device, config)
        print(f"  {'Metric':<35} {'Value':>10}")
        print(f"  {'-'*46}")
        for k, v in results.items():
            print(f"  {k:<35} {v:>10.4f}")

    print(f"\n{'='*60}")
    print("Interpretation:")
    print("  gc_dist_loss          : lower is better (distance prediction MSE)")
    print("  gc_action_loss        : lower is better (goal-conditioned action MSE)")
    print("  gc_action_waypts_cos_sim : higher is better (1.0 = perfect, >0.7 is good)")
    print("  uc_action_loss        : lower is better (unconditional action MSE)")
    print("  uc_action_waypts_cos_sim : goal-free navigation quality (>0.5 is decent)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
