# Changes Made to visualnav-transformer

All changes made to prepare the NoMaD model for fine-tuning on a custom dataset.

---

## 1. Bug Fixes in Existing Code

### `train/train.py`

**a) Hardcoded wandb entity**
- **Problem:** `wandb.init()` had `entity="gnmv2"` hardcoded — would log runs under someone else's account.
- **Fix:** Changed to `entity=config.get("wandb_entity", None)` — now read from the yaml config.

**b) `ViB` vision encoder undefined**
- **Problem:** The `elif config["vision_encoder"] == "vib":` branch called `ViB(...)` which was never imported and `vib_placeholder.py` is empty — would silently crash with `NameError`.
- **Fix:** Replaced with a clear `raise NotImplementedError(...)` message.

**c) Relative paths broke when running from outside `train/`**
- **Problem:** `train.py` opened `config/defaults.yaml` and saved logs to `logs/` using relative paths, so it only worked if you `cd train/` first.
- **Fix:** All paths now resolved relative to the script's own location using `os.path.dirname(os.path.abspath(__file__))`.

**d) `persistent_workers=True` with `num_workers=0`**
- **Problem:** PyTorch raises `ValueError: persistent_workers option needs num_workers > 0` when both are set.
- **Fix:** Changed to `persistent_workers=config["num_workers"] > 0`.

---

### `train/vint_train/training/train_utils.py`

**Unguarded `wandb.log()` calls crash when `use_wandb: False`**
- **Problem:** `train_nomad()` and `evaluate_nomad()` called `wandb.log()` 6 times without checking `use_wandb`, causing `wandb.errors.Error: You must call wandb.init() before wandb.log()` even when wandb was disabled in config.
- **Fix:** Wrapped all 6 bare `wandb.log()` calls inside `if use_wandb:` guards.

  Affected lines (before fix):
  ```python
  wandb.log({"total_loss": loss_cpu})                          # line 669
  wandb.log({"dist_loss": dist_loss.item()})                   # line 670
  wandb.log({"diffusion_loss": diffusion_loss.item()})         # line 671
  wandb.log({"diffusion_eval_loss (random masking)": ...})     # line 873
  wandb.log({"diffusion_eval_loss (no masking)": ...})         # line 874
  wandb.log({"diffusion_eval_loss (goal masking)": ...})       # line 875
  ```

---

### `train/vint_train/data/vint_dataset.py`

**a) Python 2 pickle files fail to load**
- **Problem:** The dataset's `traj_data.pkl` files were saved with Python 2. Loading them in Python 3 without `encoding='latin1'` raises `UnicodeDecodeError`.
- **Fix:** Both `pickle.load(f)` calls changed to `pickle.load(f, encoding='latin1')`.
  - Line 220: index cache loading
  - Line 281: trajectory data loading

**b) LMDB map size too large for Windows**
- **Problem:** `lmdb.open(cache_filename, map_size=2**40)` reserves 1TB of virtual address space. Windows cannot handle this reservation and raises `lmdb.Error: There is not enough space on the disk`.
- **Fix:** Changed to `map_size=2**33` (8GB) — sufficient for this dataset on Windows.

---

## 2. New Config Files

### `train/config/nomad_finetune.yaml`
Clean fine-tuning config ready for the custom dataset. Key settings vs the original `nomad.yaml`:
- `lr: 5e-5` (lower than default `1e-4` — better for fine-tuning)
- `batch_size: 64` (reduced from 256)
- `num_workers: 0` (Windows-safe; set to 4+ on AWS Linux)
- `use_wandb: False` (disabled by default; enable by setting `True` and adding `wandb_entity`)
- `epochs: 30`
- `goal_type: image` (explicitly set — was missing from original `nomad.yaml`)
- Commented-out `load_run` field for loading pretrained weights
- Dataset section pre-filled with `custom_dataset` pointing to the local dataset paths

### `train/config/nomad_smoketest.yaml`
Minimal config for quick local validation:
- `batch_size: 4`, `epochs: 1`, `num_workers: 0`
- `print_log_freq: 10` — prints loss every 10 batches
- Same dataset paths as `nomad_finetune.yaml`

---

## 3. Dataset Preparation

### Dataset location
```
visualnav-transformer/datasets/
├── traj1/   (191 images + traj_data.pkl)
├── traj2/   (362 images + traj_data.pkl)
├── ...
└── traj20/  (217 images + traj_data.pkl)
```
Total: **20 trajectories, 5,637 images**

### Orphan images removed
Two trajectories had 1 extra image with no corresponding odometry entry:
- `traj6/270.jpg` — removed (position array had 270 entries, images had 271)
- `traj8/287.jpg` — removed (position array had 287 entries, images had 288)

### Waypoint spacing computed
Average distance between consecutive odometry positions computed from all 20 trajectories:
- **Mean: 0.0383m**, Median: 0.0441m
- Registered in `train/vint_train/data/data_config.yaml`:
  ```yaml
  custom_dataset:
    metric_waypoint_spacing: 0.04
  ```

### Train/test split created
`data_split.py` run with `--split 0.8`:
- **Train (16 trajectories):** traj1, 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19
- **Test (4 trajectories):** traj5, 9, 12, 20
- Splits saved to: `train/vint_train/data/data_splits/custom_dataset/train/traj_names.txt` and `test/traj_names.txt`

---

## 4. New Utility Scripts

### `verify_env.py`
Runs 16 import checks covering every critical dependency. Run to confirm the environment is set up correctly:
```powershell
conda run -n nomad_train python verify_env.py
```
Expected output: `ALL 16/16 CHECKS PASSED`

### `dryrun.py`
End-to-end smoke test — loads dataset, builds model, runs 5 forward+backward passes, saves a checkpoint. Completes in ~2 minutes on CPU:
```powershell
conda run -n nomad_train python dryrun.py
```
Expected output: `SMOKE TEST PASSED`

### `check_dataset.py`
Validates all trajectories in `datasets/` — checks image count matches odometry length, verifies `position` and `yaw` keys exist in each `traj_data.pkl`:
```powershell
conda run -n nomad_train python check_dataset.py
```

### `compute_spacing.py`
Computes average distance between waypoints from odometry data — used to determine `metric_waypoint_spacing` for `data_config.yaml`.

### `aws_setup.sh`
One-shot bash script to set up the complete environment on an AWS Ubuntu instance:
- Installs system packages
- Creates conda environment
- Installs PyTorch with CUDA 11.8
- Pins `huggingface_hub==0.12.1` (required for `diffusers==0.11.1` compatibility)
- Clones and installs `diffusion_policy`
- Installs `vint_train`
- Runs a post-install verification

---

## 5. Dependency Notes

### Packages installed in `nomad_train` conda env

| Package | Version | Note |
|---|---|---|
| Python | 3.8.5 | |
| torch | 2.0.1+cpu | CPU-only on Windows; CUDA version on AWS |
| torchvision | 0.15.2+cpu | |
| diffusers | 0.11.1 | Pinned — NoMaD uses old API |
| huggingface_hub | 0.12.1 | **Pinned** — newer versions removed `cached_download` which diffusers 0.11.1 needs |
| efficientnet_pytorch | 0.7.1 | Vision backbone |
| warmup_scheduler | 0.3.2 | GradualWarmupScheduler |
| lmdb | 1.4.1 | Pre-built binary — avoids C++ compiler requirement on Windows |
| diffusion_policy | from GitHub | Cloned to `c:/Users/aryan/nomad/diffusion_policy` |

### Skipped packages (Linux-only)
- `rosbag` — only needed to process `.bag` files into dataset format
- `roslz4` — same

---

## 6. How to Run Training

### Local (Windows, CPU — for testing only)
```powershell
conda run -n nomad_train python "c:\Users\aryan\nomad\visualnav-transformer\train\train.py" -c "c:\Users\aryan\nomad\visualnav-transformer\train\config\nomad_finetune.yaml"
```

### AWS (Linux, GPU — actual fine-tuning)
```bash
# 1. One-time setup
bash aws_setup.sh

# 2. Activate environment
conda activate nomad_train

# 3. (Optional) Load pretrained weights — download nomad.pth from Google Drive
# https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg
# Place at: train/logs/nomad_pretrained/nomad/latest.pth
# Then uncomment `load_run: nomad_pretrained/nomad` in nomad_finetune.yaml

# 4. Train
cd train
python train.py -c config/nomad_finetune.yaml
```

Checkpoints are saved to `train/logs/nomad_finetune/<run_name>/` after each epoch as `latest.pth` and `<epoch>.pth`.
