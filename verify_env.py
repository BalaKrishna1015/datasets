"""
Quick verification script — run with: python verify_env.py
Checks all imports needed for NoMaD fine-tuning.
"""

checks = []

def check(label, fn):
    try:
        fn()
        print(f"  OK   {label}")
        checks.append(True)
    except Exception as e:
        print(f"  FAIL {label}  ->  {e}")
        checks.append(False)

import sys
print(f"\nPython : {sys.version}")

import torch
print(f"torch  : {torch.__version__}")
print(f"CUDA   : {torch.cuda.is_available()}")
print()

check("diffusers 0.11.1",         lambda: __import__("diffusers"))
check("DDPMScheduler",            lambda: __import__("diffusers.schedulers.scheduling_ddpm", fromlist=["DDPMScheduler"]))
check("EMAModel",                 lambda: __import__("diffusers.training_utils", fromlist=["EMAModel"]))
check("efficientnet_pytorch",     lambda: __import__("efficientnet_pytorch"))
check("warmup_scheduler",         lambda: __import__("warmup_scheduler"))
check("lmdb",                     lambda: __import__("lmdb"))
check("opencv-python",            lambda: __import__("cv2"))
check("wandb",                    lambda: __import__("wandb"))
check("h5py",                     lambda: __import__("h5py"))
check("tqdm",                     lambda: __import__("tqdm"))
check("ConditionalUnet1D",        lambda: __import__("diffusion_policy.model.diffusion.conditional_unet1d", fromlist=["ConditionalUnet1D"]))
check("NoMaD model",              lambda: __import__("vint_train.models.nomad.nomad", fromlist=["NoMaD"]))
check("NoMaD_ViNT encoder",       lambda: __import__("vint_train.models.nomad.nomad_vint", fromlist=["NoMaD_ViNT"]))
check("train_eval_loop_nomad",    lambda: __import__("vint_train.training.train_eval_loop", fromlist=["train_eval_loop_nomad"]))
check("GNM model",                lambda: __import__("vint_train.models.gnm.gnm", fromlist=["GNM"]))
check("ViNT model",               lambda: __import__("vint_train.models.vint.vint", fromlist=["ViNT"]))

print()
passed = sum(checks)
total = len(checks)
if passed == total:
    print(f"ALL {total}/{total} CHECKS PASSED — environment is ready to train!")
else:
    print(f"{passed}/{total} checks passed. Fix the FAIL items above before training.")
    sys.exit(1)
