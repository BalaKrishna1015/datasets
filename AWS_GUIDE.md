# Working on an AWS Instance — Complete Beginner Guide

---

## What is an AWS Instance?

An AWS EC2 instance is basically a **remote computer** (running Linux/Ubuntu) sitting in a data center.  
It has a powerful GPU (like NVIDIA A100, V100, or T4) that you use for training.

You **do not** get a screen or a desktop like Windows. You work through a terminal.  
Think of it as: **you type commands → they run on that remote computer**.

---

## How to Connect to It

Your company will give you:
- An **IP address** (e.g., `54.123.45.67`)
- A **`.pem` key file** (e.g., `mykey.pem`) — this is like a password file
- A **username** (usually `ubuntu` or `ec2-user`)

### Step 1 — Connect via SSH (from your Windows terminal)

```powershell
ssh -i "C:\path\to\mykey.pem" ubuntu@54.123.45.67
```

First time connecting you'll see:
```
Are you sure you want to continue connecting? (yes/no)
```
Type `yes` and press Enter.

You are now **inside the AWS machine**. Everything you type now runs there, not on your laptop.

---

## What It Looks Like

It is just a terminal — no VS Code, no desktop, no mouse. Pure text:

```
ubuntu@ip-172-31-12-34:~$
```

That `$` is the prompt. You type commands after it.

---

## Option A — Work Purely in the Terminal (Basic)

This is the simplest way. You use command-line text editors.

### Nano (easiest editor)
```bash
nano train/config/nomad_finetune.yaml    # open a file
# edit with arrow keys
# Ctrl+O → save    Ctrl+X → exit
```

### Vim (more powerful but steeper learning curve)
```bash
vim train/config/nomad_finetune.yaml
# Press i to start editing
# Press Esc, then :wq to save and quit
# Press Esc, then :q! to quit without saving
```

---

## Option B — VS Code Remote SSH (Recommended — feels just like VS Code)

This lets you use **VS Code on your laptop** but everything actually runs on AWS.

### Setup (one time)
1. Open VS Code on your laptop
2. Install the extension: **Remote - SSH** (search in Extensions panel)
3. Press `Ctrl+Shift+P` → type `Remote-SSH: Connect to Host`
4. Enter: `ubuntu@54.123.45.67`
5. When asked for key, point it to your `.pem` file

You now have **full VS Code** — file explorer, editor, terminal — all running on AWS.  
You edit files on AWS exactly like you edit files locally.

### Configure the SSH key (so you don't have to enter it every time)
Create/edit the file `C:\Users\aryan\.ssh\config`:
```
Host aws-gpu
    HostName 54.123.45.67
    User ubuntu
    IdentityFile C:\path\to\mykey.pem
```
Now you can just connect with: `ssh aws-gpu`

---

## How to Get Your Code onto AWS

### Option 1 — Git (Recommended)
If your code is on GitHub/GitLab:
```bash
git clone https://github.com/yourusername/visualnav-transformer.git
cd visualnav-transformer
```

### Option 2 — SCP (copy files from your laptop to AWS)
Run this on your **Windows laptop** (not on AWS):
```powershell
# Copy entire repo folder to AWS
scp -i "C:\path\to\mykey.pem" -r "C:\Users\aryan\nomad\visualnav-transformer" ubuntu@54.123.45.67:~/

# Copy just the dataset
scp -i "C:\path\to\mykey.pem" -r "C:\Users\aryan\nomad\visualnav-transformer\datasets" ubuntu@54.123.45.67:~/visualnav-transformer/
```

### Option 3 — VS Code Remote SSH drag & drop
If you're using VS Code Remote SSH, just drag and drop files into the VS Code file explorer.

---

## Running the Training on AWS

Once connected and code is uploaded:

```bash
# 1. Setup environment (first time only — takes ~10 min)
cd ~/visualnav-transformer
bash aws_setup.sh

# 2. Activate the conda environment
conda activate nomad_train

# 3. Start training
cd train
python train.py -c config/nomad_finetune.yaml
```

You'll see:
```
Start ViNT DP Training Epoch 0/29

Train Batch:  14%|████      | 10/70 [00:04<00:25, loss=0.87]
Train Batch:  28%|████████  | 20/70 [00:08<00:20, loss=0.81]
...
Saved model to logs/nomad_finetune/.../latest.pth
```

With a GPU, each epoch takes **~2 minutes** instead of ~2 hours on CPU.

---

## Keeping Training Running After You Close Your Laptop

If you close your laptop / disconnect SSH, the training **will stop**.  
Use `tmux` to keep it running even after disconnecting:

```bash
# Start a tmux session
tmux new -s training

# Inside tmux, run your training
conda activate nomad_train
cd ~/visualnav-transformer/train
python train.py -c config/nomad_finetune.yaml

# Detach from tmux (training keeps running in background)
# Press: Ctrl+B, then D

# Later, reconnect to see the training output
tmux attach -t training
```

---

## Monitoring Training Progress

### Watch loss in real time
The progress bar shows loss on screen while training runs.

### Check saved checkpoints
```bash
ls -lh ~/visualnav-transformer/train/logs/nomad_finetune/
# You'll see: latest.pth, 0.pth, 1.pth, 2.pth ...  (one per epoch)
```

### Check GPU usage
```bash
watch -n 1 nvidia-smi     # updates every 1 second
```
You want **GPU Util** close to 100% — means it's using the GPU fully.

---

## Downloading Results (Checkpoints) to Your Laptop

Run this on your **Windows laptop**:
```powershell
# Download the trained model weights
scp -i "C:\path\to\mykey.pem" ubuntu@54.123.45.67:~/visualnav-transformer/train/logs/nomad_finetune/nomad_finetune_*/latest.pth "C:\Users\aryan\nomad\visualnav-transformer\train\logs\"
```

---

## Quick Reference — Most Used Commands on AWS

| What | Command |
|---|---|
| List files | `ls -la` |
| Change directory | `cd foldername` |
| Go back one folder | `cd ..` |
| Show current directory | `pwd` |
| Read a file | `cat filename.py` |
| Edit a file | `nano filename.py` |
| Check disk space | `df -h` |
| Check RAM/CPU | `htop` |
| Check GPU | `nvidia-smi` |
| See running processes | `ps aux` |
| Kill a process | `kill <PID>` |
| Create a folder | `mkdir foldername` |
| Copy a file | `cp source dest` |
| Move/rename a file | `mv source dest` |
| Delete a file | `rm filename` |

---

## Summary — Workflow for Your Project

```
Your Laptop (Windows)          AWS Instance (Ubuntu + GPU)
─────────────────────          ───────────────────────────
1. Prepare code ✓              
2. Prepare dataset ✓           
3. Test locally ✓              
                     ──SSH──►  4. Upload code + dataset
                               5. Run: bash aws_setup.sh
                               6. Run: python train.py -c config/nomad_finetune.yaml
                               7. Training runs for ~1 hour
                     ◄──SCP──  8. Download latest.pth checkpoint
9. Use checkpoint for
   deployment/inference
```
