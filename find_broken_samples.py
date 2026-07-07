"""
Find any None-returning or broken samples in the ViNT_Dataset.
"""
import sys, yaml, torch
sys.path.insert(0, r'c:\Users\aryan\nomad\visualnav-transformer\train')

with open(r'c:\Users\aryan\nomad\visualnav-transformer\train\config\defaults.yaml') as f:
    config = yaml.safe_load(f)
with open(r'c:\Users\aryan\nomad\visualnav-transformer\train\config\nomad_finetune.yaml') as f:
    config.update(yaml.safe_load(f))

from vint_train.data.vint_dataset import ViNT_Dataset

ds_cfg = list(config['datasets'].values())[0]
ds = ViNT_Dataset(
    data_folder         = ds_cfg['data_folder'],
    data_split_folder   = ds_cfg['train'],
    dataset_name        = 'custom_dataset',
    image_size          = config['image_size'],
    waypoint_spacing    = ds_cfg.get('waypoint_spacing', 1),
    min_dist_cat        = config['distance']['min_dist_cat'],
    max_dist_cat        = config['distance']['max_dist_cat'],
    min_action_distance = config['action']['min_dist_cat'],
    max_action_distance = config['action']['max_dist_cat'],
    negative_mining     = ds_cfg.get('negative_mining', True),
    len_traj_pred       = config['len_traj_pred'],
    learn_angle         = config['learn_angle'],
    context_size        = config['context_size'],
    context_type        = config['context_type'],
    end_slack           = ds_cfg.get('end_slack', 0),
    goals_per_obs       = ds_cfg.get('goals_per_obs', 1),
    normalize           = config['normalize'],
    goal_type           = config['goal_type'],
)

print(f"Dataset size: {len(ds)}")
print("Scanning all samples for None values...")

broken = []
for i in range(len(ds)):
    try:
        sample = ds[i]
        if sample is None:
            broken.append((i, "sample is None"))
            continue
        for k, v in sample.items():
            if v is None:
                broken.append((i, f"key '{k}' is None"))
                break
    except Exception as e:
        broken.append((i, str(e)))

if broken:
    print(f"\nFound {len(broken)} broken samples:")
    for idx, reason in broken[:20]:
        print(f"  idx {idx}: {reason}")
else:
    print("All samples OK!")
