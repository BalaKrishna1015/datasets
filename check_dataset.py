import pickle, numpy as np, os

dataset_root = r'c:\Users\aryan\nomad\visualnav-transformer\datasets'
trajs = sorted(os.listdir(dataset_root))
print(f'Total trajectories: {len(trajs)}')
print()

errors = []
for traj in trajs:
    traj_path = os.path.join(dataset_root, traj)
    pkl_path = os.path.join(traj_path, 'traj_data.pkl')
    imgs = sorted([f for f in os.listdir(traj_path) if f.endswith('.jpg')],
                  key=lambda x: int(x.replace('.jpg', '')))

    if not os.path.exists(pkl_path):
        errors.append(f'{traj}: MISSING traj_data.pkl')
        print(f'{traj:8s}: MISSING traj_data.pkl')
        continue

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    keys = list(data.keys())
    has_pos = 'position' in data
    has_yaw = 'yaw' in data
    pos_shape = data['position'].shape if has_pos else None
    yaw_shape = data['yaw'].shape if has_yaw else None

    # Check image count matches odometry length
    T = len(imgs)
    pos_T = pos_shape[0] if pos_shape else None
    count_match = (T == pos_T) if pos_T else False

    status = 'OK' if (has_pos and has_yaw and count_match) else 'ISSUE'
    if not count_match:
        note = f' !! imgs({T}) != position({pos_T})'
    else:
        note = ''

    print(f'{traj:8s}: {T:3d} imgs | position{pos_shape} | yaw{yaw_shape} | [{status}]{note}')

    if not (has_pos and has_yaw):
        errors.append(f'{traj}: missing keys, found {keys}')
    if not count_match:
        errors.append(f'{traj}: image count ({T}) != position length ({pos_T})')

print()
if errors:
    print('ISSUES FOUND:')
    for e in errors:
        print(f'  {e}')
else:
    print('Dataset looks good — all trajectories valid!')
    print()
    total_imgs = sum(
        len([f for f in os.listdir(os.path.join(dataset_root, t)) if f.endswith('.jpg')])
        for t in trajs
    )
    print(f'Total images across all trajectories: {total_imgs}')
