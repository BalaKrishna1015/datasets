import pickle, numpy as np, os

dataset_root = r'c:\Users\aryan\nomad\visualnav-transformer\datasets'
all_spacings = []

for traj in sorted(os.listdir(dataset_root)):
    pkl = os.path.join(dataset_root, traj, 'traj_data.pkl')
    with open(pkl, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    pos = data['position']
    dists = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    all_spacings.extend(dists.tolist())

arr = np.array(all_spacings)
print(f'Mean spacing   : {arr.mean():.4f} m')
print(f'Median spacing : {np.median(arr):.4f} m')
print(f'Std            : {arr.std():.4f} m')
print(f'Min / Max      : {arr.min():.4f} / {arr.max():.4f} m')
