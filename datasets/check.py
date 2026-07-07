import pickle
from pathlib import Path

root = Path(".")

print(
    f"{'Trajectory':<10} {'Images':<8} {'Position Shape':<20} {'Yaw Shape':<15}"
)

for traj_dir in sorted(root.glob("traj*")):

    image_count = len(list(traj_dir.glob("*.jpg")))

    pkl_file = traj_dir / "traj_data.pkl"

    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        position = data["position"]
        yaw = data["yaw"]

        print(
            f"{traj_dir.name:<10} "
            f"{image_count:<8} "
            f"{str(position.shape):<20} "
            f"{str(yaw.shape):<15}"
        )

    except Exception as e:
        print(f"{traj_dir.name:<10} ERROR: {e}")
