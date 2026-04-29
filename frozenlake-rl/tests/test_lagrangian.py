import os
import numpy as np


src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(src_path)

lagrange_stats = os.path.join(project_path, "training_stats", f"lagrangian_ql_FrozenLake_easy.npz")

lagrange = np.load(lagrange_stats)

for key in lagrange.files:
    print(f"{key}: {lagrange[key].shape}")