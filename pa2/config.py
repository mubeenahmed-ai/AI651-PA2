"""Constants fixed by the AI651 PA2 handout (Question 2) and project paths."""
import os

# ---- data -----------------------------------------------------------------
N_VIDEOS = 10_000
N_FRAMES = 20                      # frames are numbered 1..20 in the handout
FRAME_SHAPE = (1, 64, 64)
SPLITS = {"train": (0, 8000), "val": (8000, 9000), "test": (9000, 10000)}
N_IN, N_OUT = 5, 5
SPEEDS = (1, 2, 3)

# ---- training protocol (Training and evaluation) ----------------------------
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
SEEDS = (0, 1)

# ---- model constraints --------------------------------------------------------
MODEL_A_PARAMS = 480_784
PARAM_LIMIT_D = 721_176            # 1.5 x model A

# Architecture registry. "A" is fixed by the handout; the D_* entries are
# candidate designs for model D (all within the parameter budget, all with a
# larger one-step reach than A -- you still have to show this yourself in Q2.12).
# Add your own entry here if you want a different design.
ARCHS = {
    "A":     dict(patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 1)),
    # 5x5 kernels in both layers, hidden width reduced to stay under the budget
    "D_k5":  dict(patch=4, hidden=(46, 46), kernels=(5, 5), dilations=(1, 1)),
    # second layer dilated (dilation 2); same parameter count as A
    "D_dil": dict(patch=4, hidden=(64, 64), kernels=(3, 3), dilations=(1, 2)),
    # 8x8 patch folding (8x8 grid of 64-channel cells)
    "D_p8":  dict(patch=8, hidden=(64, 64), kernels=(3, 3), dilations=(1, 1)),
}

# Training regimes (model table in "What Can Data Fix?").
# arch=None means "use the chosen model-D architecture".
REGIMES = {
    "A":  dict(arch="A", speed_aug=False, sym_aug=False),
    "B":  dict(arch="A", speed_aug=False, sym_aug=True),
    "C":  dict(arch="A", speed_aug=True,  sym_aug=False),
    "D":  dict(arch=None, speed_aug=True,  sym_aug=False),
    "DB": dict(arch=None, speed_aug=True,  sym_aug=True),
}


def project_paths(root):
    """Create (if needed) and return the folder layout on Google Drive."""
    paths = {
        "root": root,
        "data": os.path.join(root, "data"),
        "ckpt": os.path.join(root, "checkpoints"),
        "results": os.path.join(root, "results"),
        "metrics": os.path.join(root, "results", "metrics"),
        "figures": os.path.join(root, "figures"),
        "gifs": os.path.join(root, "gifs"),
        "tables": os.path.join(root, "tables"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths
