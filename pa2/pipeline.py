"""High-level glue used by the notebooks.

Model keys
----------
    "blank", "copylast"          deterministic reference predictors (no seed)
    "A", "B", "C", "D", "DB"     learned models of the model table (seeds 0, 1)
    "symB", "symD", "symDB"      symmetrised wrappers of B / D / D+B (eq. 7)
"""
import os

import torch

from . import config
from .data import load_moving_mnist, split_videos
from .data import eval_windows
from .evaluation import get_metrics, horizon_mse, predict, summarise
from .models import Blank, CopyLast, Symmetrised, build_arch, check_arch_budget, n_params
from .train import train_run
from .utils import atomic_json_save, get_device, hardware_string, load_json


class Context:
    def __init__(self, project_root, d_arch=None, log=print):
        self.paths = config.project_paths(project_root)
        self.device = get_device()
        self.log = log
        self.d_arch = d_arch
        self.data = None
        self._gpu_cache = {}
        log(f"Device: {hardware_string(self.device)}")
        if self.device.type != "cuda":
            log("WARNING: no GPU. Runtime > Change runtime type > T4 GPU before training.")

    # ---- data -----------------------------------------------------------
    def load_data(self):
        if self.data is None:
            self.log("Loading Moving MNIST (downloaded to Drive the first time) ...")
            self.data = split_videos(load_moving_mnist(self.paths["data"]))
            self.log({k: tuple(v.shape) for k, v in self.data.items()})
        return self.data

    def on_device(self, split):
        """uint8 split on the GPU (train: ~650 MB) for fast batching."""
        if split not in self._gpu_cache:
            self._gpu_cache[split] = self.load_data()[split].to(self.device)
        return self._gpu_cache[split]

    @property
    def test(self):
        return self.load_data()["test"]

    # ---- model D choice ----------------------------------------------------
    def set_d_arch(self, arch):
        check_arch_budget(arch, is_model_d=True)
        self.d_arch = arch
        from .utils import atomic_json_save
        atomic_json_save({"d_arch": arch, "params": n_params(build_arch(arch)),
                          "config": config.ARCHS[arch]},
                         os.path.join(self.paths["results"], "d_choice.json"))
        self.log(f"Model D architecture: {arch} {config.ARCHS[arch]} "
                 f"({n_params(build_arch(arch)):,} params)")

    def load_d_arch(self):
        rec = load_json(os.path.join(self.paths["results"], "d_choice.json"))
        if rec is None:
            raise RuntimeError("Model D not chosen yet: run notebook 4 first.")
        config.ARCHS.setdefault(rec["d_arch"], rec["config"])   # custom designs defined in a notebook
        self.d_arch = rec["d_arch"]
        return self.d_arch

    def regime(self, key):
        r = dict(config.REGIMES[key])
        if r["arch"] is None:
            if self.d_arch is None:
                self.load_d_arch()
            r["arch"] = self.d_arch
        return r

    def run_name(self, key):
        r = self.regime(key)
        return key if r["arch"] == "A" else f"{key}-{r['arch']}"

    def run_dir(self, key, seed):
        return os.path.join(self.paths["ckpt"], self.run_name(key), f"seed{seed}")

    # ---- training -----------------------------------------------------------
    def train(self, key, seed, **kw):
        r = self.regime(key)
        check_arch_budget(r["arch"], is_model_d=key in ("D", "DB"))
        return train_run(self.run_dir(key, seed), r["arch"], r["speed_aug"], r["sym_aug"], seed,
                         self.on_device("train"), self.device, val_videos=self.on_device("val"),
                         log=self.log, **kw)

    def train_log(self, key, seed):
        info = load_json(os.path.join(self.run_dir(key, seed), "log.json"))
        if not info or not info.get("done"):
            raise RuntimeError(f"{key} seed {seed} is not trained yet")
        return info

    def load_trained(self, key, seed):
        r = self.regime(key)
        path = os.path.join(self.run_dir(key, seed), "final.pt")
        if not os.path.exists(path):
            raise RuntimeError(f"{key} seed {seed} is not trained yet ({path})")
        m = build_arch(r["arch"]).to(self.device)
        m.load_state_dict(torch.load(path, map_location=self.device))
        return m.eval()

    # ---- models by key ---------------------------------------------------------
    def model(self, key, seed=None):
        if key == "blank":
            return Blank().to(self.device)
        if key == "copylast":
            return CopyLast().to(self.device)
        if key.startswith("sym"):
            return Symmetrised(self.load_trained(key[3:], seed)).to(self.device).eval()
        return self.load_trained(key, seed)

    def cache_key(self, key):
        if key in ("blank", "copylast"):
            return key
        if key.startswith("sym"):
            return "sym" + self.run_name(key[3:])
        return self.run_name(key)

    @staticmethod
    def seeds(key):
        return [None] if key in ("blank", "copylast") else list(config.SEEDS)

    def metrics(self, key, seed, stage, recompute=False):
        return get_metrics(self.paths, self.cache_key(key), seed, stage,
                           lambda: self.model(key, seed), self.test, self.device, recompute)

    def summary(self, key, seed, stages=("std",)):
        got = {s: self.metrics(key, seed, s) for s in stages if s != "time"}
        s = summarise(got.get("std"), got.get("tr"), got.get("speed"))
        if "time" in stages:
            s["inference_s"] = self.metrics(key, seed, "time")["inference_s"]
        return s

    def per_seed(self, key, stages):
        return [self.summary(key, s, stages) for s in self.seeds(key)]

    def param_count(self, key):
        if key in ("blank", "copylast"):
            return 0
        base = key[3:] if key.startswith("sym") else key
        return n_params(build_arch(self.regime(base)["arch"]))

    def train_time(self, key):
        """Mean per-epoch training time of the underlying learned model, per seed."""
        base = key[3:] if key.startswith("sym") else key
        return [self.train_log(base, s)["mean_epoch_time_s"] for s in config.SEEDS]

    # ---- model-D design selection on the VALIDATION split ----------------------
    def pilot(self, arch, epochs=10, seed=0):
        """Short speed-augmented run of a candidate D design, scored on validation only."""
        n = check_arch_budget(arch, is_model_d=True)
        rd = os.path.join(self.paths["ckpt"], "pilot", f"{arch}_e{epochs}", f"seed{seed}")
        res_path = os.path.join(rd, "val_metrics.json")
        rec = load_json(res_path)
        if rec is not None:
            return rec
        model, info = train_run(rd, arch, True, False, seed, self.on_device("train"), self.device,
                                val_videos=self.on_device("val"), epochs=epochs, log=self.log)
        val = self.load_data()["val"]
        rec = {"arch": arch, "params": n, "epochs": epochs, "seed": seed,
               "mean_epoch_time_s": info["mean_epoch_time_s"]}
        for k in (1, 2, 3):
            X, Y, mask = eval_windows(val, k=k)
            m = horizon_mse(predict(model, X, self.device), Y, mask)
            rec[f"val_E2_{k}x"] = float(m[:, :2].mean())
            if k < 3:
                rec[f"val_E5_{k}x"] = float(m.mean())
        atomic_json_save(rec, res_path)
        return rec

    # ---- training curves -----------------------------------------------------------
    def plot_curves(self, keys, out_path=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        for key in keys:
            for s in config.SEEDS:
                try:
                    h = self.train_log(key, s)["history"]
                except RuntimeError:
                    continue
                ls = "-" if s == 0 else "--"
                ax[0].plot([r["epoch"] for r in h], [r["train_loss"] for r in h], ls, label=f"{key} s{s}")
                v = [(r["epoch"], r["val_E5_1x"]) for r in h if "val_E5_1x" in r]
                if v:
                    ax[1].plot(*zip(*v), ls + "o", ms=3, label=f"{key} s{s}")
        ax[0].set(xlabel="epoch", ylabel="train loss (masked MSE)", yscale="log", title="training loss")
        ax[1].set(xlabel="epoch", ylabel="val E(5) at 1x", yscale="log", title="validation monitor")
        for a in ax:
            a.legend(fontsize=7)
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig
