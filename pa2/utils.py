import json
import math
import os
import platform
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Drive-safe IO: write to a temp file next to the target, then rename.
# A disconnect mid-write leaves the previous file intact.
# ---------------------------------------------------------------------------

def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz_save(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# Hardware / timing
# ---------------------------------------------------------------------------

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_string(device=None):
    device = device or get_device()
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        return f"{p.name} ({p.total_memory / 2**30:.0f} GB), torch {torch.__version__}"
    return f"CPU {platform.processor() or platform.machine()} x{os.cpu_count()}, torch {torch.__version__}"


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self.t0


# ---------------------------------------------------------------------------
# Number formatting for the report tables
#   errors to three significant figures, scientific notation below 1e-3,
#   entries for learned models as "mean (seed 0, seed 1)".
# ---------------------------------------------------------------------------

def fmt(v, sig=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    v = float(v)
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3:
        return f"{v:.{sig - 1}e}"
    e = math.floor(math.log10(abs(v)))
    dec = sig - 1 - e
    r = round(v, dec)
    if r != 0 and math.floor(math.log10(abs(r))) != e:      # rounding crossed a power of 10
        dec -= 1
        r = round(v, dec)
    return f"{r:.{dec}f}" if dec > 0 else f"{int(round(r))}"


def fmt_seeds(values, sig=3):
    """values: list of per-seed numbers (len 1 for deterministic predictors)."""
    values = [float(v) for v in values]
    if len(values) == 1:
        return fmt(values[0], sig)
    return f"{fmt(np.mean(values), sig)} ({', '.join(fmt(v, sig) for v in values)})"


def fmt_int(values):
    values = sorted({int(v) for v in values})
    return ", ".join(f"{v:,}" for v in values)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def markdown_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save_table(paths, name, header, rows, note=""):
    """Writes tables/<name>.md and tables/<name>.csv and returns the markdown."""
    import csv
    md = markdown_table(header, rows)
    if note:
        md += "\n\n" + note
    with open(os.path.join(paths["tables"], name + ".md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(paths["tables"], name + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return md


def show_table(md):
    try:
        from IPython.display import Markdown, display
        display(Markdown(md))
    except Exception:
        print(md)
