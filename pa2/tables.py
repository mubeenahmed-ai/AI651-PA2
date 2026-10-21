"""Tables 2-6. Learned models: "mean (seed 0, seed 1)"; Blank / Copy-last: one value."""
from .utils import fmt_int, fmt_seeds, save_table

NAMES = {"A": "A", "B": "B", "C": "C", "D": "D", "DB": "D+B", "blank": "Blank",
         "copylast": "Copy-last", "symB": "sym. B", "symD": "sym. D", "symDB": "sym. D+B"}


def _col(per_seed, key):
    return fmt_seeds([s[key] for s in per_seed])


def table2(ctx):
    header = ["Model", "MSE1", "MSE2", "MSE3", "MSE4", "MSE5", "E(5)", "Params"]
    rows = []
    for key in ("A", "blank", "copylast"):
        ps = ctx.per_seed(key, ("std",))
        rows.append([NAMES[key]] + [_col(ps, f"MSE{h}") for h in range(1, 6)] +
                    [_col(ps, "E5_1x"), fmt_int([ctx.param_count(key)])])
    return save_table(ctx.paths, "table2_baseline", header, rows,
                      note="Test split, 1×, t = 1. Learned model: mean (seed 0, seed 1).")


def table3(ctx):
    header = ["Model", "E(5)", "C_M", "E_M", "C_R", "E_R"]
    rows = []
    for key in ("A", "blank", "copylast"):
        ps = ctx.per_seed(key, ("std", "tr"))
        rows.append([NAMES[key]] + [_col(ps, k) for k in ("E5_1x", "C_M", "E_M", "C_R", "E_R")])
    return save_table(ctx.paths, "table3_mirror_rotation", header, rows,
                      note="Test split, 1×, t = 1, averaged over the five horizons.")


def table4(ctx):
    header = ["Model", "E(2) 1×", "E(2) 2×", "E(2) 3×", "E(5) 1×", "E(5) 2×"]
    rows = []
    for key in ("A", "copylast"):
        ps = ctx.per_seed(key, ("std", "speed"))
        rows.append([NAMES[key]] + [_col(ps, k) for k in ("E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x")])
    return save_table(ctx.paths, "table4_speed", header, rows,
                      note="Test split, t = 1. Only E(2) is comparable across speeds.")


def _big_rows(ctx, keys):
    rows = []
    for key in keys:
        ps = ctx.per_seed(key, ("std", "tr", "speed", "time"))
        rows.append([NAMES[key]] +
                    [_col(ps, k) for k in ("E5_1x", "C_M", "E_M", "C_R", "E_R",
                                          "E2_1x", "E2_2x", "E2_3x")] +
                    [fmt_int([ctx.param_count(key)]),
                     fmt_seeds(ctx.train_time(key)),
                     _col(ps, "inference_s")])
    return rows


BIG_HEADER = ["Model", "E(5) 1×", "C_M", "E_M", "C_R", "E_R", "E(2) 1×", "E(2) 2×", "E(2) 3×",
              "Params", "Train s/epoch", "Inference s"]


def table5(ctx, hardware):
    rows = _big_rows(ctx, ("A", "B", "C", "D"))
    note = (f"Test split. Model D = {ctx.d_arch}. Timings on {hardware}; training time = mean "
            "wall-clock per epoch over 30 epochs (training batches only); inference = 1,000 "
            "original test videos at batch 32.")
    return save_table(ctx.paths, "table5_data_fix", BIG_HEADER, rows, note=note)


def table6(ctx, keys, hardware):
    rows = _big_rows(ctx, keys)
    note = (f"Test split. Symmetrised models (eq. 7). Training time is that of the underlying "
            f"learned model. Timings on {hardware}. Inference includes all 8 transformations, "
            "forward passes, inverse transformations and the averaging.")
    return save_table(ctx.paths, "table6_final", BIG_HEADER, rows, note=note)
