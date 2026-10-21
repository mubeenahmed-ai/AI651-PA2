# AI651 PA2 — Question 2: Motion or Memory?

ConvLSTM next-frame forecasting on Moving MNIST: baseline, mirror/rotation consistency tests,
speed tests, data augmentation (models B–D) and symmetrised forecasters.

## Repository contents

Folders are added as the notebooks are run.

```
pa2/            all code (data, models, training, evaluation, figures, tables)
notebooks/      the Colab notebooks (run in order 1 -> 5), committed with their outputs once run
predictions/    Q2.5 and Q2.11 predictions, each committed before the corresponding experiments
figures/        Figures 1-6 (PNG)
gifs/           animated GIFs for Figures 3 and 6(a)
tables/         Tables 1-6 (.md and .csv)
results/        cached per-video metrics, chosen example videos, model-D choice, Figure 1 stats
```

Checkpoints and the dataset are not in the repository (they live on Google Drive); every
notebook re-downloads the data and retrains or reloads models as needed.

---

Question 1 is a written design proposal and needs no code. The optional companion notebook for Q1 runs on CPU as provided.

## Files

- `pa2_code.zip` contains the `pa2/` package (all the scripts):
  - `config.py`: protocol constants, the model registry (A and the D candidates) and the training regimes.
  - `data.py`: loading, the manual split, windows (eq. 1–2), mirror/rotation, the group G and per-epoch sampling plans.
  - `models.py`: the ConvLSTM forecaster (exact handout equations), Blank, Copy-last and the symmetrised wrapper (eq. 7).
  - `train.py`: the resumable training loop with Drive checkpoints and per-epoch timing.
  - `evaluation.py`: MSE_h, E(5), E(2), C_T, E_T and inference timing, cached per model and seed.
  - `checks.py`: Table 1 and Figure 1 (Farneback flow).
  - `visuals.py`: Figures 2–6 and the GIFs.
  - `tables.py`: Tables 2–6, formatted as "mean (seed 0, seed 1)" to three significant figures.
  - `pipeline.py` and `utils.py`: glue code, atomic Drive-safe saving and number formatting.
- There are five notebooks, run in this order:
  1. `PA2_Q2_1_setup.ipynb`: data, Table 1, model A parameter check, Figure 1, and a smoke test with a run-time estimate.
  2. `PA2_Q2_2_baseline.ipynb`: trains A (seeds 0 and 1), then Table 2 and Figure 2.
  3. `PA2_Q2_3_testing_baseline.ipynb`: Tables 3–4, Figures 3–5 and the Figure 3 GIFs.
  4. `PA2_Q2_4_what_data_can_fix.ipynb`: model D selection on validation, trains B, C and D, then Table 5.
  5. `PA2_Q2_5_final_model.ipynb`: symmetrised models, optional D+B, Table 6, Figure 6 with GIFs, and the export zip.

## First run

1. Upload the five notebooks to Colab: *File → Upload notebook*, or put them in Drive and open them from there.
2. Select *Runtime → Change runtime type → T4 GPU*.
3. Run the first cell of notebook 1. It mounts Drive and asks you to upload `pa2_code.zip` once. The code is unpacked to `MyDrive/AI651_PA2/code/`, and every later notebook imports it from there.
4. The first data load downloads Moving MNIST (~800 MB) into `MyDrive/AI651_PA2/data/`. Later sessions reuse it.

If you edit a script, set `UPDATE_CODE = True` in the first cell and upload the new zip, or upload just the changed `.py` file.

## Drive layout (`MyDrive/AI651_PA2/`)

```
code/pa2/                 the scripts
data/MovingMNIST/         dataset cache
checkpoints/<run>/seed<s>/ckpt.pt, final.pt, log.json   (pilot runs under checkpoints/pilot/)
results/metrics/          cached per-video metrics (.npz) and inference times
results/selection.json    videos chosen for Figures 2, 3, 5 (Figure 6 reuses them)
results/d_choice.json     the model D design you chose
figures/  gifs/  tables/  report material (tables as .md and .csv)
```

## Disconnects and resuming

Reconnect to a GPU runtime and run the notebook again from the top.

- **Finished runs** (`log.json` with `"done": true`) are loaded, not retrained.
- **Interrupted runs** continue from `ckpt.pt`. It is written at the end of every epoch and every 5 minutes inside an epoch. The batch order of an epoch depends only on (seed, epoch), so a resumed run is identical to an uninterrupted one.
- **Metrics** that are already computed are read from `results/metrics/`.

## Rules the notebooks enforce or help with

- **Splits:** videos are split by dataset index (0–7,999 train, 8,000–8,999 val, 9,000–9,999 test). The built-in `split=` option is never used.
- **Q2.5 gate:** notebook 3 starts only after you set `Q2_5_COMMITTED = True`. Commit your Q2.5 predictions to GitHub first.
- **Q2.11 gate:** notebook 4 starts only after you set `Q2_11_COMMITTED = True`. Commit your Q2.11 predictions first.
- **No test leakage:** notebook 2 only evaluates at 1× on the original clips. Mirror, rotation and speed experiments start in notebook 3.
- **Model D selection:** pilots are scored on the validation split only. The candidates in `config.ARCHS` are all ≤ 721,176 parameters. To try your own design, add an entry in notebook 4.
- **Hardware consistency:** keep every timed training run on the same GPU type. Notebook 4 warns if the logs show mixed hardware.

## Definitions worth stating in your report

- **Training time per epoch** is the mean over 30 epochs of the time spent in training batches: data preparation, forward, backward and optimiser step. Checkpoint writes to Drive and the validation monitor (every 5 epochs) are excluded.
- **Inference time** covers all 1,000 original (1×, t = 1) test videos at batch size 32, under `eval()` and `no_grad()`. It is counted from when each batch is on the GPU, after one untimed warm-up batch. For symmetrised models it includes all 8 transformations, forward passes, inverse transformations and the averaging.
- **Overlap detection** for picking figure videos: digits translate rigidly and are combined with a pixel-wise max. A drop in the per-frame count of bright pixels therefore marks overlap.

## Left to you

- **Reach:** the reach estimates for Q2.3 and Q2.12.
- **Pre-registered predictions:** Q2.5 and Q2.11, committed before the corresponding notebooks run.
- **Report:** all written answers, and the AI-use disclosure (prompts, outputs and your edits) at the end of the report.

## For the GitHub repository

Notebook 5's last cell downloads `pa2_outputs.zip` containing `figures/`, `gifs/`, `tables/`, `results/` and `pa2/`. Add the notebooks too. The GIF paths printed in notebooks 3 and 5 are relative to this layout (for example `gifs/fig3_M_rank1_test123.gif`), so keep the folder names when you copy them in.
