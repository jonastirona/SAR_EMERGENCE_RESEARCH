## Quick context for automated coding agents

This repo implements LSTM (and a transformer mentioned in README) pipelines for Solar Active Region (SAR) emergence prediction. Focus your edits on the `lstm/` package first: it contains data loaders, model definitions, training loops, Ray Tune grid search, and evaluation plotting.

Key files
- `lstm/functions.py` — core helpers: data loading (`load_ar_data`), preprocessing (`process_data`, `prepare_dataset`), model classes (`LSTM`, `VanillaLSTM`), training/validation loops, and filename parsing (`get_params`).
- `lstm/train.py` — single-experiment runner (expects positional args; builds DataLoaders and saves model `.pth`).
- `lstm/grid_search.py` — Ray Tune tuner + wandb integration for hyperparameter search (uses `tune.Tuner` and `OptunaSearch`).
- `lstm/eval.py` — evaluation + plotting helpers; expects saved model files and handles `module.` prefix in state dicts.
- `data/ARxxxxx/` — per-active-region `.npz` files; `load_ar_data` expects `mean_pmdop{AR}_flat.npz`, `mean_mag{AR}_flat.npz`, `mean_int{AR}_flat.npz` and pulls arrays named `arr_0..arr_4`.

How the system is wired (big-picture)
- Data -> `load_ar_data` (per AR `.npz`) -> `process_data` (min-max scaling) -> `prepare_dataset` (builds sequences across tiles) -> DataLoader -> `train_epoch` / `train_epochHybrid` -> model saved to `lstm/results/`.
- Eval loads saved `.pth` files with `get_params` (parses hyperparameters from filename using regex) and `initialize_lstm` (strips `module.` prefix when needed).
- Hyperparameter search uses Ray Tune + ASHAScheduler and reports `RMSE` to early stopping logic.

Project-specific conventions and pitfalls (important)
- File-name encoded metadata: LSTM models saved as `pred{num_pred}_r{rid}_i{num_in}_n{num_layers}_h{hidden}_e{epochs}_lr{lr}_d{dropout}.pth`. The regex in `get_params` expects that exact pattern — avoid changing it unless updating `get_params`.
- `isVanillaLSTM` global in `functions.py` toggles which model class is imported elsewhere. Changing it affects `train.py`, `grid_search.py`, and `eval.py` behaviour.
- `BASE_PATH`, `DATA_PATH`, `RESULTS_PATH` are computed relative to cwd in `functions.py`. Tests/agents should run from repository root or adjust `BASE_PATH` accordingly.
- `prepare_dataset` returns scaling tuples (m_scale, flux_scale, cont_int_scale) — downstream scripts reuse them to avoid inconsistent normalization.
- Missing `.npz` files: `load_ar_data` prints a warning and returns None; callers check for None and exit. Add unit tests for that branch if you change data-loading behavior.

Common commands and examples
- Install deps (same on Linux/Windows if python is on PATH):
  - pip install -r requirements.txt
- Run a single LSTM training (example matching `train.py` signature):
  - python lstm/train.py 12 4 110 4 10 5 0.00093 0.0 32
    (args: num_pred, rid_of_top, num_in, num_layers, hidden_size, n_epochs, learning_rate, dropout, batch_size)
- Run the Ray Tune hyperparameter search (from repo root):
  - python lstm/grid_search.py 50
    (argument = sample_size used by the script)
- Evaluate a trained model for an AR (examples inside `lstm/eval.py`):
  - python lstm/eval.py

Integration points
- Weights & Biases: scripts call `wandb.init()` and log artifacts. The README suggests using a `.env` with WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT — set those before running hyperparameter search.
- Ray Tune: `grid_search.py` requires Ray and GPU/CPU resources; it's configured to `ray.init(num_cpus=4, num_gpus=2)` — adapt to CI or local machine.
- Torch: models are saved as state_dicts (sometimes wrapped with `module.` when saved from DataParallel). `eval` strips this prefix.

What to change cautiously
- Model filename regex in `get_params` and any filename construction. Many scripts rely on that exact pattern.
- `isVanillaLSTM` flag and the I/O shapes used by `VanillaLSTM` vs the decoder-based `LSTM` — they expect different forward signatures (VanillaLSTM.forward(x) vs LSTM.forward(x, y=None,...)).

Quick-to-check edge cases for PRs
- Run data loading for one AR to ensure `load_ar_data` finds arrays named `arr_*` in `.npz` files.
- Run `python -c "import torch; print(torch.cuda.is_available())"` in CI or local to confirm GPU assumptions before landing GPU-only changes.
- When changing scaling or sequence lengths, re-run `lstm/eval.py` on one AR and verify the plotting scripts don't error on array shape mismatches.

Files to link in PR descriptions
- Always reference `lstm/functions.py` and `lstm/eval.py` when changing data shapes, and `lstm/grid_search.py` when altering hyperparameter search behavior.

If anything here is unclear or you want me to expand examples (e.g., unit tests for dataset loading, or a simplified training harness), tell me which area to expand.
