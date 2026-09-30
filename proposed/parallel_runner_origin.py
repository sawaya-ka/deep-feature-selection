#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4×4 環境で seed ごとにプロセスを立てて GPU に並列割り当てするスクリプト
- mode=launch : 複数GPUにシードをラウンドロビンで割当て並列実行
- mode=worker : 単一seedを処理
- mode=merge  : 各seedの結果CSVを縦に結合
- --T はMLP/CNN/RNNの更新回数。Transformerは既定でその10倍。
- --tr-T でTransformerの総更新回数を独立指定（例: 40000）。
- --models tr でTransformerだけを実行・マージ。他モデルの結果は変更しない。
- 学習は毎回初期値から。CSV/診断NPZは再開用チェックポイントではない。
- TransformerはPre-LN、W1を含む全層をAdamで更新。
- 各 *_steps.csv は実際の更新回数（Transformerも圧縮せず保存）。
"""

import os, sys, argparse, subprocess, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import shutil
import tempfile


def notify_telegram(msg: str):
    token = os.environ.get("TG_TOKEN")
    chat  = os.environ.get("TG_CHAT_ID")
    if not (token and chat):
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": int(chat), "text": msg, "disable_web_page_preview": True}
    try:
        subprocess.run(
            ["curl","-s","-X","POST",url,"-H","Content-Type: application/json","-d", json.dumps(payload)],
            check=False
        )
    except Exception:
        pass

# あなたの元コードの関数をインポート
from util_origin import (
    generate_sim_data,
    torch_nn_feature_selection_path,
    torch_cnn1d_feature_selection_path,
    torch_rnn_feature_selection_path,
    torch_transformer_feature_selection_path,
)

alpha = 0.1
TRANSFORMER_T_MULTIPLIER = 10
TRANSFORMER_ADAM_LR = 5e-4
COMPUTE_EVERY = 10
MODEL_ORDER = ("tr", "rnn", "mlp", "cnn")


def _selected_models(models=None):
    if models is None:
        return MODEL_ORDER
    models = (models,) if isinstance(models, str) else tuple(models)
    if not models or len(set(models)) != len(models) or set(models) - set(MODEL_ORDER):
        raise ValueError(f"models must be distinct names from {MODEL_ORDER}.")
    return tuple(model for model in MODEL_ORDER if model in models)


def _transformer_horizon(T, tr_T=None):
    if T < 1 or (tr_T is not None and tr_T < 1):
        raise ValueError("T and tr-T must be positive.")
    return TRANSFORMER_T_MULTIPLIER * T if tr_T is None else tr_T


def _install_outputs(staging_dir, outdir):
    """Back up existing selected files, then atomically replace each file."""
    files = sorted(Path(staging_dir).iterdir())
    existing = [outdir / path.name for path in files if (outdir / path.name).exists()]
    if existing:
        backup_dir = outdir / "backups" / str(time.time_ns())
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, backup_dir / path.name)
        print(f"[backup] {len(existing)} files -> {backup_dir}", flush=True)
    for path in files:
        os.replace(path, outdir / path.name)


def _check_existing_transformer_data(seed_dir, **current):
    """Avoid mixing a rerun on different data with retained model results."""
    saved_path = seed_dir / "tr_diagnostics.npz"
    if not saved_path.is_file():
        return
    with np.load(saved_path, allow_pickle=False) as saved:
        for key, value in current.items():
            if key not in saved or not np.array_equal(saved[key], value):
                raise ValueError(
                    f"Existing Transformer data/seeds differ at {key}: {saved_path}. "
                    "Use the original --m, --n, seed and data-generation settings, "
                    "or choose a new --outdir. No result files were replaced."
                )


def _make_transformer_optimizer(model):
    """Use Adam on all Transformer parameters, including the dense input W1."""
    import torch

    return torch.optim.Adam(
        model.parameters(), lr=TRANSFORMER_ADAM_LR,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )


def _recorded_steps(T, compute_every):
    """Match util_origin's every-k-updates plus final-update recording."""
    if T < 1 or compute_every < 1:
        raise ValueError("T and compute_every must be positive.")
    steps = list(range(compute_every, T + 1, compute_every))
    if not steps or steps[-1] != T:
        steps.append(T)
    return np.asarray(steps, dtype=int)

# ========== ワーカー（1シードだけ処理） ==========
def run_one_seed(seed:int, m:int, n:int, T:int, outdir:Path, models=None, tr_T=None):
    models = _selected_models(models)
    transformer_T = _transformer_horizon(T, tr_T)
    if seed is None:
        raise ValueError("--seed is required in worker mode.")
    outdir = Path(outdir)
    seed_dir = outdir / f"seed_{seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    streams = np.random.SeedSequence(seed).spawn(4)
    data_seed, split_seed, model_seed1, model_seed2 = [
        int(stream.generate_state(1, dtype=np.uint32)[0])
        for stream in streams
    ]

    X, y, true_idx, B = generate_sim_data(m=m, n=n, rng=seed)

    # --- Transformer ---
    if "tr" in models:
        _check_existing_transformer_data(
            seed_dir, X=X, y=y, B=B, true_idx=np.asarray(true_idx),
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
        )
        print(
            f"[Transformer config] updates={transformer_T}, norm_first=True, "
            f"all layers (including W1)=Adam(lr={TRANSFORMER_ADAM_LR:.3e}, weight_decay=0)",
            flush=True,
        )
        path_tr = torch_transformer_feature_selection_path(
            X=X, y=y, alpha=0.1, T=transformer_T,
            true_idx=true_idx,
            lift_dim=64,
            d_model=256, nhead=4, num_layers=2,
            dim_feedforward=516, dropout=0.1,
            activation="relu", init_mode="he", pool="mean",
            use_sinusoidal_pos=True, fc_dims=[64],
            batch_size=128, lr=TRANSFORMER_ADAM_LR, weight_decay=0.0,
            loss="mse", psi_method="mean",
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY, xi_batch=256,
            xi_mode="sumgrad",
            norm_first=True,
            optimizer_factory=_make_transformer_optimizer,
        )
        diagnostics = dict(
            steps=path_tr.steps,
            xi1=path_tr.xi1_hist,
            xi2=path_tr.xi2_hist,
            M=path_tr.M_hist,
            tau=path_tr.tau,
            R_plus=path_tr.R_plus,
            R_minus=path_tr.R_minus,
            FDP_hat=path_tr.FDPhat,
            FDP=path_tr.FDR_true,
            typeII=path_tr.TypeII,
            true_idx=np.asarray(true_idx),
            X=X, y=y, B=B,
            seed_split=split_seed,
            seed_model1=model_seed1,
            seed_model2=model_seed2,
            loss_steps=path_tr.loss_steps,
            loss=path_tr.loss_values,
            base_T=T,
            transformer_T=transformer_T,
            optimizer="adam",
            optimizer_scope="all_parameters_including_W1",
            adam_lr=TRANSFORMER_ADAM_LR,
            adam_betas=(0.9, 0.999),
            adam_eps=1e-8,
            weight_decay=0.0,
            norm_first=True,
        )
        _save_block(path_tr, seed_dir, prefix="tr", diagnostics=diagnostics)
        del path_tr, diagnostics; _gc_cuda()

    # --- RNN ---
    if "rnn" in models:
        path_rnn = torch_rnn_feature_selection_path(
            X=X, y=y, alpha=0.1, T=T,
            true_idx=true_idx,
            rnn_type="lstm", hidden_size=128, num_layers=2, bidirectional=True,
            input_proj_dim=4,
            fc_dims=[64], activation="relu", init_mode="xavier",
            dropout=0.1,
            batch_size=128, lr=5e-3, weight_decay=0.0,
            loss="mse", psi_method="mean",
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY, xi_batch=256,
            xi_mode="sumgrad"
        )
        _save_block(path_rnn, outdir / f"seed_{seed:04d}", prefix="rnn")
        del path_rnn; _gc_cuda()

    # --- MLP ---
    if "mlp" in models:
        path = torch_nn_feature_selection_path(
            X=X, y=y, alpha=alpha, T=T,
            true_idx=true_idx,
            hidden_dims=[1024, 1024, 512, 128],
            batch_size=128,
            lr=5e-3,                # constant LR (independent of X)
            weight_decay=0.0,
            loss="mse",
            psi_method="mean",
            seed_split=split_seed,
            seed_model1=model_seed1,
            seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY,
            xi_batch=256,
            activation="relu",
            init_mode="he"
        )
        _save_block(path, outdir / f"seed_{seed:04d}", prefix="mlp")
        del path; _gc_cuda()

    # --- CNN ---
    if "cnn" in models:
        path_cnn = torch_cnn1d_feature_selection_path(
            X=X, y=y, alpha=0.1, T=T,
            true_idx=true_idx,
            conv_channels=[64, 128, 128],
            kernel_sizes=[11, 9, 7],
            strides=[1, 1, 1],
            fc_dims=[128, 64],
            lift_dim=64,
            activation="relu", init_mode="he",
            dropout=0.1, use_bn=False,
            batch_size=128, lr=5e-4, weight_decay=0.0,
            loss="mse", psi_method="mean",
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY, xi_batch=256,
            xi_mode="sumgrad"  # or "steinized"
        )
        _save_block(path_cnn, outdir / f"seed_{seed:04d}", prefix="cnn")
        del path_cnn; _gc_cuda()




def _save_block(path_obj, seed_dir:Path, prefix:str, diagnostics=None):
    """PathResult から各指標をCSVに保存"""
    def _to2d(a):
        a = np.asarray(a)
        if a.ndim == 1: a = a[None,:]
        if a.size == 0: a = np.empty((0,0))
        return a
    import numpy as np, pandas as pd
    fdr  = _to2d(path_obj.FDR_true if path_obj.FDR_true is not None else [])
    t2   = _to2d(path_obj.TypeII if path_obj.TypeII is not None else [])
    loss = _to2d(path_obj.loss_values)
    steps = np.asarray(path_obj.steps, dtype=int)
    if not np.array_equal(steps, np.asarray(path_obj.loss_steps)):
        raise ValueError(f"Selection/loss recording steps differ: {prefix}")
    for name, values in (("fdr", fdr), ("typeII", t2), ("loss", loss)):
        if values.shape != (1, len(steps)):
            raise ValueError(f"Invalid {prefix}_{name} shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {prefix}_{name} values.")
        if name in ("fdr", "typeII") and np.any((values < 0) | (values > 1)):
            raise ValueError(f"{prefix}_{name} values outside [0, 1].")
    # Finish writing all new files before touching previous results.
    with tempfile.TemporaryDirectory(dir=seed_dir, prefix=f".{prefix}_") as tmp:
        staging = Path(tmp)
        for name, values in (("fdr", fdr), ("typeII", t2), ("loss", loss),
                             ("steps", steps[None, :])):
            pd.DataFrame(values).to_csv(
                staging / f"{prefix}_{name}.csv", index=False, header=False
            )
        if diagnostics is not None:
            np.savez_compressed(staging / f"{prefix}_diagnostics.npz", **diagnostics)
        _install_outputs(staging, seed_dir)

def _gc_cuda():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

# ========== ランチャ（並列実行） ==========
def launch(seeds, m, n, T, outdir, n_gpu=None, models=None, tr_T=None):
    models = _selected_models(models)
    transformer_T = _transformer_horizon(T, tr_T)
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and contain no duplicates.")
    t0 = time.time()
    try:
        import torch
        if n_gpu is None:
            n_gpu = torch.cuda.device_count()
    except Exception:
        if n_gpu is None: n_gpu = 1

    procs = []
    try:
        outdir = Path(outdir); (outdir/"logs").mkdir(parents=True, exist_ok=True)
        if n_gpu is None or n_gpu < 1:
            raise RuntimeError("At least one GPU is required.")

        workers_per_gpu = 2
        max_parallel = n_gpu * workers_per_gpu

        for i, s in enumerate(seeds):
            dev = i % n_gpu

            if i >= max_parallel:
                previous_index = i - max_parallel
                returncode = procs[previous_index].wait()
                if returncode != 0:
                    raise RuntimeError(
                        f"Worker for seed {seeds[previous_index]} failed "
                        f"with exit code {returncode}."
                    )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"]=str(dev)
            env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
            cmd=[sys.executable,__file__,"--mode","worker","--seed",str(s),"--m",str(m),"--n",str(n),"--T",str(T),"--outdir",str(outdir),
                 "--tr-T", str(transformer_T), "--models", *models]
            log_path = outdir / "logs" / (
                f"seed_{s:04d}_{'-'.join(models)}_T{T}_trT{transformer_T}_{time.time_ns()}.log"
            )
            with open(log_path, "x") as log:
                print(f"[LAUNCH] seed={s} -> GPU{dev}; log={log_path}")
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
            procs.append(process)
            time.sleep(0.2)
        for s, p in zip(seeds, procs):
            returncode = p.wait()
            if returncode != 0:
                raise RuntimeError(
                    f"Worker for seed {s} failed "
                    f"with exit code {returncode}."
                )

        elapsed = time.time() - t0

        def fmt(sec: float) -> str:
            h = int(sec // 3600); m_ = int((sec % 3600) // 60); s = int(sec % 60)
            return f"{h}h {m_:02d}m {s:02d}s" if h else (f"{m_}m {s:02d}s" if m_ else f"{s}s")

        notify_telegram(f"All seeds finished.\n"
                        f"Seeds: {seeds[0]}..{seeds[-1]}\n"
                        f"Elapsed: {fmt(elapsed)}\n"
                        f"Out: {outdir} on {os.uname().nodename}")
    
    except BaseException as e:
        for process in procs:
            if process.poll() is None:
                process.terminate()

        for process in procs:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        if isinstance(e, Exception):
            notify_telegram(f"❌ Job failed: {e}")
        raise



# ========== マージ ==========
def merge_all(outdir, seeds, T, compute_every=COMPUTE_EVERY, models=None, tr_T=None):
    outdir = Path(outdir)
    models = _selected_models(models)
    transformer_T = _transformer_horizon(T, tr_T)
    seeds = list(seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be nonempty and contain no duplicates.")
    if T < 1 or compute_every < 1:
        raise ValueError("T and compute_every must be positive.")

    model_steps = {
        model: _recorded_steps(
            transformer_T if model == "tr" else T,
            compute_every,
        )
        for model in models
    }
    keys = [f"{model}_{metric}"
            for model in models
            for metric in ("fdr", "typeII", "loss")]
    matrices = {}

    for key in keys:
        model = key.split("_", 1)[0]
        expected_columns = len(model_steps[model])
        rows = []
        for seed in seeds:
            path = outdir / f"seed_{seed:04d}" / f"{key}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing result: {path}")
            values = pd.read_csv(path, header=None).to_numpy(dtype=float)
            if values.shape != (1, expected_columns):
                raise ValueError(
                    f"Unexpected result shape {values.shape}, expected "
                    f"(1, {expected_columns}): {path}. Use the same --T and "
                    f"--tr-T as the launch (Transformer: {transformer_T} updates)."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite result: {path}")
            if key.endswith(("_fdr", "_typeII")):
                if np.any((values < 0) | (values > 1)):
                    raise ValueError(f"Metric outside [0, 1]: {path}")
            rows.append(values)
        matrices[key] = np.vstack(rows)

    # Check saved axes when available. Older CSV-only outputs can still be
    # merged if their lengths match; their axes follow the same recording rule.
    for model, expected_steps in model_steps.items():
        for seed in seeds:
            path = outdir / f"seed_{seed:04d}" / f"{model}_steps.csv"
            if path.is_file():
                actual_steps = pd.read_csv(path, header=None).to_numpy(dtype=float)
                if not np.array_equal(actual_steps, expected_steps[None, :]):
                    raise ValueError(f"Unexpected recording steps: {path}")

    # Validate every input before writing any merged result.
    with tempfile.TemporaryDirectory(dir=outdir, prefix=".merge_") as tmp:
        staging = Path(tmp)
        for key, matrix in matrices.items():
            pd.DataFrame(matrix).to_csv(
                staging / f"{key}.csv", header=False, index=False
            )
        for model, steps in model_steps.items():
            pd.DataFrame(steps[None, :]).to_csv(
                staging / f"{model}_steps.csv", header=False, index=False
            )
        _install_outputs(staging, outdir)
    for key, matrix in matrices.items():
        print(f"[merge] wrote {key}.csv shape={matrix.shape}")

# ========== CLI ==========
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["launch","worker","merge"],default="launch")
    ap.add_argument("--seeds",type=str,default="1-20")
    ap.add_argument("--seed",type=int)
    ap.add_argument("--m",type=int,default=2000)
    ap.add_argument("--n",type=int,default=500)
    ap.add_argument("--T",type=int,default=3000,
                    help="Base updates for MLP/CNN/RNN; Transformer defaults to 10*T.")
    ap.add_argument("--tr-T", dest="tr_T", type=int, default=None,
                    help="Transformer total updates from initialization (default: 10*T).")
    ap.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=None,
                    help="Models to run or merge, e.g. --models tr (default: all).")
    ap.add_argument("--outdir",type=str,default="res/iter_parallel")
    ap.add_argument("--ngpu",type=int,default=None)
    args=ap.parse_args()

    if args.mode == "worker":
        run_one_seed(args.seed, args.m, args.n, args.T, Path(args.outdir),
                     models=args.models, tr_T=args.tr_T)
        return

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(value) for value in args.seeds.split(",")]

    if args.mode == "merge":
        merge_all(Path(args.outdir), seeds, T=args.T, compute_every=COMPUTE_EVERY,
                  models=args.models, tr_T=args.tr_T)
        return

    launch(seeds, args.m, args.n, args.T, args.outdir, args.ngpu,
           models=args.models, tr_T=args.tr_T)

if __name__=="__main__": main()
