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
- マルチクラス分類: K個のlogitを出力し、Cross-Entropyで学習。
- --xi-scalar contrast（既定）は最後と最初のクラスのlogit差。
- 学習は毎回初期値から。CSV/診断NPZは再開用チェックポイントではない。
- TransformerはPre-LN、W1を含む全層をAdamで更新。
- 各 *_steps.csv は実際の更新回数（Transformerも圧縮せず保存）。
- 3クラスの非線形回転teacher。--teacher-lambda 0.5 --teacher-tau 0.1 が既定。
- --label-mode argmax は完全に決定的なラベル（温度は正の値を指定）。
- 全モデルについて *_diagnostics.npz に重要度・選択数・教師情報を保存。
"""

import os, sys, argparse, subprocess, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import gc
import shutil
import tempfile
import hashlib


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


alpha = 0.1
TRANSFORMER_T_MULTIPLIER = 10
TRANSFORMER_ADAM_LR = 5e-4
COMPUTE_EVERY = 10
MODEL_ORDER = ("tr", "rnn", "mlp", "cnn")
XI_SCALARS = ("contrast", "logsumexp", "sum", "max", "l2")
TEACHER_NAME = "rotated_softmax_v1"
FORMAT_VERSION = 2
COMMON_FIELDS = (
    "format_version", "task", "seed", "m", "n", "K", "data_seed",
    "seed_split", "seed_model1", "seed_model2", "data_sha256",
    "loss", "alpha", "psi_method", "xi_mode", "xi_scalar", "compute_every",
    "teacher", "teacher_lambda", "teacher_tau", "teacher_basis_seed", "label_mode",
)
SEED_FIELDS = {"seed", "data_seed", "seed_split", "seed_model1", "seed_model2", "data_sha256"}


def _validate_classification_args(m, n, K, xi_scalar):
    if m < 2 or m % 2 or n < 6 or n % 2:
        raise ValueError("Use even m >= 2 and even n >= 6 for the three-index teacher.")
    if K != 3:
        raise ValueError("The rotating-boundary teacher requires --K 3.")
    if xi_scalar not in XI_SCALARS:
        raise ValueError(f"xi-scalar must be one of {XI_SCALARS}.")


def _teacher_config(K, teacher_lambda=0.5, teacher_tau=0.1,
                    teacher_basis_seed=314159, label_mode="softmax"):
    if K != 3:
        raise ValueError("The rotating-boundary teacher requires --K 3.")
    if not np.isfinite(teacher_lambda) or teacher_lambda < 0:
        raise ValueError("teacher-lambda must be finite and nonnegative.")
    if not np.isfinite(teacher_tau) or teacher_tau <= 0:
        raise ValueError("teacher-tau must be finite and strictly positive.")
    if (isinstance(teacher_basis_seed, (bool, np.bool_))
            or not isinstance(teacher_basis_seed, (int, np.integer))
            or teacher_basis_seed < 0):
        raise ValueError("teacher-basis-seed must be a nonnegative integer.")
    if label_mode not in ("softmax", "argmax"):
        raise ValueError("label-mode must be 'softmax' or 'argmax'.")
    return dict(teacher=TEACHER_NAME, teacher_lambda=float(teacher_lambda),
                teacher_tau=float(teacher_tau), teacher_basis_seed=int(teacher_basis_seed),
                label_mode=label_mode)


def _data_fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _read_config(seed_dir, model):
    path = seed_dir / f"{model}_config.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing experiment configuration: {path}. Legacy classification "
            "outputs cannot be verified; use a new output directory for the corrected run."
        )
    config = json.loads(path.read_text())
    if any(key not in config for key in COMMON_FIELDS):
        raise ValueError(f"Incomplete experiment configuration: {path}")
    return config


def _check_retained_models(seed_dir, selected, current):
    for model in MODEL_ORDER:
        if model in selected:
            continue
        if not any((seed_dir / f"{model}_{metric}.csv").exists()
                   for metric in ("fdr", "typeII", "loss")):
            continue
        saved = _read_config(seed_dir, model)
        for key in COMMON_FIELDS:
            if saved[key] != current[key]:
                raise ValueError(
                    f"Retained {model} result differs at {key} in {seed_dir}. "
                    "Use matching data/scalarization settings or a new --outdir."
                )


def _validate_merge_configs(outdir, seeds, model_steps, K, xi_scalar, compute_every,
                            teacher_config):
    configs = {model: [] for model in model_steps}
    by_seed = {}
    for model, steps in model_steps.items():
        template = None
        for seed in seeds:
            cfg = _read_config(outdir / f"seed_{seed:04d}", model)
            expected = dict(format_version=FORMAT_VERSION, task="multiclass_classification",
                            seed=seed, model=model, updates=int(steps[-1]),
                            K=K, xi_scalar=xi_scalar, compute_every=compute_every,
                            loss="ce", alpha=alpha, psi_method="mean", xi_mode="sumgrad",
                            **teacher_config)
            for key, value in expected.items():
                if cfg.get(key) != value:
                    raise ValueError(f"seed {seed}, {model}: configuration mismatch at {key}.")
            setting = {key: value for key, value in cfg.items() if key not in SEED_FIELDS}
            if template is not None and setting != template:
                raise ValueError(f"Inconsistent {model} settings across seeds.")
            template = setting
            common = {key: cfg[key] for key in COMMON_FIELDS}
            if seed in by_seed and by_seed[seed] != common:
                raise ValueError(f"Different data or scalarization across models for seed {seed}.")
            by_seed[seed] = common
            configs[model].append(cfg)
    # Keep retained merged results comparable when merging only a subset.
    for model in MODEL_ORDER:
        if model in model_steps or not (outdir / f"{model}_fdr.csv").exists():
            continue
        path = outdir / f"{model}_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing configuration for retained merged results: {path}")
        saved = json.loads(path.read_text())
        if saved.get("seeds") != seeds or len(saved.get("runs", [])) != len(seeds):
            raise ValueError(f"Retained merged {model} uses a different seed list.")
        for seed, cfg in zip(seeds, saved["runs"]):
            if {key: cfg.get(key) for key in COMMON_FIELDS} != by_seed[seed]:
                raise ValueError(f"Retained merged {model} differs for seed {seed}.")
    return configs


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
    """Match util_clsf's every-k-updates plus final-update recording."""
    if T < 1 or compute_every < 1:
        raise ValueError("T and compute_every must be positive.")
    steps = list(range(compute_every, T + 1, compute_every))
    if not steps or steps[-1] != T:
        steps.append(T)
    return np.asarray(steps, dtype=int)

# ========== ワーカー（1シードだけ処理） ==========
def run_one_seed(seed:int, m:int, n:int, T:int, outdir:Path, K:int=3,
                 models=None, tr_T=None, xi_scalar="contrast",
                 teacher_lambda=0.5, teacher_tau=0.1,
                 teacher_basis_seed=314159, label_mode="softmax"):
    teacher_config = _teacher_config(
        K, teacher_lambda, teacher_tau, teacher_basis_seed, label_mode)
    _validate_classification_args(m, n, K, xi_scalar)
    from util_clsf import (
        generate_rotated_softmax_data, torch_nn_feature_selection_path,
        torch_cnn1d_feature_selection_path, torch_rnn_feature_selection_path,
        torch_transformer_feature_selection_path,
    )
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

    X, y, true_idx, B, teacher_details = generate_rotated_softmax_data(
        m=m, n=n, K=K, seed=data_seed,
        twist=teacher_lambda, tau=teacher_tau, basis_seed=teacher_basis_seed,
        label_mode=label_mode, return_details=True,
    )
    common_config = dict(
        format_version=FORMAT_VERSION, task="multiclass_classification", seed=int(seed),
        m=m, n=n, K=K, data_seed=data_seed, seed_split=split_seed,
        seed_model1=model_seed1, seed_model2=model_seed2,
        data_sha256=_data_fingerprint(X, y, true_idx, B),
        loss="ce", alpha=alpha, psi_method="mean", xi_mode="sumgrad",
        xi_scalar=xi_scalar, compute_every=COMPUTE_EVERY,
        **teacher_config,
    )
    _check_retained_models(seed_dir, models, common_config)
    # Reject different teachers/data even when rerunning the same selected model.
    # Changing the horizon or optimizer is allowed; the data protocol is fixed.
    for model in models:
        existing = seed_dir / f"{model}_config.json"
        if existing.is_file():
            saved = _read_config(seed_dir, model)
            for key in COMMON_FIELDS:
                if saved[key] != common_config[key]:
                    raise ValueError(
                        f"Existing {model} result differs at {key} in {seed_dir}. "
                        "Use a new --outdir for a different teacher/data protocol."
                    )
        elif any((seed_dir / f"{model}_{metric}.csv").exists()
                 for metric in ("fdr", "typeII", "loss")):
            raise ValueError(f"Unverified legacy results in {seed_dir}; use a new --outdir.")
    print(
        f"[Teacher] {TEACHER_NAME}, lambda={teacher_lambda}, tau={teacher_tau}, "
        f"label_mode={label_mode}, class_counts={teacher_details['class_counts'].tolist()}, "
        f"Bayes_accuracy_on_X={teacher_details['bayes_accuracy_on_X']:.4f}, "
        f"Bayes_CE_on_X={teacher_details['bayes_ce_on_X']:.4f}", flush=True,
    )

    def save_result(path_obj, model, updates, diagnostics=None):
        config = dict(common_config, model=model, updates=updates,
                      optimizer="adam" if model == "tr" else "sgd",
                      learning_rate=(TRANSFORMER_ADAM_LR if model == "tr"
                                     else 5e-4 if model == "cnn" else 5e-3),
                      weight_decay=0.0,
                      norm_first=(True if model == "tr" else None))
        all_diagnostics = dict(
            steps=path_obj.steps, xi1=path_obj.xi1_hist, xi2=path_obj.xi2_hist,
            M=path_obj.M_hist, tau=path_obj.tau,
            R_plus=path_obj.R_plus, R_minus=path_obj.R_minus,
            FDP_hat=path_obj.FDPhat, FDP=path_obj.FDR_true, typeII=path_obj.TypeII,
            loss_steps=path_obj.loss_steps, loss=path_obj.loss_values,
            true_idx=np.asarray(true_idx), B=B,
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            K=K, xi_scalar=xi_scalar, data_seed=data_seed,
            teacher_probabilities=teacher_details["probabilities"],
            reference_contrast_mean_gradient=teacher_details["reference_contrast_mean_gradient"],
            reference_is_log_odds=(label_mode == "softmax"),
            class_counts=teacher_details["class_counts"],
            bayes_accuracy_on_X=teacher_details["bayes_accuracy_on_X"],
            bayes_ce_on_X=teacher_details["bayes_ce_on_X"],
            **teacher_config,
        )
        if diagnostics is not None:
            all_diagnostics.update(diagnostics)
        _save_block(path_obj, seed_dir, prefix=model,
                    diagnostics=all_diagnostics, config=config)

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
            true_idx=true_idx, num_classes=K,
            lift_dim=64,
            d_model=256, nhead=4, num_layers=2,
            dim_feedforward=512, dropout=0.1,
            activation="relu", init_mode="he", pool="mean",
            use_sinusoidal_pos=True, fc_dims=[64],
            batch_size=128, lr=TRANSFORMER_ADAM_LR, weight_decay=0.0,
            loss="ce", psi_method="mean",
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY, xi_batch=256,
            xi_mode="sumgrad", xi_scalar=xi_scalar,
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
            K=K, xi_scalar=xi_scalar, data_seed=data_seed,
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
        save_result(path_tr, "tr", transformer_T, diagnostics=diagnostics)
        del path_tr, diagnostics; _gc_cuda()

    # --- CNN ---
    if "cnn" in models:
        path_cnn = torch_cnn1d_feature_selection_path(
            X=X, y=y, alpha=0.1, T=T,
            true_idx=true_idx, num_classes=K,
            conv_channels=[32, 32],
            kernel_sizes=[3,3],
            strides=[1, 1],
            fc_dims=[32, 16],
            lift_dim=16,
            activation="relu", init_mode="he",
            dropout=0., use_bn=False,
            batch_size=128, lr=0.03, weight_decay=0.0,
            loss="ce", psi_method="mean",
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY, xi_batch=256,
            xi_mode="sumgrad", xi_scalar=xi_scalar
        )
        save_result(path_cnn, "cnn", T)
        del path_cnn; _gc_cuda()

    # --- RNN ---
    if "rnn" in models:
        path_rnn = torch_rnn_feature_selection_path(
            X=X, y=y, alpha=0.1, T=T,
            true_idx=true_idx, num_classes=K,
            rnn_type="lstm", hidden_size=32, num_layers=1, bidirectional=True,
            input_proj_dim=4,
            lift_dim=16,
            fc_dims=[32], activation="relu", init_mode="xavier",
            dropout=0.,
            batch_size=128, lr=0.03, weight_decay=0.0,
            loss="ce", psi_method="mean",
            seed_split=split_seed, seed_model1=model_seed1, seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY, xi_batch=256,
            xi_mode="sumgrad", xi_scalar=xi_scalar
        )
        save_result(path_rnn, "rnn", T)
        del path_rnn; _gc_cuda()

    # --- MLP ---
    if "mlp" in models:
        path = torch_nn_feature_selection_path(
            X=X, y=y, alpha=alpha, T=T,
            true_idx=true_idx, num_classes=K,
            hidden_dims=[1024, 512, 128],
            batch_size=256,
            lr=5e-3,                # constant LR (independent of X)
            weight_decay=0.0,
            loss="ce",
            psi_method="mean",
            seed_split=split_seed,
            seed_model1=model_seed1,
            seed_model2=model_seed2,
            compute_every=COMPUTE_EVERY,
            xi_batch=256,
            activation="relu",
            init_mode="he", xi_scalar=xi_scalar
        )
        save_result(path, "mlp", T)
        del path; _gc_cuda()




def _save_block(path_obj, seed_dir:Path, prefix:str, diagnostics=None, config=None):
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
    raw_steps = np.asarray(path_obj.steps)
    if (raw_steps.ndim != 1 or raw_steps.size == 0
            or not np.isfinite(raw_steps).all() or np.any(raw_steps <= 0)
            or np.any(raw_steps != np.floor(raw_steps)) or np.any(np.diff(raw_steps) <= 0)):
        raise ValueError(f"Invalid {prefix} recording steps.")
    steps = raw_steps.astype(int)
    if not np.array_equal(steps, np.asarray(path_obj.loss_steps)):
        raise ValueError(f"Selection/loss recording steps differ: {prefix}")
    if config is not None and not np.array_equal(
            steps, _recorded_steps(config["updates"], config["compute_every"])):
        raise ValueError(f"Recording steps differ from the {prefix} configuration.")
    for name, values in (("fdr", fdr), ("typeII", t2), ("loss", loss)):
        if values.shape != (1, len(steps)):
            raise ValueError(f"Invalid {prefix}_{name} shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {prefix}_{name} values.")
        if name in ("fdr", "typeII") and np.any((values < 0) | (values > 1)):
            raise ValueError(f"{prefix}_{name} values outside [0, 1].")
        if name == "loss" and np.any(values < 0):
            raise ValueError(f"{prefix}_loss values must be nonnegative.")
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
        if config is not None:
            (staging / f"{prefix}_config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        _install_outputs(staging, seed_dir)

def _gc_cuda():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

# ========== ランチャ（並列実行） ==========
def launch(seeds, m, n, T, outdir, n_gpu=None, K=3, models=None, tr_T=None,
           xi_scalar="contrast", workers_per_gpu=2,
           teacher_lambda=0.5, teacher_tau=0.1,
           teacher_basis_seed=314159, label_mode="softmax"):
    _validate_classification_args(m, n, K, xi_scalar)
    _teacher_config(K, teacher_lambda, teacher_tau, teacher_basis_seed, label_mode)
    if workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive.")
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
                 "--tr-T", str(transformer_T), "--K", str(K),
                 "--teacher-lambda", str(teacher_lambda), "--teacher-tau", str(teacher_tau),
                 "--teacher-basis-seed", str(teacher_basis_seed), "--label-mode", label_mode,
                 "--xi-scalar", xi_scalar, "--models", *models]
            log_path = outdir / "logs" / (
                f"seed_{s:04d}_{'-'.join(models)}_K{K}_{xi_scalar}_T{T}_trT{transformer_T}_{time.time_ns()}.log"
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
def merge_all(outdir, seeds, T, compute_every=COMPUTE_EVERY, models=None, tr_T=None,
              K=3, xi_scalar="contrast", teacher_lambda=0.5, teacher_tau=0.1,
              teacher_basis_seed=314159, label_mode="softmax"):
    teacher_config = _teacher_config(
        K, teacher_lambda, teacher_tau, teacher_basis_seed, label_mode)
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

    # Check saved axes when available. Verified configurations are required
    # below even if an axis file is missing.
    for model, expected_steps in model_steps.items():
        for seed in seeds:
            path = outdir / f"seed_{seed:04d}" / f"{model}_steps.csv"
            if path.is_file():
                actual_steps = pd.read_csv(path, header=None).to_numpy(dtype=float)
                if not np.array_equal(actual_steps, expected_steps[None, :]):
                    raise ValueError(f"Unexpected recording steps: {path}")

    configs = _validate_merge_configs(
        outdir, seeds, model_steps, K, xi_scalar, compute_every, teacher_config)
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
        for model in models:
            (staging / f"{model}_config.json").write_text(json.dumps(
                dict(seeds=seeds, runs=configs[model]),
                indent=2, sort_keys=True, allow_nan=False,
            ) + "\n")
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
    ap.add_argument("--T",type=int,default=2000,
                    help="Base updates for MLP/CNN/RNN; Transformer defaults to 10*T.")
    ap.add_argument("--tr-T", dest="tr_T", type=int, default=None,
                    help="Transformer total updates from initialization (default: 10*T).")
    ap.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=None,
                    help="Models to run or merge, e.g. --models tr (default: all).")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--teacher-lambda", type=float, default=0.5,
                    help="Rotation strength; 0 gives a linear boundary.")
    ap.add_argument("--teacher-tau", type=float, default=0.1,
                    help="Positive softmax temperature; smaller gives less label noise.")
    ap.add_argument("--teacher-basis-seed", type=int, default=314159,
                    help="Fixed teacher directions shared by all simulation seeds.")
    ap.add_argument("--label-mode", choices=("softmax", "argmax"), default="softmax",
                    help="Sample softmax labels or use a deterministic argmax teacher.")
    ap.add_argument("--xi-scalar", choices=XI_SCALARS, default="contrast",
                    help="Label-free scalar score differentiated for feature importance.")
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--outdir",type=str,default="res_clsf/rotated")
    ap.add_argument("--ngpu",type=int,default=None)
    args=ap.parse_args()
    teacher_options = dict(
        teacher_lambda=args.teacher_lambda, teacher_tau=args.teacher_tau,
        teacher_basis_seed=args.teacher_basis_seed, label_mode=args.label_mode,
    )
    _teacher_config(args.K, **teacher_options)

    if args.mode == "worker":
        run_one_seed(args.seed, args.m, args.n, args.T, Path(args.outdir),
                     models=args.models, tr_T=args.tr_T, K=args.K, xi_scalar=args.xi_scalar,
                     **teacher_options)
        return

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(value) for value in args.seeds.split(",")]

    if args.mode == "merge":
        merge_all(Path(args.outdir), seeds, T=args.T, compute_every=COMPUTE_EVERY,
                  models=args.models, tr_T=args.tr_T, K=args.K, xi_scalar=args.xi_scalar,
                  **teacher_options)
        return

    launch(seeds, args.m, args.n, args.T, args.outdir, args.ngpu,
           models=args.models, tr_T=args.tr_T, K=args.K, xi_scalar=args.xi_scalar,
           workers_per_gpu=args.workers_per_gpu, **teacher_options)

if __name__=="__main__": main()
