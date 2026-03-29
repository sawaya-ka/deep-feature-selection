#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel runner for the MDS extension.

Queue-based launcher version:
- at most one worker process per visible GPU at any time;
- when one seed finishes on a GPU, the next pending seed is launched there;
- avoids the old launch-mode behavior that started all seeds at once.
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from util_origin import generate_sim_data
from util_mds import (
    torch_nn_feature_selection_path_mds,
    torch_cnn1d_feature_selection_path_mds,
    torch_rnn_feature_selection_path_mds,
    torch_transformer_feature_selection_path_mds,
)

alpha = 0.1


def notify_telegram(msg: str):
    token = os.environ.get("TG_TOKEN")
    chat = os.environ.get("TG_CHAT_ID")
    if not (token and chat):
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": int(chat), "text": msg, "disable_web_page_preview": True}
    try:
        subprocess.run(
            ["curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
            check=False,
        )
    except Exception:
        pass


# ========== worker ==========
def run_one_seed(
    seed: int,
    m: int,
    n: int,
    T: int,
    outdir: Path,
    mds_reps: int,
    mds_seed_base: int,
    vary_model_seeds: bool,
):
    outdir = Path(outdir)
    seed_dir = outdir / f"seed_{seed:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    X, y, true_idx, B = generate_sim_data(m=m, n=n, rng=seed)
    del B

    mds_seed = int(mds_seed_base + 100000 * seed)
    mds_common = dict(
        mds_reps=mds_reps,
        mds_seed=mds_seed,
        vary_model_seeds=vary_model_seeds,
        gc_each_split=True,
    )

    path = torch_nn_feature_selection_path_mds(
        X=X,
        y=y,
        alpha=alpha,
        T=T,
        true_idx=true_idx,
        hidden_dims=[1024, 1024, 512, 128],
        batch_size=256,
        lr=5e-3,
        weight_decay=0.0,
        loss="mse",
        psi_method="mean",
        seed_model1=111,
        seed_model2=222,
        compute_every=10,
        xi_batch=2048,
        activation="relu",
        init_mode="he",
        **mds_common,
    )
    _save_block(path, seed_dir, prefix="mlp")
    del path
    _gc_cuda()

    path_cnn = torch_cnn1d_feature_selection_path_mds(
        X=X,
        y=y,
        alpha=alpha,
        T=T,
        true_idx=true_idx,
        conv_channels=[64, 128, 128],
        kernel_sizes=[11, 9, 7],
        strides=[1, 1, 1],
        fc_dims=[128, 64],
        activation="relu",
        init_mode="he",
        dropout=0.1,
        use_bn=False,
        batch_size=128,
        lr=5e-3,
        weight_decay=0.0,
        loss="mse",
        psi_method="mean",
        seed_model1=11,
        seed_model2=22,
        compute_every=10,
        xi_batch=2048,
        xi_mode="sumgrad",
        **mds_common,
    )
    _save_block(path_cnn, seed_dir, prefix="cnn")
    del path_cnn
    _gc_cuda()

    path_rnn = torch_rnn_feature_selection_path_mds(
        X=X,
        y=y,
        alpha=alpha,
        T=T,
        true_idx=true_idx,
        rnn_type="lstm",
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        input_proj_dim=4,
        fc_dims=[64],
        activation="relu",
        init_mode="xavier",
        dropout=0.1,
        batch_size=128,
        lr=5e-3,
        weight_decay=0.0,
        loss="mse",
        psi_method="mean",
        seed_model1=11,
        seed_model2=22,
        compute_every=10,
        xi_batch=512,
        xi_mode="sumgrad",
        **mds_common,
    )
    _save_block(path_rnn, seed_dir, prefix="rnn")
    del path_rnn
    _gc_cuda()

    path_tr = torch_transformer_feature_selection_path_mds(
        X=X,
        y=y,
        alpha=alpha,
        T=T,
        true_idx=true_idx,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=516,
        dropout=0.1,
        activation="relu",
        init_mode="he",
        pool="mean",
        use_sinusoidal_pos=True,
        fc_dims=[64],
        batch_size=128,
        lr=5e-3,
        weight_decay=0.0,
        loss="mse",
        psi_method="mean",
        seed_model1=11,
        seed_model2=22,
        compute_every=10,
        xi_batch=512,
        xi_mode="sumgrad",
        **mds_common,
    )
    _save_block(path_tr, seed_dir, prefix="tr")
    del path_tr
    _gc_cuda()


def _save_block(path_obj, seed_dir: Path, prefix: str):
    def _to2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a[None, :]
        if a.size == 0:
            a = np.empty((0, 0))
        return a

    payloads = {
        "fdr": getattr(path_obj, "FDR_true", []),
        "typeII": getattr(path_obj, "TypeII", []),
        "loss": getattr(path_obj, "loss_values", []),
        "selected_size": getattr(path_obj, "selected_size", []),
        "cutoff": getattr(path_obj, "cutoff", []),
        "ds_fdr_mean": getattr(path_obj, "ds_FDR_true_mean", []),
        "ds_typeII_mean": getattr(path_obj, "ds_TypeII_mean", []),
    }
    for name, arr in payloads.items():
        pd.DataFrame(_to2d(arr)).to_csv(seed_dir / f"{prefix}_{name}.csv", index=False, header=False)


def _gc_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


# ========== launcher ==========
def _visible_gpu_list(requested_n_gpu=None):
    env_val = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env_val:
        gpus = [x.strip() for x in env_val.split(",") if x.strip() != ""]
    else:
        try:
            import torch
            gpus = [str(i) for i in range(torch.cuda.device_count())]
        except Exception:
            gpus = ["0"]
    if requested_n_gpu is not None:
        gpus = gpus[:requested_n_gpu]
    if not gpus:
        gpus = [""]
    return gpus


def launch(seeds, m, n, T, outdir, mds_reps, mds_seed_base, vary_model_seeds, n_gpu=None, poll_seconds=5.0):
    t0 = time.time()
    outdir = Path(outdir)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)

    gpu_list = _visible_gpu_list(n_gpu)
    n_slots = max(1, len(gpu_list))
    pending = list(seeds)
    running = {}
    finished = []
    failed = []

    def build_cmd(seed):
        cmd = [
            sys.executable,
            __file__,
            "--mode", "worker",
            "--seed", str(seed),
            "--m", str(m),
            "--n", str(n),
            "--T", str(T),
            "--outdir", str(outdir),
            "--mds-reps", str(mds_reps),
            "--mds-seed-base", str(mds_seed_base),
        ]
        if vary_model_seeds:
            cmd.append("--vary-model-seeds")
        return cmd

    def start_one(slot):
        if not pending:
            return False
        seed = pending.pop(0)
        gpu = gpu_list[slot]
        env = os.environ.copy()
        if gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log_path = outdir / "logs" / f"seed_{seed:04d}.log"
        logfh = open(log_path, "w")
        proc = subprocess.Popen(build_cmd(seed), env=env, stdout=logfh, stderr=logfh)
        running[slot] = {"proc": proc, "seed": seed, "gpu": gpu if gpu != "" else "CPU", "logfh": logfh}
        print(f"[LAUNCH] seed={seed} -> GPU {running[slot]['gpu']} (pid={proc.pid})", flush=True)
        return True

    try:
        for slot in range(min(n_slots, len(pending))):
            start_one(slot)

        while running:
            time.sleep(poll_seconds)
            completed_slots = []
            for slot, info in list(running.items()):
                ret = info["proc"].poll()
                if ret is None:
                    continue
                info["logfh"].close()
                seed = info["seed"]
                gpu = info["gpu"]
                if ret == 0:
                    finished.append(seed)
                    print(f"[DONE] seed={seed} on GPU {gpu}", flush=True)
                else:
                    failed.append((seed, ret, gpu))
                    print(f"[FAIL] seed={seed} on GPU {gpu} (exit={ret})", flush=True)
                completed_slots.append(slot)

            for slot in completed_slots:
                running.pop(slot, None)
                start_one(slot)

        elapsed = time.time() - t0

        def fmt(sec: float) -> str:
            h = int(sec // 3600)
            m_ = int((sec % 3600) // 60)
            s = int(sec % 60)
            return f"{h}h {m_:02d}m {s:02d}s" if h else (f"{m_}m {s:02d}s" if m_ else f"{s}s")

        if failed:
            msg = (
                f"⚠️ MDS seeds finished with failures.\n"
                f"OK: {len(finished)} / {len(seeds)}\n"
                f"Failed: {failed}\n"
                f"MDS reps: {mds_reps}\n"
                f"Elapsed: {fmt(elapsed)}\n"
                f"Out: {outdir} on {os.uname().nodename}"
            )
            notify_telegram(msg)
            raise RuntimeError(msg)

        notify_telegram(
            f"All MDS seeds finished.\n"
            f"Seeds: {seeds[0]}..{seeds[-1]}\n"
            f"MDS reps: {mds_reps}\n"
            f"Elapsed: {fmt(elapsed)}\n"
            f"Out: {outdir} on {os.uname().nodename}"
        )
    except Exception as e:
        notify_telegram(f"❌ MDS job failed: {e}")
        raise


# ========== merge ==========
def merge_all(outdir):
    outdir = Path(outdir)
    prefixes = ["mlp", "cnn", "rnn", "tr"]
    metrics = ["fdr", "typeII", "loss", "selected_size", "cutoff", "ds_fdr_mean", "ds_typeII_mean"]
    seed_dirs = sorted([p for p in outdir.glob("seed_*") if p.is_dir()])

    for prefix in prefixes:
        for metric in metrics:
            paths = [sd / f"{prefix}_{metric}.csv" for sd in seed_dirs]
            frames = [pd.read_csv(p, header=None).values for p in paths if p.exists()]
            if not frames:
                continue
            mat = np.vstack(frames)
            pd.DataFrame(mat).to_csv(outdir / f"{prefix}_{metric}.csv", header=False, index=False)
            print(f"[merge] wrote {prefix}_{metric}.csv shape={mat.shape}")


# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["launch", "worker", "merge"], default="launch")
    ap.add_argument("--seeds", type=str, default="1-20")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--m", type=int, default=2000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--outdir", type=str, default="res/iter_parallel_mds")
    ap.add_argument("--ngpu", type=int, default=None)
    ap.add_argument("--mds-reps", type=int, default=50)
    ap.add_argument("--mds-seed-base", type=int, default=2025)
    ap.add_argument("--vary-model-seeds", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    args = ap.parse_args()

    if args.mode == "worker":
        run_one_seed(
            seed=args.seed,
            m=args.m,
            n=args.n,
            T=args.T,
            outdir=Path(args.outdir),
            mds_reps=args.mds_reps,
            mds_seed_base=args.mds_seed_base,
            vary_model_seeds=args.vary_model_seeds,
        )
        return

    if args.mode == "merge":
        merge_all(Path(args.outdir))
        return

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    launch(
        seeds=seeds,
        m=args.m,
        n=args.n,
        T=args.T,
        outdir=args.outdir,
        mds_reps=args.mds_reps,
        mds_seed_base=args.mds_seed_base,
        vary_model_seeds=args.vary_model_seeds,
        n_gpu=args.ngpu,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
