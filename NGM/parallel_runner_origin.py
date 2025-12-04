#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, subprocess, time
from pathlib import Path
import numpy as np
import pandas as pd
import gc

from util_origin import (
    # 既存API（simモードで使用）
    generate_sim_data,
    torch_nn_feature_selection_path,
    torch_cnn1d_feature_selection_path,
    torch_rnn_feature_selection_path,
    torch_transformer_feature_selection_path,
    # 追加API（scenarioモードで使用）
    generate_design_with_splits,
    generate_y_given_X,
    torch_nn_feature_selection_path_from_splits,
    torch_cnn1d_feature_selection_path_from_splits,
    torch_rnn_feature_selection_path_from_splits,
)

alpha = 0.1

# ========== ワーカー（1シードだけ処理） ==========
def run_one_seed(seed:int, m:int, n:int, T:int, outdir:Path,
                 scenario:str|None=None, b1_scale:float=2.0):
    outdir = Path(outdir)
    (outdir / f"seed_{seed:04d}").mkdir(parents=True, exist_ok=True)

    # --- データ生成 ---
    if scenario in {"gaussian","t3","spiked","var"}:
        X_full, X_A, X_B = generate_design_with_splits(
            m=m, n=n, scenario=scenario, seed=seed,
            r_spike=2, var_alpha=[0.5], var_burnin=200
        )
        y_full, true_idx, B = generate_y_given_X(
            X_full, b1_scale=b1_scale, q_star=8, rng=seed+12345
        )
        m2 = m//2
        y_A, y_B = y_full[:m2], y_full[m2:]
    else:
        X, y, true_idx, B = generate_sim_data(m=m, n=n, rng=seed)

    # --- MLP ---
    if scenario in {"gaussian","t3","spiked","var"}:
        path = torch_nn_feature_selection_path_from_splits(
            X_A, X_B, y_A, y_B, alpha=alpha, T=T,
            true_idx=true_idx,
            hidden_dims=[1024, 1024, 512, 128],
            batch_size=256, lr=5e-3, weight_decay=0.0,
            loss="mse", psi_method="mean", xi_batch=2048,
            activation="relu", init_mode="he",
            seed_model1=111, seed_model2=222,
        )
    else:
        path = torch_nn_feature_selection_path(
            X=X, y=y, alpha=alpha, T=T,
            true_idx=true_idx,
            hidden_dims=[1024, 1024, 512, 128],
            batch_size=256, lr=5e-3, weight_decay=0.0,
            loss="mse", psi_method="mean",
            seed_split=2025, seed_model1=11, seed_model2=22,
            compute_every=10, xi_batch=512,
            activation="relu", init_mode="he",
            xi_mode="sumgrad"
        )
    _save_block(path, outdir / f"seed_{seed:04d}", prefix="mlp",
                scenario=scenario if scenario else None)

    # --- CNN ---
    if scenario in {"gaussian","t3","spiked","var"}:
        path_cnn = torch_cnn1d_feature_selection_path_from_splits(
            X_A, X_B, y_A, y_B, alpha=alpha, T=T,
            true_idx=true_idx,
            conv_channels=[64, 64], kernel_sizes=[11, 7], strides=[1, 1],
            fc_dims=[128], activation="relu", init_mode="he", bias_init=0.0,
            dropout=0.0, use_bn=False,
            batch_size=128, lr=5e-3, weight_decay=0.0,
            loss="mse", psi_method="mean", xi_batch=2048,
            seed_model1=11, seed_model2=22,
        )
    else:
        path_cnn = torch_cnn1d_feature_selection_path(
            X=X, y=y, alpha=alpha, T=T,
            true_idx=true_idx,
            lift_dim=None,
            conv_channels=[64, 64], kernel_sizes=[11, 7], strides=[1, 1],
            fc_dims=[128], activation="relu", init_mode="he", bias_init=0.0,
            dropout=0.0, use_bn=False,
            batch_size=128, lr=5e-3, weight_decay=0.0,
            loss="mse", psi_method="mean",
            seed_split=2025, seed_model1=11, seed_model2=22,
            compute_every=10, xi_batch=512,
            xi_mode="sumgrad"
        )
    _save_block(path_cnn, outdir / f"seed_{seed:04d}", prefix="cnn",
                scenario=scenario if scenario else None)

    # # --- RNN ---
    # if scenario in {"gaussian","t3","spiked","var"}:
    #     path_rnn = torch_rnn_feature_selection_path_from_splits(
    #         X_A, X_B, y_A, y_B, alpha=alpha, T=T,
    #         true_idx=true_idx,
    #         rnn_type="lstm", hidden_size=128, num_layers=2, bidirectional=True,
    #         input_proj_dim=4, fc_dims=[64],
    #         activation="relu", init_mode="xavier", bias_init=0.0, dropout=0.1,
    #         batch_size=128, lr=5e-3, weight_decay=0.0,
    #         loss="mse", psi_method="mean", xi_batch=2048,
    #         seed_model1=11, seed_model2=22,
    #     )
    # else:
    #     path_rnn = torch_rnn_feature_selection_path(
    #         X=X, y=y, alpha=alpha, T=T,
    #         true_idx=true_idx,
    #         rnn_type="lstm", hidden_size=128, num_layers=2, bidirectional=True,
    #         input_proj_dim=4,
    #         fc_dims=[64], activation="relu", init_mode="xavier",
    #         dropout=0.1,
    #         batch_size=128, lr=5e-3, weight_decay=0.0,
    #         loss="mse", psi_method="mean",
    #         seed_split=2025, seed_model1=11, seed_model2=22,
    #         compute_every=10, xi_batch=512,
    #         xi_mode="sumgrad"
    #     )
    # _save_block(path_rnn, outdir / f"seed_{seed:04d}", prefix="rnn",
    #             scenario=scenario if scenario else None)

    # --- Transformer は sim モードのみ ---
    if scenario is None:
        path_tr = torch_transformer_feature_selection_path(
            X=X, y=y, alpha=alpha, T=T,
            true_idx=true_idx,
            batch_size=128, lr=5e-3, weight_decay=0.0,
            loss="mse", psi_method="mean",
            seed_split=2025, seed_model1=11, seed_model2=22,
            compute_every=10, xi_batch=512,
            xi_mode="sumgrad"
        )
        _save_block(path_tr, outdir / f"seed_{seed:04d}", prefix="tr", scenario=None)
        del path_tr; _gc_cuda()

def _save_block(path_obj, seed_dir:Path, prefix:str, scenario:str|None=None):
    seed_dir.mkdir(parents=True, exist_ok=True)
    suf = ("" if scenario is None else f"_{scenario}")
    # FDR / TypeII / loss の配列形式（既存 path_obj と同じ前提）
    fdr = getattr(path_obj, "FDR_true", None)
    t2  = getattr(path_obj, "TypeII", None)
    loss = getattr(path_obj, "FDPhat", None)  # 既存に「学習損失」を別名で保存していない場合は FDPhat をダミーとして保存

    if fdr is not None:  pd.DataFrame(fdr).to_csv(seed_dir/f"{prefix}{suf}_fdr.csv", index=False, header=False)
    if t2 is not None:   pd.DataFrame(t2).to_csv(seed_dir/f"{prefix}{suf}_typeII.csv", index=False, header=False)
    if loss is not None: pd.DataFrame(loss).to_csv(seed_dir/f"{prefix}{suf}_loss.csv", index=False, header=False)

def _gc_cuda():
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

# ========== ランチャ（複数シード並列） ==========
def launch(seeds, m, n, T, outdir, n_gpu=None, scenario:str|None=None, b1_scale:float=2.0):
    t0 = time.time()
    try:
        import torch
        if n_gpu is None:
            n_gpu = torch.cuda.device_count()
    except Exception:
        if n_gpu is None: n_gpu = 1

    outdir = Path(outdir); (outdir/"logs").mkdir(parents=True, exist_ok=True)
    procs=[]
    for i, s in enumerate(seeds):
        dev = i % n_gpu
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"]=str(dev)
        env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
        cmd=[sys.executable,__file__,"--mode","worker","--seed",str(s),"--m",str(m),"--n",str(n),"--T",str(T),"--outdir",str(outdir)]
        if scenario is not None:
            cmd += ["--scenario", str(scenario), "--b1_scale", str(b1_scale)]
        log=open(outdir/"logs"/f"seed_{s:04d}.log","w")
        print(f"[LAUNCH] seed={s} -> GPU{dev}")
        procs.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=log))
        time.sleep(0.2)
    for p in procs: p.wait()

    elapsed = time.time() - t0
    print(f"[DONE] {len(seeds)} seeds in {elapsed:.1f}s; out={outdir}")

# ========== 集計 ==========
def merge_all(outdir):
    outdir=Path(outdir)
    # 従来（sim）集計
    keys=["mlp_fdr","mlp_typeII","mlp_loss","cnn_fdr","cnn_typeII","cnn_loss",
          "rnn_fdr","rnn_typeII","rnn_loss","tr_fdr","tr_typeII","tr_loss"]
    seed_dirs=sorted([p for p in outdir.glob("seed_*") if p.is_dir()])
    for key in keys:
        prefix,metric=key.split("_",1)
        paths=[sd/f"{prefix}_{metric}.csv" for sd in seed_dirs]
        frames=[pd.read_csv(p,header=None).values for p in paths if p.exists()]
        if not frames: continue
        mat=np.vstack(frames)
        pd.DataFrame(mat).to_csv(outdir/f"{key}.csv",header=False,index=False)
        print(f"[merge] wrote {key}.csv shape={mat.shape}")
    # シナリオ集計
    scenarios=["gaussian","t3","spiked","var"]
    for sc in scenarios:
        for key in ["mlp_fdr","mlp_typeII","mlp_loss","cnn_fdr","cnn_typeII","cnn_loss","rnn_fdr","rnn_typeII","rnn_loss"]:
            prefix,metric=key.split("_",1)
            paths=[sd/f"{prefix}_{sc}_{metric}.csv" for sd in seed_dirs]
            frames=[pd.read_csv(p,header=None).values for p in paths if p.exists()]
            if not frames: continue
            mat=np.vstack(frames)
            pd.DataFrame(mat).to_csv(outdir/f"{key}_{sc}.csv",header=False,index=False)
            print(f"[merge] wrote {key}_{sc}.csv shape={mat.shape}")

# ========== CLI ==========
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["launch","worker","merge"],default="launch")
    ap.add_argument("--seeds",type=str,default="1-20")
    ap.add_argument("--seed",type=int)
    ap.add_argument("--m",type=int,default=2000)
    ap.add_argument("--n",type=int,default=500)
    ap.add_argument("--T",type=int,default=1000)
    ap.add_argument("--outdir",type=str,default="res/from_splits")
    ap.add_argument("--ngpu",type=int,default=None)
    ap.add_argument("--scenario",type=str,default=None,
                    help="one of {gaussian,t3,spiked,var}; None = original sim DGP")
    ap.add_argument("--b1_scale",type=float,default=2.0,
                    help="signal scale for y when using scenario mode")
    args=ap.parse_args()

    if args.mode=="worker":
        run_one_seed(args.seed,args.m,args.n,args.T,Path(args.outdir),
                     scenario=args.scenario,b1_scale=args.b1_scale); return
    if args.mode=="merge":
        merge_all(Path(args.outdir)); return

    # launch
    if "-" in args.seeds:
        a,b=args.seeds.split("-"); seeds=list(range(int(a),int(b)+1))
    else:
        seeds=[int(s) for s in args.seeds.split(",")]
    launch(seeds,args.m,args.n,args.T,args.outdir,args.ngpu,
           scenario=args.scenario,b1_scale=args.b1_scale)

if __name__=="__main__": main()
