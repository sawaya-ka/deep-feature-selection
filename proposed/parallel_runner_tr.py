#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T4×4 環境で seed ごとにプロセスを立てて GPU に並列割り当てするスクリプト
- mode=launch : 複数GPUにシードをラウンドロビンで割当て並列実行
- mode=worker : 単一seedを処理
- mode=merge  : 各seedの結果CSVを縦に結合
"""

import os, sys, argparse, subprocess, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import gc


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
from util_tr import (
    generate_sim_data,
    torch_nn_feature_selection_path,
    torch_cnn1d_feature_selection_path,
    torch_rnn_feature_selection_path,
    torch_transformer_feature_selection_path,
)

# ========== ワーカー（1シードだけ処理） ==========
def run_one_seed(seed:int, m:int, n:int, T:int, outdir:Path):
    outdir = Path(outdir)
    (outdir / f"seed_{seed:04d}").mkdir(parents=True, exist_ok=True)

    X, y, true_idx, B = generate_sim_data(m=m, n=n, rng=seed)

    # # --- MLP ---
    # path = torch_nn_feature_selection_path(
    #     X, y, alpha=0.1, T=T,
    #     true_idx=true_idx,
    #     hidden_dims=[1024,1024,512,128],
    #     batch_size=256, lr=1e-3, weight_decay=0.0,
    #     loss="mse", psi_method="mean",
    #     seed_split=2025, seed_model1=111, seed_model2=222,
    #     compute_every=10, xi_batch=512, activation="relu", init_mode="he",
    #     lr_schedule=None
    # )
    # _save_block(path, outdir / f"seed_{seed:04d}", prefix="mlp")
    # del path; _gc_cuda()

    # # --- CNN ---
    # path_cnn = torch_cnn1d_feature_selection_path(
    #     X, y, alpha=0.1, T=T,
    #     true_idx=true_idx,
    #     conv_channels=[64,128,128], kernel_sizes=[11,9,7], strides=[1,1,1],
    #     fc_dims=[128,64], activation="relu", init_mode="he", dropout=0.1, use_bn=False,
    #     batch_size=128, lr=5e-3, weight_decay=0.0, loss="mse", psi_method="mean",
    #     seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=512,
    #     xi_mode="sumgrad"
    # )
    # _save_block(path_cnn, outdir / f"seed_{seed:04d}", prefix="cnn")
    # del path_cnn; _gc_cuda()

    # # --- RNN ---
    # path_rnn = torch_rnn_feature_selection_path(
    #     X, y, alpha=0.1, T=T,
    #     true_idx=true_idx,
    #     rnn_type="lstm", hidden_size=128, num_layers=2, bidirectional=True,
    #     input_proj_dim=4, fc_dims=[64], activation="relu", init_mode="xavier", dropout=0.1,
    #     batch_size=128, lr=5e-3, weight_decay=0.0, loss="mse", psi_method="mean",
    #     seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=512,
    #     xi_mode="sumgrad"
    # )
    # _save_block(path_rnn, outdir / f"seed_{seed:04d}", prefix="rnn")
    # del path_rnn; _gc_cuda()

    # --- Transformer ---
    path_tr = torch_transformer_feature_selection_path(
        X, y, alpha=0.1, T=T, true_idx=true_idx,
        lift_type="dense", id_encoding="linear",
        d_model=256, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1,
        pos_mode="none", pool="mean", fc_dims=[32],    # ← 入力直後の振幅を生かすため mean pooling
        lift_init_mode="xavier",
        batch_size=128, lr=3e-4, weight_decay=0.01, loss="mse", psi_method="mean",
        compute_every=10, xi_batch=1024, xi_mode="steinized"  # ← xi を steinized、バッチ少し増
    )
    _save_block(path_tr, outdir / f"seed_{seed:04d}", prefix="tr")
    del path_tr; _gc_cuda()

def _save_block(path_obj, seed_dir:Path, prefix:str):
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
    pd.DataFrame(fdr).to_csv(seed_dir/f"{prefix}_fdr.csv", index=False, header=False)
    pd.DataFrame(t2).to_csv(seed_dir/f"{prefix}_typeII.csv", index=False, header=False)
    pd.DataFrame(loss).to_csv(seed_dir/f"{prefix}_loss.csv", index=False, header=False)

def _gc_cuda():
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

# ========== ランチャ（並列実行） ==========
def launch(seeds, m, n, T, outdir, n_gpu=None):
    t0 = time.time()
    try:
        import torch
        if n_gpu is None:
            n_gpu = torch.cuda.device_count()
    except Exception:
        if n_gpu is None: n_gpu = 1

    try:
        outdir = Path(outdir); (outdir/"logs").mkdir(parents=True, exist_ok=True)
        procs=[]
        for i, s in enumerate(seeds):
            dev = i % n_gpu
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"]=str(dev)
            env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
            cmd=[sys.executable,__file__,"--mode","worker","--seed",str(s),"--m",str(m),"--n",str(n),"--T",str(T),"--outdir",str(outdir)]
            log=open(outdir/"logs"/f"seed_{s:04d}.log","w")
            print(f"[LAUNCH] seed={s} -> GPU{dev}")
            procs.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=log))
            time.sleep(0.2)
        for p in procs: p.wait()

        elapsed = time.time() - t0

        def fmt(sec: float) -> str:
            h = int(sec // 3600); m_ = int((sec % 3600) // 60); s = int(sec % 60)
            return f"{h}h {m_:02d}m {s:02d}s" if h else (f"{m_}m {s:02d}s" if m_ else f"{s}s")

        notify_telegram(f"All seeds finished.\n"
                        f"Seeds: {seeds[0]}..{seeds[-1]}\n"
                        f"Elapsed: {fmt(elapsed)}\n"
                        f"Out: {outdir} on {os.uname().nodename}")
    
    except Exception as e:
        notify_telegram(f"❌ Job failed: {e}")
        raise



# ========== マージ ==========
def merge_all(outdir):
    outdir=Path(outdir)
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

# ========== CLI ==========
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["launch","worker","merge"],default="launch")
    ap.add_argument("--seeds",type=str,default="1-20")
    ap.add_argument("--seed",type=int)
    ap.add_argument("--m",type=int,default=2000)
    ap.add_argument("--n",type=int,default=500)
    ap.add_argument("--T",type=int,default=3000)
    ap.add_argument("--outdir",type=str,default="res/iter_parallel")
    ap.add_argument("--ngpu",type=int,default=None)
    args=ap.parse_args()

    if args.mode=="worker":
        run_one_seed(args.seed,args.m,args.n,args.T,Path(args.outdir)); return
    if args.mode=="merge":
        merge_all(Path(args.outdir)); return
    # launch
    if "-" in args.seeds:
        a,b=args.seeds.split("-"); seeds=list(range(int(a),int(b)+1))
    else:
        seeds=[int(s) for s in args.seeds.split(",")]
    launch(seeds,args.m,args.n,args.T,args.outdir,args.ngpu)

if __name__=="__main__": main()
