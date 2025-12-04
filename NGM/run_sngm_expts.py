# run_sngm_expts.py
# 20 seeds × {gaussian,t3,spiked} を GPU 枚数に合わせて同時実行。
# - 同時起動数を GPU 台数に制限（CPU 過負荷を防ぐ）
# - 各ジョブで OMP/MKL スレッドを 1 に制限
# - progress.json を各 seed フォルダに書く
# - summary.csv を書き出す

import os
import json
import time
from pathlib import Path
from multiprocessing import Process
import numpy as np
import torch
import pandas as pd

from sngm import sngm_select, SNGMConfig
from util_origin import generate_design_with_splits, generate_y_given_X

import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # 既に設定済みなら無視

OUT_ROOT = Path("./res_sngm_m=500")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["gaussian", "t3", "spiked"]
SEEDS = list(range(1, 21))
M, N = 500, 500
Q_FDR = 0.1

# 軽量プリセット（必要に応じ調整）
CFG_BASE = dict(
    q_fdr=Q_FDR,
    epochs=60,
    act="relu",
    screen_k=N // 2,            # 500 -> 250
    approx_kw="subsample:800",  # "full" か "subsample:400/800" 推奨
    c_grid_num=6,
    verbose=True,
    log_every_epoch=10,
    sigma_mode="fixed"
)

def fdr_power(selected: np.ndarray, true_idx: np.ndarray, p: int):
    S = set(map(int, np.array(selected).reshape(-1).tolist()))
    T = set(map(int, np.array(true_idx).reshape(-1).tolist()))
    R = len(S); TP = len(S & T); FP = R - TP
    fdr = FP / max(R, 1)
    power = TP / max(len(T), 1)
    return dict(FDR=float(fdr), Power=float(power), R=R, TP=TP, FP=FP)

def run_one(seed: int, scenario: str, gpu_id: int):
    # CPU 過負荷防止（各プロセスでスレッド1）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    od = OUT_ROOT / scenario / f"seed_{seed:04d}"
    od.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] scenario={scenario} seed={seed:04d} gpu={gpu_id} device={device} out={od}")

    # 既に完了済みならスキップ
    metrics_path = od / "metrics.json"
    if metrics_path.exists():
        print(f"[SKIP] {metrics_path} exists.")
        return

    # データ生成
    X_full, _, _ = generate_design_with_splits(
        m=M, n=N, scenario=scenario, seed=seed, r_spike=2, var_alpha=[0.5], var_burnin=200
    )
    y_full, true_idx, _ = generate_y_given_X(X_full, b1_scale=2.0, q_star=8, rng=seed + 12345)

    # SNGM 実行（進捗を出力先に書く）
    cfg = SNGMConfig(**CFG_BASE, progress_path=str(od / "progress.json"))
    out = sngm_select(X_full.astype(np.float32), y_full.astype(np.float32), cfg, device=device)

    met = fdr_power(out["selected"], true_idx, p=N)
    rec = dict(seed=seed, scenario=scenario, tau=out["tau"], **met,
               n_sel=int(len(out["selected"])))
    (od / "metrics.json").write_text(json.dumps(rec, indent=2))
    (od / "selected.json").write_text(json.dumps(dict(selected=list(map(int, out["selected"])))))
    print(f"[DONE] scenario={scenario} seed={seed:04d} -> FDR={rec['FDR']:.3f}, Power={rec['Power']:.3f}, R={rec['R']}")

def schedule_all(gpus=(0, 1, 2, 3)):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # ← 追加
    tasks = [(sc, sd) for sc in SCENARIOS for sd in SEEDS]
    slots = {g: None for g in gpus}  # gpu -> Process or None
    i = 0

    while i < len(tasks) or any(p is not None for p in slots.values()):
        # 空き GPU に投入
        for g in gpus:
            if i >= len(tasks):
                break
            if slots[g] is None or not slots[g].is_alive():
                if slots[g] is not None:
                    slots[g].join()
                sc, sd = tasks[i]; i += 1
                p = ctx.Process(target=run_one, args=(sd, sc, g))  # ← ここを ctx.Process に
                p.start()
                slots[g] = p
        time.sleep(0.5)

    # すべて join
    for g, p in slots.items():
        if p is not None:
            p.join()

    # サマリ
    rows = []
    for scn in SCENARIOS:
        scn_dir = OUT_ROOT / scn
        if not scn_dir.exists():
            continue
        for sd in sorted(scn_dir.glob("seed_*")):
            mpath = sd / "metrics.json"
            if mpath.exists():
                rows.append(json.loads(mpath.read_text()))

    if not rows:
        print("[WARN] no metrics found. Check logs and paths.")
        return

    by_scn = {}
    for r in rows:
        by_scn.setdefault(r["scenario"], []).append(r)

    summary_rows = []
    for scn, lst in by_scn.items():
        F = np.array([x["FDR"]   for x in lst], dtype=float)
        P = np.array([x["Power"] for x in lst], dtype=float)
        R = np.array([x["R"]     for x in lst], dtype=float)
        summ = dict(
            scenario=scn,
            FDR_mean=float(np.mean(F)), FDR_std=float(np.std(F)),
            Power_mean=float(np.mean(P)), Power_std=float(np.std(P)),
            R_mean=float(np.mean(R)), R_std=float(np.std(R)),
            n_seeds=len(lst)
        )
        summary_rows.append(summ)
        print("[Summary]", summ)

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT_ROOT / "summary.csv", index=False)
    print(f"[WROTE] {OUT_ROOT / 'summary.csv'}")

if __name__ == "__main__":
    gstr = os.environ.get("GPUS", "0,1,2,3")
    gpus = tuple(int(x) for x in gstr.split(",") if x.strip() != "")
    print(f"[LAUNCH] GPUs={gpus} seeds={SEEDS} scenarios={SCENARIOS}")
    schedule_all(gpus=gpus)
