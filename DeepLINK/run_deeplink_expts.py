#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, subprocess, json
from pathlib import Path
import numpy as np
import pandas as pd

# === ここをあなたのプロジェクト構成に合わせて調整 ===
# util_origin.py があるディレクトリを PYTHONPATH に追加
PROJ_ROOT = Path(__file__).resolve().parent  # このスクリプトの場所
sys.path.append(str(PROJ_ROOT))
from util_origin import generate_design_with_splits, generate_y_given_X

# DeepLINK の場所（環境変数で指すのが楽）
DEEPLINK_DIR = os.environ.get("DEEPLINK_DIR", str(PROJ_ROOT / "DeepLINK"))
DEEPLINK_PY  = str(Path(DEEPLINK_DIR) / "deeplink.py")

# 実験設定
SCENARIOS = ["gaussian", "t3", "spiked"]   # 今回は VAR を除外
SEEDS     = list(range(1, 21))             # 1..20
M, N      = 500, 500
B1_SCALE  = 2.0
Q_FDR     = 0.1                             # DeepLINK 側の FDR レベル

# 出力ルート（DeepLINK出力と集計）
OUT_ROOT  = PROJ_ROOT / "deeplink_runs_m=500"     # 例: ./deeplink_runs/spiked/seed_0001/...
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_npy(X: np.ndarray, y: np.ndarray, outdir: Path):
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False).reshape(-1, 1)
    np.save(outdir / "X.npy", X)
    np.save(outdir / "y.npy", y)

def run_deeplink(X_path: Path, y_path: Path, outdir: Path, q: float = Q_FDR,
                 l1: float = 1e-3, lr: float = 1e-3, act: str = "elu",
                 loss: str = "mean_squared_error", center_and_scale: bool = True):
    ensure_dir(outdir)
    cmd = [
        sys.executable, DEEPLINK_PY,
        "-X", str(X_path),
        "-y", str(y_path),
        "-o", str(outdir),
        "-q", str(q),
        "-l", str(l1),
        "-r", str(lr),
        "-a", act,
        "-L", loss
    ]
    if not center_and_scale:
        cmd.append("-s")  # -s を付けると「中心化・標準化をしない」
    print("[DeepLINK]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    (outdir / "deeplink_stdout.txt").write_text(res.stdout or "")
    (outdir / "deeplink_stderr.txt").write_text(res.stderr or "")
    if res.returncode != 0:
        raise RuntimeError(f"DeepLINK failed. See logs in {outdir}")

def read_selected_indices(outdir: Path):
    """
    DeepLINKは selected_variable_ko / selected_variable_ko+ の2ファイルを吐く。
    まず ko+ を優先して読み、無ければ ko を読む。
    """
    for name in ["selected_variable_ko+", "selected_variable_ko"]:
        p = outdir / name
        if p.exists():
            # スペース区切りの 0-based インデックス（一行 or 複数行）
            txt = p.read_text().strip()
            if not txt:
                return np.array([], dtype=int)
            # 空白と改行で分割
            toks = txt.split()
            return np.array(list(map(int, toks)), dtype=int)
    # 両方無ければ空集合扱い
    return np.array([], dtype=int)

def fdr_and_power(selected: np.ndarray, true_idx: np.ndarray, n: int):
    """FDR = FP / max(R,1), Power = TP / |S_true|"""
    S = set(map(int, selected.tolist()))
    T = set(map(int, np.array(true_idx, dtype=int).tolist()))
    R = len(S)
    TP = len(S & T)
    FP = R - TP
    fdr = FP / max(R, 1)
    power = TP / max(len(T), 1)
    return fdr, power, R, TP, FP

def run_one(seed: int, scenario: str):
    # 1) データ生成（あなたのDGPをそのまま利用）
    X_full, _, _ = generate_design_with_splits(m=M, n=N, scenario=scenario, seed=seed,
                                               r_spike=2, var_alpha=[0.5], var_burnin=200)
    y_full, true_idx, B = generate_y_given_X(X_full, b1_scale=B1_SCALE, q_star=8, rng=seed+12345)

    # 2) 保存 & DeepLINK 実行
    outdir = OUT_ROOT / scenario / f"seed_{seed:04d}"
    ensure_dir(outdir)
    save_npy(X_full, y_full, outdir)
    run_deeplink(outdir / "X.npy", outdir / "y.npy", outdir)

    # 3) DeepLINK 出力の読み込み & 評価
    selected = read_selected_indices(outdir)
    fdr, power, R, TP, FP = fdr_and_power(selected, true_idx, n=N)

    # ログ保存
    rec = dict(seed=seed, scenario=scenario, n=N, m=M,
               q=Q_FDR, b1_scale=B1_SCALE,
               R=R, TP=TP, FP=FP, FDR=fdr, Power=power,
               n_selected=len(selected))
    (outdir / "metrics.json").write_text(json.dumps(rec, indent=2))
    return rec

def main():
    all_rows = []
    for sc in SCENARIOS:
        for sd in SEEDS:
            try:
                rec = run_one(sd, sc)
                all_rows.append(rec)
                print(f"[OK] scenario={sc} seed={sd} -> FDR={rec['FDR']:.3f}, Power={rec['Power']:.3f}, R={rec['R']}")
            except Exception as e:
                print(f"[FAIL] scenario={sc} seed={sd}: {e}")

    # 4) 集計（シナリオごと・平均と標準偏差）
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(OUT_ROOT / "deeplink_seedwise.csv", index=False)

        summary = df.groupby("scenario").agg(
            FDR_mean=("FDR","mean"), FDR_std=("FDR","std"),
            Power_mean=("Power","mean"), Power_std=("Power","std"),
            R_mean=("R","mean"), R_std=("R","std")
        ).reset_index()
        print("\n=== DeepLINK summary (mean ± std over seeds) ===")
        print(summary.to_string(index=False))
        summary.to_csv(OUT_ROOT / "deeplink_summary.csv", index=False)
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()

