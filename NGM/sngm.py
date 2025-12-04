# sngm.py
# Simultaneous Neural Gaussian Mirror (SNGM) – PyTorch implementation (fast & observable)
# 修正点:
#  - サブサンプル近似では U,V もサブセット idx に限定して Gram を構築（m'×m'）
#  - 1D-RBF 距離を (A + c B + c^2 C) で展開して c グリッドをベクトル化
#  - 進捗ログを progress.json に書き出し（c_j と学習エポック）
#
# 返り値: dict("Mj","tau","selected","c_js","feat_idx","plus_idx","minus_idx")

import json
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------
# RBF kernels & utilities
# ------------------------
def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    # X: (n, d)
    s = np.sum(X * X, axis=1, keepdims=True)  # (n,1)
    D2 = s + s.T - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    return D2

def _pairwise_sq_diffs_1d(x: np.ndarray) -> np.ndarray:
    # x: (n,)
    # returns (n,n) with (x_i - x_j)^2
    diff = x.reshape(-1, 1) - x.reshape(1, -1)
    return diff * diff

def _pairwise_diffs_1d(x: np.ndarray) -> np.ndarray:
    # x: (n,)
    return x.reshape(-1, 1) - x.reshape(1, -1)

def _median_heuristic_sigma_from_D2(D2: np.ndarray) -> float:
    tri = D2[np.triu_indices_from(D2, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    med = np.median(tri)
    return math.sqrt(0.5 * med) if med > 0 else 1.0

def center_gram_inplace(K: np.ndarray) -> np.ndarray:
    # K <- H K H を in-place で（余分な行列生成を抑える簡易版）
    n = K.shape[0]
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    grand = K.mean()
    K -= row_mean
    K -= col_mean
    K += grand
    return K


# ----------------------------------------------------
# Kernel conditional measure [I^K_j(c)]^2 (Eq. III.5)
#   [IK_j(c)]^2 = (1/n^2) [ (H K^U H) ◦ (H K^V H) ◦ K^W ]_{++}
#   U = X_j^+, V = X_j^-, W = X_{-j}
#
# kw_precomputed:
#   - None:        exact (build KW from W)  ← 現在は重いので非推奨
#   - np.ndarray:  "full" proxy Gram (m×m)   ← 近似
#   - (idx, KWsub) "subsample" proxy         ← 強い近似（推奨）
# sigma_mode:
#   - "fixed":  U,V ともに sigma を xj のみから固定推定（速い）
#   - "per_c":  各 c ごとに sigma を再推定（精密だが遅い）
# ----------------------------------------------------
def kernel_conditional_measure_subsample_fast(
    x_sub: np.ndarray, z_sub: np.ndarray, KW_sub: np.ndarray,
    c_grid: np.ndarray, sigma_mode: str = "fixed"
) -> Tuple[np.ndarray, float]:
    """
    サブサンプル idx 上でのみ U,V の Gram を構築し、全 c を一括評価。
    戻り値: values (len(c_grid),), best_val
    """
    # dU(c) = (x_i - x_j) + c (z_i - z_j)
    # dV(c) = (x_i - x_j) - c (z_i - z_j)
    dX = _pairwise_diffs_1d(x_sub)   # (m',m')
    dZ = _pairwise_diffs_1d(z_sub)   # (m',m')
    A = dX * dX                      # (x_i-x_j)^2
    B = 2.0 * dX * dZ                # 2 (x_i-x_j)(z_i-z_j)
    C = dZ * dZ                      # (z_i-z_j)^2

    # sigma を固定（中値ヒューリスティックを x_sub のみで）
    if sigma_mode == "fixed":
        sigma = _median_heuristic_sigma_from_D2(A)
        inv2sigma2 = 1.0 / (2.0 * sigma * sigma)

    n_eff = x_sub.shape[0]
    vals = np.empty(len(c_grid), dtype=float)

    for k, c in enumerate(c_grid):
        # U: A + c B + c^2 C,  V: A - c B + c^2 C
        if sigma_mode == "per_c":
            # U
            D2U = A + c * B + (c * c) * C
            sigmaU = _median_heuristic_sigma_from_D2(D2U)
            KU = np.exp(-D2U / (2.0 * sigmaU * sigmaU))
            center_gram_inplace(KU)
            # V
            D2V = A - c * B + (c * c) * C
            sigmaV = _median_heuristic_sigma_from_D2(D2V)
            KV = np.exp(-D2V / (2.0 * sigmaV * sigmaV))
            center_gram_inplace(KV)
        else:
            D2U = A + c * B + (c * c) * C
            D2V = A - c * B + (c * c) * C
            KU = np.exp(-D2U * inv2sigma2)
            KV = np.exp(-D2V * inv2sigma2)
            center_gram_inplace(KU)
            center_gram_inplace(KV)

        vals[k] = float((KU * KV * KW_sub).sum() / (n_eff * n_eff))

    return vals, float(vals.min())


def kernel_conditional_measure_exact(
    U: np.ndarray, V: np.ndarray, W: np.ndarray
) -> float:
    # 厳密版（遅い）：各 j ごとに KW(W) を作る
    # 実務で使う場合は "subsample" を推奨
    def rbf_gram_1d(x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        diff2 = _pairwise_sq_diffs_1d(x)
        if sigma is None:
            sigma = _median_heuristic_sigma_from_D2(diff2)
        return np.exp(-diff2 / (2.0 * sigma * sigma))

    KU = rbf_gram_1d(U); center_gram_inplace(KU)
    KV = rbf_gram_1d(V); center_gram_inplace(KV)
    # W は多次元
    D2W = _pairwise_sq_dists(W)
    sigmaW = _median_heuristic_sigma_from_D2(D2W)
    KW = np.exp(-D2W / (2.0 * sigmaW * sigmaW))
    n = KU.shape[0]
    return float((KU * KV * KW).sum() / (n * n))


# ------------------------
# Simple MLP (2 hidden)
# ------------------------
class MLP2(nn.Module):
    def __init__(self, din: int, h1: int, h2: int, act: str = "relu"):
        super().__init__()
        acts = {"relu": nn.ReLU(), "elu": nn.ELU(), "tanh": nn.Tanh()}
        self.fc1 = nn.Linear(din, h1)
        self.a1  = acts.get(act, nn.ReLU())
        self.fc2 = nn.Linear(h1, h2)
        self.a2  = acts.get(act, nn.ReLU())
        self.out = nn.Linear(h2, 1)

    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        return self.out(x)


# -------------------------------------------------
# Connection-weights importance (Eq. III.6, single-output)
# -------------------------------------------------
@torch.no_grad()
def connection_importance_first_layer(model: MLP2) -> Tuple[np.ndarray, np.ndarray]:
    W0 = model.fc1.weight.detach().cpu().numpy().T  # (din, h1)
    W1 = model.fc2.weight.detach().cpu().numpy().T  # (h1, h2)
    W2 = model.out.weight.detach().cpu().numpy().T  # (h2, 1)
    cvec = (W1 @ W2).reshape(-1)                    # (h1,)
    return W0, cvec


# -------------------------------------------
# SNGM main
# -------------------------------------------
@dataclass
class SNGMConfig:
    q_fdr: float = 0.1
    act: str = "relu"
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 60
    screen_k: Optional[int] = None         # e.g., p//2
    approx_kw: str = "subsample:800"       # "none" | "full" | "subsample:<m_sub>"
    c_grid_num: int = 6
    verbose: bool = True
    log_every_epoch: int = 10
    sigma_mode: str = "fixed"              # "fixed" | "per_c"
    progress_path: Optional[str] = None    # write heartbeat JSON if set


def _write_progress(cfg: SNGMConfig, payload: dict):
    if cfg.progress_path is None:
        return
    try:
        with open(cfg.progress_path, "w") as f:
            json.dump(payload, f)
    except Exception:
        pass


def sngm_select(X: np.ndarray, y: np.ndarray, cfg: SNGMConfig,
                device: Optional[torch.device] = None) -> dict:
    """
    X: (m, p), y: (m,)
    returns dict with keys: 'Mj', 'tau', 'selected', 'c_js', 'feat_idx', 'plus_idx', 'minus_idx'
    """
    t_all0 = perf_counter()
    m, p = X.shape
    y = y.reshape(-1).astype(np.float32)
    feat_idx = np.arange(p)
    X_work = X

    # optional screening
    if cfg.screen_k is not None and cfg.screen_k < p:
        if cfg.verbose:
            print(f"[SNGM] screening: p={p} -> k={cfg.screen_k}")
        corrs = np.empty(p, dtype=float)
        for j in range(p):
            cj = np.corrcoef(X[:, j], y)[0, 1]
            corrs[j] = abs(cj) if np.isfinite(cj) else 0.0
        sel = np.argsort(-corrs)[: cfg.screen_k]
        X_work = X[:, sel]
        feat_idx = sel
        p = X_work.shape[1]

    if cfg.verbose:
        print(f"[SNGM] m={m}, p(after screen)={p}, approx_kw={cfg.approx_kw}, c_grid={cfg.c_grid_num}")

    # precompute KW proxy
    t0 = perf_counter()
    kw_proxy: Union[None, np.ndarray, Tuple[np.ndarray, np.ndarray]]
    kw_proxy = None
    subs_idx = None
    if cfg.approx_kw == "full":
        # full Gram on all rows
        D2 = _pairwise_sq_dists(X_work)
        sigmaW = _median_heuristic_sigma_from_D2(D2)
        kw_proxy = np.exp(-D2 / (2.0 * sigmaW * sigmaW))
        if cfg.verbose:
            print(f"[SNGM] precomputed KW(full) in {perf_counter()-t0:.2f}s")
    elif cfg.approx_kw.startswith("subsample:"):
        try:
            m_sub = int(cfg.approx_kw.split(":")[1])
        except Exception:
            m_sub = min(m, 800)
        rng = np.random.default_rng(0)
        subs_idx = np.sort(rng.choice(m, size=min(m, m_sub), replace=False))
        X_sub = X_work[subs_idx]
        D2 = _pairwise_sq_dists(X_sub)
        sigmaW = _median_heuristic_sigma_from_D2(D2)
        KW_sub = np.exp(-D2 / (2.0 * sigmaW * sigmaW))
        kw_proxy = (subs_idx, KW_sub)
        if cfg.verbose:
            print(f"[SNGM] precomputed KW(subsample m'={len(subs_idx)}) in {perf_counter()-t0:.2f}s")
    elif cfg.approx_kw == "none":
        if cfg.verbose:
            print(f"[SNGM] using exact KW per feature (slow)")

    # choose c_j and build mirror pairs
    rng = np.random.default_rng(12345)
    Z = rng.standard_normal(size=(m, p)).astype(np.float32)
    c_grid_base = np.linspace(0.25, 2.0, num=cfg.c_grid_num)  # scale by std(xj)
    c_js = np.zeros(p, dtype=np.float32)
    X_plus = np.zeros_like(X_work, dtype=np.float32)
    X_minus = np.zeros_like(X_work, dtype=np.float32)

    t_c = perf_counter()
    for j in range(p):
        tj = perf_counter()
        xj = X_work[:, j].astype(np.float32)
        std = np.std(xj) + 1e-8
        c_grid = std * c_grid_base
        z = Z[:, j]

        if isinstance(kw_proxy, tuple):
            # ---- サブサンプル近似：U,V も subs_idx 上だけで評価 ----
            idx, KW_sub = kw_proxy
            x_sub = xj[idx]
            z_sub = z[idx]
            vals, _ = kernel_conditional_measure_subsample_fast(
                x_sub, z_sub, KW_sub, c_grid, sigma_mode=cfg.sigma_mode
            )
            cj = float(c_grid[np.argmin(vals)])
        elif isinstance(kw_proxy, np.ndarray):
            # ---- full 近似：U,V は m×m で作ってから中心化、KW は近似を使用 ----
            # U,V の Gram を都度 m×m で計算（そこそこ重い）
            def rbf_gram_1d(x: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
                diff2 = _pairwise_sq_diffs_1d(x)
                if sigma is None:
                    sigma = _median_heuristic_sigma_from_D2(diff2)
                return np.exp(-diff2 / (2.0 * sigma * sigma))
            best_val = float("inf")
            cj = float(c_grid[0])
            for c in c_grid:
                U = xj + c * z
                V = xj - c * z
                KU = rbf_gram_1d(U); center_gram_inplace(KU)
                KV = rbf_gram_1d(V); center_gram_inplace(KV)
                n = KU.shape[0]
                val = float((KU * KV * kw_proxy).sum() / (n * n))
                if val < best_val:
                    best_val, cj = val, float(c)
        else:
            # ---- exact（非常に重いので推奨しない）----
            best_val = float("inf")
            cj = float(c_grid[0])
            for c in c_grid:
                U = xj + c * z
                V = xj - c * z
                W = np.delete(X_work, j, axis=1)
                val = kernel_conditional_measure_exact(U, V, W)
                if val < best_val:
                    best_val, cj = val, float(c)

        c_js[j] = cj
        X_plus[:, j]  = xj + cj * z
        X_minus[:, j] = xj - cj * z

        if cfg.verbose and (j % 5 == 0 or j == p - 1):
            print(f"[SNGM] c_j done: j={j}/{p-1} in {perf_counter()-tj:.2f}s (best c={cj:.3g})")
        _write_progress(cfg, {
            "stage": "choose_c",
            "j_done": j + 1, "p_total": p,
            "elapsed_sec": round(perf_counter() - t_all0, 2)
        })

    if cfg.verbose:
        print(f"[SNGM] c_j stage total {perf_counter()-t_c:.2f}s")

    # prepare SNGM input: [x1+, x1-, x2+, x2-, ...]
    Xm = np.empty((m, 2 * p), dtype=np.float32)
    plus_idx = []
    minus_idx = []
    for j in range(p):
        Xm[:, 2 * j]     = X_plus[:, j];   plus_idx.append(2 * j)
        Xm[:, 2 * j + 1] = X_minus[:, j];  minus_idx.append(2 * j + 1)

    # train once
    din = 2 * p
    h1 = max(16, int(20 * math.log(p + 1.0)))
    h2 = max(8,  int(10 * math.log(p + 1.0)))
    model = MLP2(din, h1, h2, act=cfg.act)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    X_tensor = torch.from_numpy(Xm).to(device)
    y_tensor = torch.from_numpy(y).to(device).view(-1, 1)
    ds = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    if cfg.verbose:
        print(f"[SNGM] start training: epochs={cfg.epochs}, batch={cfg.batch_size}, act={cfg.act}")
    for ep in range(cfg.epochs):
        et = perf_counter()
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        if cfg.verbose and ((ep + 1) % cfg.log_every_epoch == 0 or (ep + 1) == cfg.epochs):
            print(f"[SNGM] epoch {ep+1}/{cfg.epochs} done in {perf_counter()-et:.2f}s")
        _write_progress(cfg, {
            "stage": "train",
            "epoch": ep + 1, "epochs": cfg.epochs,
            "elapsed_sec": round(perf_counter() - t_all0, 2)
        })

    # connection-weights importance
    W0, cvec = connection_importance_first_layer(model)
    L_all = W0 @ cvec
    L_plus  = L_all[plus_idx]
    L_minus = L_all[minus_idx]

    # mirror statistics (Eq. III.7)
    Mj = np.abs(L_plus + L_minus) - np.abs(L_plus - L_minus)

    # FDP thresholding (Eq. II.3, II.4)
    def _fdp(t: float) -> float:
        num = int(np.sum(Mj <= -t))
        den = max(1, int(np.sum(Mj >= t)))
        return num / den

    ts = np.unique(np.abs(Mj))
    ts.sort()
    tau = None
    for t in ts:
        if _fdp(float(t)) <= cfg.q_fdr:
            tau = float(t)
            break
    if tau is None:
        tau = float(ts[-1]) if len(ts) > 0 else 0.0

    selected_local = np.where(Mj >= tau)[0]
    selected = feat_idx[selected_local]

    if cfg.verbose:
        print(f"[SNGM] threshold tau={tau:.4g}, selected={len(selected)}")
    _write_progress(cfg, {
        "stage": "done",
        "tau": tau, "n_selected": int(len(selected)),
        "elapsed_sec": round(perf_counter() - t_all0, 2)
    })

    return {
        "Mj": Mj,
        "tau": tau,
        "selected": selected,
        "c_js": c_js,
        "feat_idx": feat_idx,
        "plus_idx": np.array(plus_idx),
        "minus_idx": np.array(minus_idx),
    }
