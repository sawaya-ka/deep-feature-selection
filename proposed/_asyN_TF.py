# ============================================================
# Unified FS Pipeline (v2 互換の出力) + Transformer 安定化パッチ統合版
# ============================================================
import os, math, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Callable

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Array = np.ndarray

# ===========================
# Utils
# ===========================
def standardize_global(X: Array) -> Tuple[Array, Array, Array]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=0, keepdims=True)
    sigma = sigma + 1e-8
    Xn = (X - mu) / sigma
    return Xn, mu.squeeze(0), sigma.squeeze(0)

def make_lr_schedule(T: int, base_lr: float = 1e-3, warmup_ratio: float = 0.2, min_lr_ratio: float = 1.0
) -> Callable[[int], float]:
    """t=1..T: linear warmup -> cosine。min_lr_ratio=1.0 で“実質定数LR”"""
    W = max(1, int(T * warmup_ratio))
    def lr_t(t: int) -> float:
        if t <= W:
            return base_lr * (t / W)
        s = (t - W) / max(1, T - W)
        return base_lr * (min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * s)))
    return lr_t

@torch.no_grad()
def _toggle_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

# ===========================
# 入力勾配集計
# ===========================
def steinized_input_direction(model, Xn, batch_size=2048):
    model.eval(); _toggle_requires_grad(model, False)
    d = Xn.shape[1]
    acc = torch.zeros(d, device=Xn.device)
    count = 0; start = 0
    while start < Xn.shape[0]:
        end = min(start+batch_size, Xn.shape[0])
        Xb = Xn[start:end].detach().clone().requires_grad_(True)
        out = model(Xb).view(-1); s = out.sum()
        grads = torch.autograd.grad(s, Xb, retain_graph=False, create_graph=False)[0]
        acc += (Xb * grads).sum(dim=0)
        count += (end-start); start = end
    _toggle_requires_grad(model, True)
    return (acc / count).detach().cpu().numpy()

def sum_input_gradients_std(model: nn.Module, Xn: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    # RNN系は backward が eval だと動かないので train、それ以外は eval
    if any(isinstance(m, (nn.RNNBase, nn.LSTM, nn.GRU)) for m in model.modules()):
        model.train()
    else:
        model.eval()
    _toggle_requires_grad(model, False)
    nfeat = Xn.shape[1]
    xi_std = torch.zeros(nfeat, device=Xn.device)
    start = 0
    while start < Xn.shape[0]:
        end = min(start + batch_size, Xn.shape[0])
        Xb = Xn[start:end].detach().clone().requires_grad_(True)
        out = model(Xb).view(-1); s = out.sum()
        grads = torch.autograd.grad(s, Xb, retain_graph=False, create_graph=False)[0]
        xi_std += grads.sum(dim=0)
        start = end
    _toggle_requires_grad(model, True)
    return xi_std.detach().cpu().numpy()

# ===========================
# 集約/しきい値/評価
# ===========================
def psi_aggregate(a: Array, b: Array, method: str = "mean", eps: float = 1e-12) -> Array:
    if method == "min": return np.minimum(a, b)
    if method == "geomean": return np.sqrt(np.maximum(a*b, 0.0) + eps)
    if method == "mean": return 0.5 * (a + b)
    if method == "harmmean": return (2.0 * a * b) / (a + b + eps)
    raise ValueError("Unknown psi method")

def _strict_threshold_candidates(M: Array) -> Array:
    vals = M[np.isfinite(M)]
    abs_vals = np.unique(np.abs(vals)); abs_vals = abs_vals[abs_vals > 0.0]
    if abs_vals.size == 0: return np.array([], dtype=float)
    abs_vals.sort(); a = np.concatenate([[0.0], abs_vals])
    return 0.5 * (a[:-1] + a[1:])

def strict_mirror_fdp_threshold(M: Array, alpha: float) -> Tuple[float, int, int, float]:
    assert 0.0 < alpha < 1.0
    cand = _strict_threshold_candidates(M)
    if cand.size == 0: return np.inf, 0, 0, np.inf
    for u in cand:
        Rp = int((M > u).sum()); Rm = int((M < -u).sum()); den = max(Rp, 1)
        fdp = Rm / den
        if fdp <= alpha: return float(u), Rp, Rm, float(fdp)
    return np.inf, 0, 0, np.inf

def true_metrics(selected: Array, true_idx: Optional[Array], n: int) -> Tuple[Optional[float], Optional[float]]:
    if true_idx is None or len(true_idx) == 0: return None, None
    S = set(int(i) for i in selected.tolist()); T = set(int(i) for i in np.array(true_idx).tolist())
    R = len(S); TP = len(S & T); FP = R - TP; FN = len(T - S)
    FDR_true = FP / max(R, 1); TypeII = FN / max(len(T), 1)
    return float(FDR_true), float(TypeII)

# ===========================
# PathResult
# ===========================
@dataclass
class PathResult:
    steps: Array; tau: Array; R_plus: Array; R_minus: Array; FDPhat: Array
    FDR_true: Optional[Array]; TypeII: Optional[Array]
    last_xi1: Array; last_xi2: Array; last_M: Array; last_selected: Array; last_tau: float
    loss_steps: Array = field(default_factory=lambda: np.array([], dtype=int))
    loss_values: Array = field(default_factory=lambda: np.array([], dtype=float))
    xi1_hist: Array = field(default_factory=lambda: np.empty((0,0), dtype=float))
    xi2_hist: Array = field(default_factory=lambda: np.empty((0,0), dtype=float))
    M_hist:   Array = field(default_factory=lambda: np.empty((0,0), dtype=float))

# ===========================
# MLP
# ===========================
class TorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, bias_init=0.0, init_mode="he", fixed_sigma=0.01, activation="relu"):
        super().__init__()
        dims = [input_dim] + hidden_dims + [1]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=True) for i in range(len(dims)-1)])
        self.hidden_dims = hidden_dims; self.bias_init = bias_init; self.init_mode = init_mode
        self.fixed_sigma = fixed_sigma; self.activation = activation.lower()
        self.reset_parameters()
    def _act(self, z):
        a = self.activation
        if a == "relu": return F.relu(z)
        if a == "leaky_relu": return F.leaky_relu(z, negative_slope=0.01)
        if a == "elu": return F.elu(z, alpha=1.0)
        if a in ("silu", "swish"): return F.silu(z)
        if a == "gelu": return F.gelu(z)
        if a == "tanh": return torch.tanh(z)
        if a == "softplus": return F.softplus(z, beta=1.0)
        raise ValueError(f"unknown activation: {a}")
    def reset_parameters(self):
        with torch.no_grad():
            for lin in self.layers:
                din = lin.in_features
                if self.init_mode == "he": sigma = np.sqrt(2.0 / din)
                elif self.init_mode == "xavier": sigma = np.sqrt(1.0 / din)
                elif self.init_mode is None: sigma = self.fixed_sigma
                else: raise ValueError("Unsupported init_mode")
                lin.weight.normal_(0.0, float(sigma)); lin.bias.fill_(float(self.bias_init))
    def forward(self, x):
        h = x
        for i in range(len(self.hidden_dims)):
            h = self._act(self.layers[i](h))
        return self.layers[-1](h)

# ===========================
# Lifted CNN/RNN/Transformer
# ===========================
def _init_lift_weight_gaussian_(lin: nn.Linear, fan_in: int, mode: str = "he", fixed_sigma: float = 0.01):
    with torch.no_grad():
        if mode == "he": sigma = (2.0 / float(fan_in)) ** 0.5
        elif mode == "xavier": sigma = (1.0 / float(fan_in)) ** 0.5
        elif mode == "fixed": sigma = float(fixed_sigma)
        else: raise ValueError("lift_init_mode must be 'he'|'xavier'|'fixed'")
        lin.weight.normal_(0.0, sigma)
        if lin.bias is not None: lin.bias.zero_()

class TorchCNN1D_Lifted(nn.Module):
    def __init__(self, input_dim: int, lift_dim: int,
                 conv_channels: list = [64, 64], kernel_sizes: list = [11, 7], strides: list = [1, 1],
                 fc_dims: list = [128], activation: str = "relu", init_mode: str = "he",
                 bias_init: float = 0.0, dropout: float = 0.0, use_bn: bool = False,
                 lift_init_mode: str = "he", lift_fixed_sigma: float = 0.01):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(strides)
        self.activation = activation.lower(); self.init_mode = init_mode; self.bias_init = bias_init
        self.use_bn = use_bn; self.input_dim = input_dim; self.lift_dim = lift_dim
        self.W1 = nn.Linear(input_dim, lift_dim, bias=False)
        layers = []; in_ch = 1
        for c, k, s in zip(conv_channels, kernel_sizes, strides):
            pad = k // 2
            layers.append(nn.Conv1d(in_ch, c, kernel_size=k, stride=s, padding=pad, bias=True))
            if use_bn: layers.append(nn.BatchNorm1d(c))
            layers.append(self._act_layer())
            if dropout > 0: layers.append(nn.Dropout(dropout))
            in_ch = c
        self.conv = nn.Sequential(*layers); self.gap = nn.AdaptiveAvgPool1d(1)
        dims = [in_ch] + list(fc_dims) + [1]; fcs = []
        for i in range(len(dims) - 2):
            fcs.append(nn.Linear(dims[i], dims[i+1], bias=True))
            fcs.append(self._act_layer())
            if dropout > 0: fcs.append(nn.Dropout(dropout))
        fcs.append(nn.Linear(dims[-2], dims[-1], bias=True)); self.head = nn.Sequential(*fcs)
        _init_lift_weight_gaussian_(self.W1, fan_in=input_dim, mode=lift_init_mode, fixed_sigma=lift_fixed_sigma); self._init_rest()
    def _act_layer(self):
        a = self.activation
        if a == "relu": return nn.ReLU()
        if a == "leaky_relu": return nn.LeakyReLU(0.01)
        if a == "elu": return nn.ELU()
        if a in ("silu", "swish"): return nn.SiLU()
        if a == "gelu": return nn.GELU()
        if a == "tanh": return nn.Tanh()
        if a == "softplus": return nn.Softplus(beta=1.0)
        raise ValueError(f"unknown activation: {a}")
    def _init_rest(self):
        with torch.no_grad():
            for m in self.modules():
                if m is self.W1: continue
                if isinstance(m, (nn.Conv1d, nn.Linear)):
                    if self.init_mode == "he": nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    elif self.init_mode == "xavier": nn.init.xavier_normal_(m.weight)
                    else: nn.init.normal_(m.weight, 0.0, 0.01)
                    if m.bias is not None: m.bias.fill_(float(self.bias_init))
                elif isinstance(m, (nn.BatchNorm1d,)): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W1(x); y = z.unsqueeze(1); y = self.conv(y); y = self.gap(y).squeeze(-1)
        return self.head(y)

class TorchRNN_Lifted(nn.Module):
    def __init__(self, input_dim: int, lift_dim: int, rnn_type: str = "lstm",
                 hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False,
                 input_proj_dim: int = 1, fc_dims: list = [128], activation: str = "relu",
                 init_mode: str = "xavier", bias_init: float = 0.0, dropout: float = 0.0,
                 lift_init_mode: str = "xavier", lift_fixed_sigma: float = 0.01):
        super().__init__()
        self.activation = activation.lower(); self.init_mode = init_mode; self.bias_init = bias_init
        self.rnn_type = rnn_type.lower(); self.input_dim = input_dim; self.lift_dim = lift_dim
        self.W1 = nn.Linear(input_dim, lift_dim, bias=False)
        self.input_proj = nn.Identity(); rnn_input_size = 1
        if input_proj_dim > 1: self.input_proj = nn.Linear(1, input_proj_dim, bias=True); rnn_input_size = input_proj_dim
        common = dict(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                      batch_first=True, dropout=(dropout if num_layers > 1 else 0.0),
                      bidirectional=bidirectional)
        if self.rnn_type == "lstm": self.rnn = nn.LSTM(**common)
        elif self.rnn_type == "gru": self.rnn = nn.GRU(**common)
        else: raise ValueError("rnn_type must be 'lstm' or 'gru'")
        out_dim = hidden_size * (2 if bidirectional else 1)
        dims = [out_dim] + list(fc_dims) + [1]; fcs = []
        for i in range(len(dims) - 2):
            fcs.append(nn.Linear(dims[i], dims[i+1], bias=True)); fcs.append(self._act_layer())
            if dropout > 0: fcs.append(nn.Dropout(dropout))
        fcs.append(nn.Linear(dims[-2], dims[-1], bias=True)); self.head = nn.Sequential(*fcs)
        _init_lift_weight_gaussian_(self.W1, fan_in=input_dim, mode=lift_init_mode, fixed_sigma=lift_fixed_sigma); self._init_rest()
    def _act_layer(self):
        a = self.activation
        if a == "relu": return nn.ReLU()
        if a == "leaky_relu": return nn.LeakyReLU(0.01)
        if a == "elu": return nn.ELU()
        if a in ("silu", "swish"): return nn.SiLU()
        if a == "gelu": return nn.GELU()
        if a == "tanh": return nn.Tanh()
        if a == "softplus": return nn.Softplus(beta=1.0)
        raise ValueError(f"unknown activation: {a}")
    def _init_rest(self):
        with torch.no_grad():
            if isinstance(self.input_proj, nn.Linear):
                if self.init_mode == "he": nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
                elif self.init_mode == "xavier": nn.init.xavier_normal_(self.input_proj.weight)
                else: nn.init.normal_(self.input_proj.weight, 0.0, 0.01)
                if self.input_proj.bias is not None: self.input_proj.bias.fill_(float(self.bias_init))
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    if self.init_mode == "he": nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    elif self.init_mode == "xavier": nn.init.xavier_normal_(m.weight)
                    else: nn.init.normal_(m.weight, 0.0, 0.01)
                    if m.bias is not None: m.bias.fill_(float(self.bias_init))
            for name, p in self.rnn.named_parameters():
                if "weight" in name and p.dim() >= 2:
                    if self.init_mode == "he": nn.init.kaiming_normal_(p, nonlinearity="relu")
                    elif self.init_mode == "xavier": nn.init.xavier_normal_(p)
                    else: nn.init.normal_(p, 0.0, 0.01)
                elif "bias" in name: p.fill_(float(self.bias_init))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W1(x); s = z.unsqueeze(-1); s = self.input_proj(s)
        if self.rnn_type == "lstm":
            out, (h_n, c_n) = self.rnn(s)
            last_fwd = h_n[-1]
            if self.rnn.bidirectional: last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else: last = last_fwd
        else:
            out, h_n = self.rnn(s); last_fwd = h_n[-1]
            if self.rnn.bidirectional: last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else: last = last_fwd
        return self.head(last)

# ---- Positional Encoding ----
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1); return x + self.pe[:L].unsqueeze(0)

# ---- 安定化済み Transformer（Encoderはデフォ初期化を保持）----
class TorchTransformer1D_Lifted(nn.Module):
    def __init__(self, input_dim: int, lift_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 256, dropout: float = 0.1, activation: str = "relu",
                 init_mode: str = "he", bias_init: float = 0.0, pool: str = "mean",
                 use_sinusoidal_pos: bool = True, fc_dims: list = [128],
                 lift_init_mode: str = "he", lift_fixed_sigma: float = 0.01):
        super().__init__()
        assert d_model % nhead == 0
        self.pool = pool.lower()
        self.head_activation = activation.lower()
        self.input_dim = input_dim
        self.lift_dim = lift_dim
        # --- Linear lift ---
        self.W1 = nn.Linear(input_dim, lift_dim, bias=False)
        # --- in-proj, input LayerNorm, pos ---
        self.in_proj = nn.Linear(1, d_model, bias=True)
        self.input_ln = nn.LayerNorm(d_model)  # ★ 追加：埋め込みの振れ止め
        self.pos = SinusoidalPositionalEncoding(d_model) if use_sinusoidal_pos else nn.Embedding(10000, d_model)
        # --- Encoder（初期化は触らない）---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        if self.pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        # --- head ---
        dims = [d_model] + list(fc_dims) + [1]; fcs = []
        for i in range(len(dims) - 2):
            fcs.append(nn.Linear(dims[i], dims[i+1], bias=True))
            fcs.append(self._act_layer())
            if dropout > 0: fcs.append(nn.Dropout(dropout))
        fcs.append(nn.Linear(dims[-2], dims[-1], bias=True)); self.head = nn.Sequential(*fcs)
        # --- init: W1, in_proj, head のみ適度に初期化（Encoderはそのまま）---
        _init_lift_weight_gaussian_(self.W1, fan_in=input_dim, mode=lift_init_mode, fixed_sigma=lift_fixed_sigma)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None: self.in_proj.bias.zero_()
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: m.bias.zero_()
    def _act_layer(self):
        a = self.head_activation
        if a == "relu": return nn.ReLU()
        if a == "leaky_relu": return nn.LeakyReLU(0.01)
        if a == "elu": return nn.ELU()
        if a in ("silu", "swish"): return nn.SiLU()
        if a == "gelu": return nn.GELU()
        if a == "tanh": return nn.Tanh()
        if a == "softplus": return nn.Softplus(beta=1.0)
        raise ValueError(f"unknown activation: {a}")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W1(x)              # (B,q)
        s = z.unsqueeze(-1)         # (B,q,1)
        s = self.in_proj(s)         # ★ sqrt(d_model) は掛けない
        s = self.input_ln(s)
        if isinstance(self.pos, SinusoidalPositionalEncoding):
            s = self.pos(s)
        else:
            B, L, _ = s.shape; pos_ids = torch.arange(L, device=s.device).unsqueeze(0).expand(B, L)
            s = s + self.pos(pos_ids)
        if self.pool == "cls":
            B = s.shape[0]; cls = self.cls.expand(B, -1, -1); s = torch.cat([cls, s], dim=1)
        h = self.encoder(s)
        pooled = h[:, 0, :] if self.pool == "cls" else h.mean(dim=1)
        return self.head(pooled).view(-1, 1)

# ===========================
# 学習 1 ステップ（clip_max_norm 付き・NaN ガード）
# ===========================
def one_sgd_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                 optimizer: torch.optim.Optimizer, loss_type: str,
                 batch_size: int, g: torch.Generator,
                 clip_max_norm: float = 1.0) -> float:
    m = X.shape[0]
    idx = torch.randint(low=0, high=m, size=(min(batch_size, m),), generator=g, device=X.device)
    Xb = X.index_select(0, idx); yb = y.index_select(0, idx).view(-1, 1)
    model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(Xb)  # (B,1)
    if loss_type == "mse": loss = 0.5 * torch.mean((logits - yb)**2)
    elif loss_type == "logistic": loss = F.binary_cross_entropy_with_logits(logits, yb)
    else: raise ValueError("loss_type must be 'mse' or 'logistic'")
    loss.backward()
    if clip_max_norm is not None and clip_max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
    optimizer.step()
    val = float(loss.detach().cpu().item())
    if not np.isfinite(val):
        for g in optimizer.param_groups: g["lr"] *= 0.25
    return val

# ===========================
# 汎用 Path（v2 互換：y標準化・LRスケジュール・clip 対応）
# ===========================
def torch_nn_feature_selection_path_generic(
    X: Array, y: Array, alpha: float, T: int, build_model: Callable[[int], nn.Module],
    true_idx: Optional[Array] = None, batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.0,
    loss: str = "mse", psi_method: str = "min", seed_split: int = 2025, seed_model1: int = 11, seed_model2: int = 22,
    compute_every: int = 10, xi_batch: int = 2048, xi_mode: str = "sumgrad",
    lr_schedule: Optional[Callable[[int], float]] = None,
    optimizer: str = "sgd", momentum: float = 0.0, clip_max_norm: float = 1.0,
) -> PathResult:
    assert 0.0 < alpha < 1.0 and T >= 1
    m, n = X.shape
    Xn, mu, sigma = standardize_global(X)
    # y 標準化（スケール安定）
    y_mu = y.mean(); y_std = y.std() + 1e-12; y_tilde = (y - y_mu) / y_std

    rng_np = np.random.default_rng(seed_split)
    perm = rng_np.permutation(m); m1 = m // 2
    idx1 = perm[:m1]; idx2 = perm[m1:]
    X1n = torch.tensor(Xn[idx1], dtype=torch.float32, device=device)
    X2n = torch.tensor(Xn[idx2], dtype=torch.float32, device=device)
    y1 = torch.tensor(y_tilde[idx1], dtype=torch.float32, device=device)
    y2 = torch.tensor(y_tilde[idx2], dtype=torch.float32, device=device)

    torch.manual_seed(seed_model1); model1 = build_model(n).to(device)
    torch.manual_seed(seed_model2); model2 = build_model(n).to(device)

    opt1: torch.optim.Optimizer
    opt2: torch.optim.Optimizer
    if optimizer.lower() == "adamw":
        opt1 = torch.optim.AdamW(model1.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8)
    elif optimizer.lower() == "sgd":
        opt1 = torch.optim.SGD(model1.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=(momentum>0))
        opt2 = torch.optim.SGD(model2.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=(momentum>0))
    else:
        raise ValueError("optimizer must be 'sgd' or 'adamw'")

    if lr_schedule is None:
        def lr_schedule(t: int, _lr=lr): return _lr
    g1 = torch.Generator(device=device).manual_seed(seed_model1 + 1)
    g2 = torch.Generator(device=device).manual_seed(seed_model2 + 1)

    steps, taus, Rps, Rms, FDPs = [], [], [], [], []
    FDRs, T2s = [], []
    loss_steps, loss_values = [], []; loss_accum, loss_count = 0.0, 0
    xi1_list, xi2_list, M_list = [], [], []
    last_xi1 = last_xi2 = last_M = None; last_selected = np.array([], dtype=int); last_tau = np.inf

    for t in range(1, T+1):
        lr_now = lr_schedule(t)
        for g in opt1.param_groups: g["lr"] = lr_now
        for g in opt2.param_groups: g["lr"] = lr_now

        l1 = one_sgd_step(model1, X1n, y1, opt1, loss, batch_size, g1, clip_max_norm=clip_max_norm)
        l2 = one_sgd_step(model2, X2n, y2, opt2, loss, batch_size, g2, clip_max_norm=clip_max_norm)
        loss_accum += 0.5 * (l1 + l2); loss_count += 1

        if (t % compute_every) == 0 or (t == T):
            loss_steps.append(t); loss_values.append(loss_accum / max(loss_count, 1))
            loss_accum, loss_count = 0.0, 0

            if xi_mode == "sumgrad":
                xi1_std = sum_input_gradients_std(model2, X1n, batch_size=xi_batch)
                xi2_std = sum_input_gradients_std(model1, X2n, batch_size=xi_batch)
            elif xi_mode == "steinized":
                xi1_std = steinized_input_direction(model2, X1n, batch_size=xi_batch)
                xi2_std = steinized_input_direction(model1, X2n, batch_size=xi_batch)
            else: raise ValueError("xi_mode must be 'sumgrad' or 'steinized'")

            xi1 = xi1_std / sigma; xi2 = xi2_std / sigma
            sgn = np.sign(xi1 * xi2); mag = psi_aggregate(np.abs(xi1), np.abs(xi2), method=psi_method)
            M = sgn * mag

            # ★ 履歴の append を忘れずに
            xi1_list.append(xi1.copy()); xi2_list.append(xi2.copy()); M_list.append(M.copy())

            tau, Rp, Rm, FDPhat = strict_mirror_fdp_threshold(M, alpha=alpha)
            selected = np.where(M > tau)[0] if np.isfinite(tau) else np.array([], dtype=int)
            FDR_true, TypeII = true_metrics(selected, true_idx, n)
            steps.append(t); taus.append(tau); Rps.append(Rp); Rms.append(Rm); FDPs.append(FDPhat)
            FDRs.append(FDR_true if FDR_true is not None else np.nan); T2s.append(TypeII if TypeII is not None else np.nan)
            last_xi1, last_xi2, last_M = xi1, xi2, M; last_selected = selected; last_tau = tau

    xi1_hist = np.stack(xi1_list, axis=0) if xi1_list else np.empty((0,0))
    xi2_hist = np.stack(xi2_list, axis=0) if xi2_list else np.empty((0,0))
    M_hist   = np.stack(M_list,   axis=0) if M_list   else np.empty((0,0))
    return PathResult(np.array(steps,int), np.array(taus,float), np.array(Rps,int), np.array(Rms,int),
                      np.array(FDPs,float),
                      np.array(FDRs,float) if (true_idx is not None and len(true_idx)>0) else None,
                      np.array(T2s,float) if (true_idx is not None and len(true_idx)>0) else None,
                      last_xi1, last_xi2, last_M, last_selected, float(last_tau),
                      np.array(loss_steps,int), np.array(loss_values,float), xi1_hist, xi2_hist, M_hist)

# ===========================
# v2 互換 Path（MLP/CNN/RNN/Transformer ラッパ）
# ===========================
def torch_nn_feature_selection_path(
    X: Array, y: Array, alpha: float, T: int,
    true_idx: Optional[Array] = None,
    hidden_dims: List[int] = [64, 64],
    batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.0,
    loss: str = "mse", psi_method: str = "min",
    seed_split: int = 2025, seed_model1: int = 11, seed_model2: int = 22,
    compute_every: int = 10, xi_batch: int = 2048,
    activation: str = "relu", init_mode: str = "he",
    lr_schedule: Optional[Callable[[int], float]] = None,
):
    n = X.shape[1]
    def _build(n_in: int) -> nn.Module:
        return TorchMLP(n_in, hidden_dims, bias_init=0.0, init_mode=init_mode, activation=activation)
    # MLP は v1 の感触に合わせて SGD, lrは呼び出し引数をそのまま
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode="sumgrad",
        lr_schedule=lr_schedule, optimizer="sgd", momentum=0.0, clip_max_norm=1.0
    )

def torch_cnn1d_feature_selection_path(
    X, y, alpha, T,
    true_idx=None,
    # --- lift ---
    lift_dim: int | None = None,
    lift_init_mode: str = "he",
    lift_fixed_sigma: float = 0.01,
    # --- CNN hyper ---
    conv_channels=[64, 64],
    kernel_sizes=[11, 7],
    strides=[1, 1],
    fc_dims=[128],
    activation="relu",
    init_mode="he",
    bias_init=0.0,
    dropout=0.0,
    use_bn=False,
    # --- training ---
    batch_size=128, lr=5e-3, weight_decay=0.0,          # ★ v1 相当へ復帰
    loss="mse", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22,
    compute_every=10, xi_batch=2048, xi_mode="sumgrad",
    lr_schedule: Optional[Callable[[int], float]] = None,
):
    n = X.shape[1]
    q = n if lift_dim is None else int(lift_dim)
    def _build(n_in: int) -> nn.Module:
        return TorchCNN1D_Lifted(
            input_dim=n_in, lift_dim=q,
            conv_channels=conv_channels, kernel_sizes=kernel_sizes, strides=strides,
            fc_dims=fc_dims, activation=activation, init_mode=init_mode, bias_init=bias_init,
            dropout=dropout, use_bn=use_bn,
            lift_init_mode=lift_init_mode, lift_fixed_sigma=lift_fixed_sigma
        )
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode,
        lr_schedule=lr_schedule, optimizer="sgd", momentum=0.0, clip_max_norm=1.0
    )

def torch_rnn_feature_selection_path(
    X, y, alpha, T,
    true_idx=None,
    # --- lift ---
    lift_dim: int | None = None,
    lift_init_mode: str = "xavier",
    lift_fixed_sigma: float = 0.01,
    # --- RNN hyper ---
    rnn_type="lstm", hidden_size=128, num_layers=2, bidirectional=True,
    input_proj_dim=4, fc_dims=[64],
    activation="relu", init_mode="xavier", bias_init=0.0, dropout=0.1,
    # --- training ---
    batch_size=128, lr=5e-3, weight_decay=0.0,          # ★ v1 相当へ復帰
    loss="mse", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22,
    compute_every=10, xi_batch=512, xi_mode="sumgrad",
    lr_schedule: Optional[Callable[[int], float]] = None,
):
    n = X.shape[1]
    q = n if lift_dim is None else int(lift_dim)
    def _build(n_in: int) -> nn.Module:
        return TorchRNN_Lifted(
            input_dim=n_in, lift_dim=q,
            rnn_type=rnn_type, hidden_size=hidden_size, num_layers=num_layers,
            bidirectional=bidirectional, input_proj_dim=input_proj_dim,
            fc_dims=fc_dims, activation=activation, init_mode=init_mode, bias_init=bias_init, dropout=dropout,
            lift_init_mode=lift_init_mode, lift_fixed_sigma=lift_fixed_sigma
        )
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode,
        lr_schedule=lr_schedule, optimizer="sgd", momentum=0.0, clip_max_norm=1.0
    )

def torch_transformer_feature_selection_path(
    X, y, alpha, T, true_idx=None,
    # --- lift ---
    lift_dim: int | None = None,                   # 推奨: min(n, 512)
    # --- Transformer hyper ---
    d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1,
    activation="gelu", init_mode="he", bias_init=0.0, pool="mean", use_sinusoidal_pos=True, fc_dims=[128],
    # --- training ---
    batch_size=256, lr=3e-4, weight_decay=0.01,    # ★ AdamW 推奨設定
    loss="mse", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22,
    compute_every=10, xi_batch=512, xi_mode="sumgrad",
    lr_schedule: Optional[Callable[[int], float]] = None,
):
    n = X.shape[1]
    q = n if lift_dim is None else int(lift_dim)
    def _build(n_in: int) -> nn.Module:
        return TorchTransformer1D_Lifted(
            input_dim=n_in, lift_dim=q,
            d_model=d_model, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, init_mode=init_mode, bias_init=bias_init,
            pool=pool, use_sinusoidal_pos=use_sinusoidal_pos, fc_dims=fc_dims,
            lift_init_mode="he", lift_fixed_sigma=0.01
        )
    if lr_schedule is None:
        lr_schedule = make_lr_schedule(T, base_lr=lr, warmup_ratio=0.1, min_lr_ratio=1.0)  # 実質定数
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode,
        lr_schedule=lr_schedule, optimizer="adamw", momentum=0.0, clip_max_norm=1.0
    )

# ===========================
# DGP（そのまま）
# ===========================
def generate_sim_data(m: int, n: int, rng=None):
    q_star = 8
    if n < q_star: raise ValueError(f"n must be >= {q_star}, got n={n}")
    rng = np.random.default_rng(rng)
    X = rng.standard_normal((m, n))
    half = n // 2; b1 = np.zeros(n); b1[:half] = 2.0 / np.sqrt(n)
    B = np.zeros((n, q_star)); B[:, 0] = b1
    for k in range(2, q_star + 1): B[k-1, k-1] = 1.0
    def g(x): return (x - 2.0)**2
    def h(x): return np.maximum(x, 0.0)
    z1 = X @ B[:, 0]; y = g(z1)
    for k in range(2, q_star + 1):
        z_k = X @ B[:, k-1]; z_prev = X @ B[:, k-2]; y += h(z_k) * z_prev
    y += rng.standard_normal(m)
    idx_from_b1 = np.arange(half, dtype=int); idx_from_pairs = np.arange(q_star, dtype=int)
    true_idx = np.unique(np.concatenate([idx_from_b1, idx_from_pairs]))
    return X, y, true_idx, B

# ===========================
# Demo（必要なら有効化）
# ===========================


    print("done",
          len(path_mlp.steps), len(path_cnn.steps), len(path_rnn.steps), len(path_tr.steps))

# ===========================
# Demo（正規化ガード付き）
# ===========================

mn_list = [[4000, 1000], [200, 1000], [20, 1000]]
mlp_runs, cnn_runs, rnn_runs, tr_runs = [], [], [], []

for i in range(3):
    m, n = mn_list[i]
    X, y, true_idx, B = generate_sim_data(m=m, n=n, rng=2)
    alpha = 0.1; T = 10
    PBperp = np.eye(n) - B@np.linalg.inv(B.T@B)@B.T

    path = torch_nn_feature_selection_path(
        X, y, alpha=0.1, T=T,
        true_idx=true_idx,
        hidden_dims=[1024,1024,512,128],
        batch_size=256, lr=1e-3, weight_decay=0.0,
        loss="mse", psi_method="mean",
        seed_split=2025, seed_model1=111, seed_model2=222,
        compute_every=10, xi_batch=2048, activation="relu", init_mode="he",
        lr_schedule=None
    )

    # CNN（v1相当のLRへ）
    path_cnn = torch_cnn1d_feature_selection_path(
        X, y, alpha=0.1, T=T,
        true_idx=true_idx,
        conv_channels=[64,128,128], kernel_sizes=[11,9,7], strides=[1,1,1],
        fc_dims=[128,64], activation="relu", init_mode="he", dropout=0.1, use_bn=False,
        batch_size=128, lr=5e-3, weight_decay=0.0, loss="mse", psi_method="mean",
        seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=2048,
        xi_mode="sumgrad", lr_schedule=None
    )

    # RNN（v1相当のLRへ）
    path_rnn = torch_rnn_feature_selection_path(
        X, y, alpha=0.1, T=T,
        true_idx=true_idx,
        rnn_type="lstm", hidden_size=128, num_layers=2, bidirectional=True,
        input_proj_dim=4, fc_dims=[64], activation="relu", init_mode="xavier", dropout=0.1,
        batch_size=128, lr=5e-3, weight_decay=0.0, loss="mse", psi_method="mean",
        seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=512,
        xi_mode="sumgrad", lr_schedule=None
    )

    # Transformer（安定版：AdamW + 入力LayerNorm + lift_dim短縮推奨）
    path_tr = torch_transformer_feature_selection_path(
        X, y, alpha=0.1, T=T,
        true_idx=true_idx,
        lift_dim=min(n, 512),                      # ★ 推奨
        d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1,
        activation="gelu", init_mode="he", pool="mean", use_sinusoidal_pos=True, fc_dims=[128],
        batch_size=256, lr=3e-4, weight_decay=0.01, loss="mse", psi_method="mean",
        seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=512,
        xi_mode="sumgrad", lr_schedule=None
    )
    xi_mlp, xi_cnn, xi_rnn, xi_tr = path.last_xi1, path_cnn.last_xi1, path_rnn.last_xi1, path_tr.last_xi1
    del path; torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    del path_cnn; torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    del path_rnn; torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    del path_tr; torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    c_mlp, c_cnn, c_rnn, c_tr = np.sqrt((PBperp@xi_mlp)@(PBperp@xi_mlp)), np.sqrt((PBperp@xi_cnn)@(PBperp@xi_cnn)), np.sqrt((PBperp@xi_rnn)@(PBperp@xi_rnn)), np.sqrt((PBperp@xi_tr)@(PBperp@xi_tr))
    norm_mlp, norm_cnn, norm_rnn, norm_tr = xi_mlp/c_mlp * np.sqrt(n), xi_cnn/c_cnn * np.sqrt(n), xi_rnn/c_rnn * np.sqrt(n), xi_tr/c_tr * np.sqrt(n)
    null_mlp, null_cnn, null_rnn, null_tr = np.delete(norm_mlp,true_idx), np.delete(norm_cnn,true_idx), np.delete(norm_rnn,true_idx), np.delete(norm_tr,true_idx)
    mlp_runs.append(np.asarray(null_mlp).ravel().copy())
    cnn_runs.append(np.asarray(null_cnn).ravel().copy())
    rnn_runs.append(np.asarray(null_rnn).ravel().copy())
    tr_runs.append(np.asarray(null_tr).ravel().copy())
    print(i)

M_mlp = np.vstack(mlp_runs)
M_cnn = np.vstack(cnn_runs)
M_rnn = np.vstack(rnn_runs)
M_tr  = np.vstack(tr_runs)

np.savetxt("res/_mat_mlp.csv", M_mlp, delimiter=",")
np.savetxt("res/_mat_cnn.csv", M_cnn, delimiter=",")
np.savetxt("res/_mat_rnn.csv", M_rnn, delimiter=",")
np.savetxt("res/_mat_tr.csv",  M_tr,  delimiter=",")


