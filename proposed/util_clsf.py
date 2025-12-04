# ============================================================
# PyTorch-based NN Feature Selection with Cross-fitting
# (Multi-class classification, Cross-Entropy loss)
# - Fully-connected MLP / CNN1D / RNN(LSTM/GRU) / Transformer
# - Output logits of size K; train with cross-entropy
# - Constant LR, He/Xavier init, global standardization
# - xi from grad wrt inputs of a scalarized logit functional
# - Strict mirror-FDP threshold for feature selection
# ============================================================

import os, gc, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Array = np.ndarray

# ----------------------------
# Standardization
# ----------------------------
def standardize_global(X: Array) -> Tuple[Array, Array, Array]:
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    return Xn, mu.squeeze(0), sigma.squeeze(0)

@torch.no_grad()
def _toggle_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

# ----------------------------
# Logits scalarization (B,K) -> (B,)
# ----------------------------
def _scalarize_logits(logits: torch.Tensor, mode: str = "logsumexp") -> torch.Tensor:
    """
    Convert per-sample logits to a scalar score.
    mode: 'logsumexp' (default) | 'sum' | 'max' | 'l2'
    """
    if logits.dim() == 1:
        return logits
    if logits.size(-1) == 1:
        return logits.squeeze(-1)
    mode = mode.lower()
    if mode == "logsumexp":
        return torch.logsumexp(logits, dim=1)
    if mode == "sum":
        return logits.sum(dim=1)
    if mode == "max":
        return logits.max(dim=1).values
    if mode == "l2":
        return torch.linalg.vector_norm(logits, 2, dim=1)
    raise ValueError(f"unknown scalarization mode: {mode}")

# ----------------------------
# xi estimators (standardized coords)
# ----------------------------
def sum_input_gradients_std(
    model: nn.Module,
    Xn: torch.Tensor,
    batch_size: int = 2048,
    xi_scalar: str = "logsumexp",
) -> np.ndarray:
    """
    xi_std[j] = sum_i d s(x_i) / d x_{ij}, where s(x) is a scalarization of logits.
    """
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
        Xb = Xn[start:end].detach().clone().requires_grad_(True)  # (B,n)
        logits = model(Xb)                                        # (B,K)
        out = _scalarize_logits(logits, xi_scalar)                # (B,)
        s = out.sum()
        grads = torch.autograd.grad(s, Xb, retain_graph=False, create_graph=False)[0]  # (B,n)
        xi_std += grads.sum(dim=0)
        start = end

    _toggle_requires_grad(model, True)
    return xi_std.detach().cpu().numpy()

def steinized_input_direction(
    model: nn.Module,
    Xn: torch.Tensor,
    batch_size: int = 2048,
    xi_scalar: str = "logsumexp",
) -> np.ndarray:
    """
    "Steinized" variant: accumulate sum over batch of (Xb * grad s(x)) along features.
    """
    if any(isinstance(m, (nn.RNNBase, nn.LSTM, nn.GRU)) for m in model.modules()):
        model.train()
    else:
        model.eval()
    _toggle_requires_grad(model, False)

    d = Xn.shape[1]
    acc = torch.zeros(d, device=Xn.device)
    start = 0
    while start < Xn.shape[0]:
        end = min(start + batch_size, Xn.shape[0])
        Xb = Xn[start:end].detach().clone().requires_grad_(True)
        logits = model(Xb)
        out = _scalarize_logits(logits, xi_scalar)
        s = out.sum()
        grads = torch.autograd.grad(s, Xb, retain_graph=False, create_graph=False)[0]
        acc += (Xb * grads).sum(dim=0)
        start = end

    _toggle_requires_grad(model, True)
    return acc.detach().cpu().numpy()

# ----------------------------
# Aggregation psi and strict mirror-FDP threshold
# ----------------------------
def psi_aggregate(a: Array, b: Array, method: str = "mean", eps: float = 1e-12) -> Array:
    if method == "min":      return np.minimum(a, b)
    if method == "geomean":  return np.sqrt(np.maximum(a*b, 0.0) + eps)
    if method == "mean":     return 0.5*(a+b)
    if method == "harmmean": return (2.0*a*b)/(a+b+eps)
    raise ValueError("Unknown psi method")

def _strict_threshold_candidates(M: Array) -> Array:
    vals = M[np.isfinite(M)]
    abs_vals = np.unique(np.abs(vals)); abs_vals = abs_vals[abs_vals > 0.0]
    if abs_vals.size == 0: return np.array([], dtype=float)
    abs_vals.sort(); a = np.concatenate([[0.0], abs_vals])
    return 0.5*(a[:-1] + a[1:])

def strict_mirror_fdp_threshold(M: Array, alpha: float) -> Tuple[float, int, int, float]:
    """
    tau = min{ u>0 : FDPhat(u) <= alpha },  FDPhat(u) = #{M<-u} / (#{M>u} ∨ 1)
    """
    assert 0.0 < alpha < 1.0
    cand = _strict_threshold_candidates(M)
    if cand.size == 0: return np.inf, 0, 0, np.inf
    for u in cand:
        Rp = int((M > u).sum()); Rm = int((M < -u).sum())
        fdp = Rm / max(Rp, 1)
        if fdp <= alpha: return float(u), Rp, Rm, float(fdp)
    return np.inf, 0, 0, np.inf

# ----------------------------
# Metrics (optional true support)
# ----------------------------
def true_metrics(selected: Array, true_idx: Optional[Array], n: int) -> Tuple[Optional[float], Optional[float]]:
    if true_idx is None or len(true_idx) == 0: return None, None
    S = set(int(i) for i in selected.tolist())
    T = set(int(i) for i in np.array(true_idx).tolist())
    R = len(S); TP = len(S & T); FP = R - TP; FN = len(T - S)
    return float(FP / max(R,1)), float(FN / max(len(T),1))

# ----------------------------
# Models (output logits of size K)
# ----------------------------
class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int,
                 bias_init=0.0, init_mode="he", fixed_sigma=0.01, activation="relu"):
        super().__init__()
        dims = [input_dim] + hidden_dims + [num_classes]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1], bias=True) for i in range(len(dims)-1)])
        self.hidden_dims = hidden_dims
        self.bias_init = bias_init
        self.init_mode = init_mode
        self.fixed_sigma = fixed_sigma
        self.activation = activation.lower()
        self.reset_parameters()

    def _act(self, z):
        a = self.activation
        if a == "relu":         return F.relu(z)
        if a == "leaky_relu":   return F.leaky_relu(z, 0.01)
        if a == "elu":          return F.elu(z, 1.0)
        if a in ("silu","swish"): return F.silu(z)
        if a == "gelu":         return F.gelu(z)
        if a == "tanh":         return torch.tanh(z)
        if a == "softplus":     return F.softplus(z, beta=1.0)
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
        for i in range(len(self.hidden_dims)): h = self._act(self.layers[i](h))
        return self.layers[-1](h)  # logits (B,K)

def _init_lift_weight_gaussian_(lin: nn.Linear, fan_in: int, mode: str = "he", fixed_sigma: float = 0.01):
    with torch.no_grad():
        if mode == "he":      sigma = (2.0 / float(fan_in)) ** 0.5
        elif mode == "xavier":sigma = (1.0 / float(fan_in)) ** 0.5
        elif mode == "fixed": sigma = float(fixed_sigma)
        else: raise ValueError("lift_init_mode must be 'he'|'xavier'|'fixed'")
        lin.weight.normal_(0.0, sigma)
        if lin.bias is not None: lin.bias.zero_()

class TorchCNN1D_Lifted(nn.Module):
    """
    x:(B,n) -> z=(B,q) via Linear W1(bias=False) -> reshape (B,1,q) -> Conv1d stack -> GAP -> head -> logits (B,K)
    """
    def __init__(self, input_dim: int, lift_dim: int, num_classes: int,
                 conv_channels=[64,64], kernel_sizes=[11,7], strides=[1,1],
                 fc_dims=[128], activation="relu", init_mode="he", bias_init=0.0,
                 dropout=0.0, use_bn=False, lift_init_mode="he", lift_fixed_sigma=0.01):
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes) == len(strides)
        self.activation = activation.lower(); self.init_mode = init_mode; self.bias_init = bias_init
        self.use_bn = use_bn; self.input_dim = input_dim; self.lift_dim = lift_dim
        self.W1 = nn.Linear(input_dim, lift_dim, bias=False)
        layers=[]; in_ch=1
        for c,k,s in zip(conv_channels, kernel_sizes, strides):
            pad = k//2
            layers.append(nn.Conv1d(in_ch, c, kernel_size=k, stride=s, padding=pad, bias=True))
            if use_bn: layers.append(nn.BatchNorm1d(c))
            layers.append(self._act_layer())
            if dropout>0: layers.append(nn.Dropout(dropout))
            in_ch=c
        self.conv = nn.Sequential(*layers); self.gap = nn.AdaptiveAvgPool1d(1)
        dims = [in_ch]+list(fc_dims)+[num_classes]; fcs=[]
        for i in range(len(dims)-2):
            fcs.append(nn.Linear(dims[i], dims[i+1], bias=True)); fcs.append(self._act_layer())
            if dropout>0: fcs.append(nn.Dropout(dropout))
        fcs.append(nn.Linear(dims[-2], dims[-1], bias=True)); self.head = nn.Sequential(*fcs)
        _init_lift_weight_gaussian_(self.W1, fan_in=input_dim, mode=lift_init_mode, fixed_sigma=lift_fixed_sigma)
        self._init_rest()

    def _act_layer(self):
        a = self.activation
        if a == "relu": return nn.ReLU()
        if a == "leaky_relu": return nn.LeakyReLU(0.01)
        if a == "elu": return nn.ELU()
        if a in ("silu","swish"): return nn.SiLU()
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
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W1(x); y = z.unsqueeze(1); y = self.conv(y); y = self.gap(y).squeeze(-1)
        return self.head(y)  # logits (B,K)

class TorchRNN_Lifted(nn.Module):
    """
    x:(B,n) -> z=(B,q) -> view as (B,q,1) -> optional linear proj -> RNN -> head -> logits (B,K)
    """
    def __init__(self, input_dim: int, lift_dim: int, num_classes: int,
                 rnn_type="lstm", hidden_size=128, num_layers=1, bidirectional=True,
                 input_proj_dim=4, fc_dims=[128], activation="relu", init_mode="xavier",
                 bias_init=0.0, dropout=0.0, lift_init_mode="xavier", lift_fixed_sigma=0.01):
        super().__init__()
        self.activation = activation.lower(); self.init_mode = init_mode; self.bias_init = bias_init
        self.rnn_type = rnn_type.lower(); self.input_dim = input_dim; self.lift_dim = lift_dim
        self.W1 = nn.Linear(input_dim, lift_dim, bias=False)
        self.input_proj = nn.Identity(); rnn_input_size=1
        if input_proj_dim>1: self.input_proj = nn.Linear(1, input_proj_dim, bias=True); rnn_input_size = input_proj_dim
        common = dict(input_size=rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                      batch_first=True, dropout=(dropout if num_layers>1 else 0.0), bidirectional=bidirectional)
        if self.rnn_type == "lstm": self.rnn = nn.LSTM(**common)
        elif self.rnn_type == "gru": self.rnn = nn.GRU(**common)
        else: raise ValueError("rnn_type must be 'lstm' or 'gru'")
        out_dim = hidden_size * (2 if bidirectional else 1)
        dims = [out_dim] + list(fc_dims) + [num_classes]; fcs=[]
        for i in range(len(dims)-2):
            fcs.append(nn.Linear(dims[i], dims[i+1], bias=True)); fcs.append(self._act_layer())
            if dropout>0: fcs.append(nn.Dropout(dropout))
        fcs.append(nn.Linear(dims[-2], dims[-1], bias=True)); self.head = nn.Sequential(*fcs)
        _init_lift_weight_gaussian_(self.W1, fan_in=input_dim, mode=lift_init_mode, fixed_sigma=lift_fixed_sigma)
        self._init_rest()

    def _act_layer(self):
        a = self.activation
        if a == "relu": return nn.ReLU()
        if a == "leaky_relu": return nn.LeakyReLU(0.01)
        if a == "elu": return nn.ELU()
        if a in ("silu","swish"): return nn.SiLU()
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
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.rnn.bidirectional else h_n[-1]
        else:
            out, h_n = self.rnn(s)
            last = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.rnn.bidirectional else h_n[-1]
        return self.head(last)  # logits (B,K)

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

class TorchTransformer1D_Lifted(nn.Module):
    """
    x:(B,n) -> z=(B,q) -> in_proj to d_model -> pos enc -> TransformerEncoder -> pool -> head -> logits (B,K)
    """
    def __init__(self, input_dim: int, lift_dim: int, num_classes: int,
                 d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1,
                 activation="relu", init_mode="he", bias_init=0.0, pool="mean",
                 use_sinusoidal_pos=True, fc_dims=[128], lift_init_mode="he", lift_fixed_sigma=0.01):
        super().__init__()
        assert d_model % nhead == 0
        self.init_mode = init_mode; self.bias_init = bias_init
        self.pool = pool.lower(); self.head_activation = activation.lower()
        self.input_dim = input_dim; self.lift_dim = lift_dim
        self.W1 = nn.Linear(input_dim, lift_dim, bias=False)
        self.in_proj = nn.Linear(1, d_model, bias=True)
        self.pos = SinusoidalPositionalEncoding(d_model) if use_sinusoidal_pos else nn.Embedding(10000, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        if self.pool == "cls": self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        dims = [d_model] + list(fc_dims) + [num_classes]; fcs=[]
        for i in range(len(dims)-2):
            fcs.append(nn.Linear(dims[i], dims[i+1], bias=True)); fcs.append(self._act_layer())
            if dropout>0: fcs.append(nn.Dropout(dropout))
        fcs.append(nn.Linear(dims[-2], dims[-1], bias=True)); self.head = nn.Sequential(*fcs)
        _init_lift_weight_gaussian_(self.W1, fan_in=input_dim, mode=lift_init_mode, fixed_sigma=lift_fixed_sigma)
        self._init_rest()

    def _act_layer(self):
        a = self.head_activation
        if a == "relu": return nn.ReLU()
        if a == "leaky_relu": return nn.LeakyReLU(0.01)
        if a == "elu": return nn.ELU()
        if a in ("silu","swish"): return nn.SiLU()
        if a == "gelu": return nn.GELU()
        if a == "tanh": return nn.Tanh()
        if a == "softplus": return nn.Softplus(beta=1.0)
        raise ValueError(f"unknown activation: {a}")

    def _init_rest(self):
        with torch.no_grad():
            if self.init_mode == "he": nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity="relu")
            elif self.init_mode == "xavier": nn.init.xavier_normal_(self.in_proj.weight)
            else: nn.init.normal_(self.in_proj.weight, 0.0, 0.01)
            if self.in_proj.bias is not None: self.in_proj.bias.fill_(float(self.bias_init))
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    if self.init_mode == "he": nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    elif self.init_mode == "xavier": nn.init.xavier_normal_(m.weight)
                    else: nn.init.normal_(m.weight, 0.0, 0.01)
                    if m.bias is not None: m.bias.fill_(float(self.bias_init))
            for name, p in self.encoder.named_parameters():
                if "weight" in name and p.dim() >= 2:
                    if self.init_mode == "he": nn.init.kaiming_normal_(p, nonlinearity="relu")
                    elif self.init_mode == "xavier": nn.init.xavier_normal_(p)
                    else: nn.init.normal_(p, 0.0, 0.01)
                elif "bias" in name: p.fill_(float(self.bias_init))
            if hasattr(self, "cls"): nn.init.normal_(self.cls, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W1(x); s = z.unsqueeze(-1); s = self.in_proj(s)
        if isinstance(self.pos, SinusoidalPositionalEncoding): s = self.pos(s)
        else:
            B,L,_ = s.shape; pos_ids = torch.arange(L, device=s.device).unsqueeze(0).expand(B,L); s = s + self.pos(pos_ids)
        if self.pool == "cls":
            B = s.shape[0]; cls = self.cls.expand(B, -1, -1); s = torch.cat([cls, s], dim=1)
        h = self.encoder(s)
        pooled = h[:,0,:] if self.pool=="cls" else h.mean(dim=1)
        return self.head(pooled)  # logits (B,K)

# ----------------------------
# Training step (CE)
# ----------------------------
def one_sgd_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_type: str,
    batch_size: int,
    g: torch.Generator
) -> float:
    m = X.shape[0]
    idx = torch.randint(low=0, high=m, size=(min(batch_size, m),), generator=g, device=X.device)
    Xb = X.index_select(0, idx)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(Xb)  # (B,K) or (B,1)

    loss_low = loss_type.lower()
    if loss_low in ("ce", "crossentropy", "cross-entropy"):
        yb = y.index_select(0, idx).long()  # integer labels 0..K-1
        loss = F.cross_entropy(logits, yb)
    elif loss_low == "logistic":  # binary compatibility
        yb = y.index_select(0, idx).view(-1,1).float()
        loss = F.binary_cross_entropy_with_logits(logits, yb)
    elif loss_low == "mse":       # regression compatibility
        yb = y.index_select(0, idx).view(-1,1).float()
        loss = 0.5 * torch.mean((logits - yb)**2)
    else:
        raise ValueError("loss_type must be 'ce'|'mse'|'logistic'")

    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())

# ----------------------------
# Path tracking with cross-fitting
# ----------------------------
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

def torch_nn_feature_selection_path_generic(
    X: Array, y: Array, alpha: float, T: int,
    build_model: Callable[[int], nn.Module],
    true_idx: Optional[Array] = None,
    batch_size: int = 128,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    loss: str = "ce",                   # "ce" | "mse" | "logistic"
    psi_method: str = "min",
    seed_split: int = 2025,
    seed_model1: int = 11,
    seed_model2: int = 22,
    compute_every: int = 10,
    xi_batch: int = 2048,
    xi_mode: str = "sumgrad",           # "sumgrad" | "steinized"
    xi_scalar: str = "logsumexp",       # logits -> scalar
) -> PathResult:
    assert 0.0 < alpha < 1.0 and T >= 1
    m, n = X.shape

    Xn, mu, sigma = standardize_global(X)
    rng_np = np.random.default_rng(seed_split)
    perm = rng_np.permutation(m); m1 = m // 2
    idx1 = perm[:m1]; idx2 = perm[m1:]

    X1n = torch.tensor(Xn[idx1], dtype=torch.float32, device=device)
    X2n = torch.tensor(Xn[idx2], dtype=torch.float32, device=device)

    loss_low = loss.lower()
    if loss_low in ("ce","crossentropy","cross-entropy"):
        y1 = torch.tensor(y[idx1], dtype=torch.long, device=device)
        y2 = torch.tensor(y[idx2], dtype=torch.long, device=device)
    elif loss_low == "logistic":
        y1 = torch.tensor(y[idx1], dtype=torch.float32, device=device)
        y2 = torch.tensor(y[idx2], dtype=torch.float32, device=device)
    elif loss_low == "mse":
        y1 = torch.tensor(y[idx1], dtype=torch.float32, device=device)
        y2 = torch.tensor(y[idx2], dtype=torch.float32, device=device)
    else:
        raise ValueError("unknown loss")

    torch.manual_seed(seed_model1); model1 = build_model(n).to(device)
    torch.manual_seed(seed_model2); model2 = build_model(n).to(device)

    opt1 = torch.optim.SGD(model1.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.0, nesterov=False)
    opt2 = torch.optim.SGD(model2.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.0, nesterov=False)

    g1 = torch.Generator(device=device).manual_seed(seed_model1 + 1)
    g2 = torch.Generator(device=device).manual_seed(seed_model2 + 1)

    steps, taus, Rps, Rms, FDPs = [], [], [], [], []
    FDRs, T2s = [], []
    loss_steps, loss_values = [], []; loss_accum, loss_count = 0.0, 0
    xi1_list, xi2_list, M_list = [], [], []
    last_xi1 = last_xi2 = last_M = None; last_selected = np.array([], dtype=int); last_tau = np.inf

    for t in range(1, T + 1):
        l1 = one_sgd_step(model1, X1n, y1, opt1, loss, batch_size, g1)
        l2 = one_sgd_step(model2, X2n, y2, opt2, loss, batch_size, g2)
        loss_accum += 0.5*(l1 + l2); loss_count += 1

        if (t % compute_every) == 0 or (t == T):
            loss_steps.append(t); loss_values.append(loss_accum / max(loss_count, 1))
            loss_accum, loss_count = 0.0, 0

            if xi_mode == "sumgrad":
                xi1_std = sum_input_gradients_std(model2, X1n, batch_size=xi_batch, xi_scalar=xi_scalar)
                xi2_std = sum_input_gradients_std(model1, X2n, batch_size=xi_batch, xi_scalar=xi_scalar)
            elif xi_mode == "steinized":
                xi1_std = steinized_input_direction(model2, X1n, batch_size=xi_batch, xi_scalar=xi_scalar)
                xi2_std = steinized_input_direction(model1, X2n, batch_size=xi_batch, xi_scalar=xi_scalar)
            else:
                raise ValueError("xi_mode must be 'sumgrad' or 'steinized'")

            xi1 = xi1_std / sigma; xi2 = xi2_std / sigma
            sgn = np.sign(xi1 * xi2); mag = psi_aggregate(np.abs(xi1), np.abs(xi2), method=psi_method)
            M = sgn * mag

            tau, Rp, Rm, FDPhat = strict_mirror_fdp_threshold(M, alpha=alpha)
            selected = np.where(M > tau)[0] if np.isfinite(tau) else np.array([], dtype=int)
            FDR_true, TypeII = true_metrics(selected, true_idx, n)

            steps.append(t); taus.append(tau); Rps.append(Rp); Rms.append(Rm); FDPs.append(FDPhat)
            FDRs.append(FDR_true if FDR_true is not None else np.nan); T2s.append(TypeII if TypeII is not None else np.nan)
            xi1_list.append(xi1.copy()); xi2_list.append(xi2.copy()); M_list.append(M.copy())
            last_xi1, last_xi2, last_M = xi1, xi2, M; last_selected = selected; last_tau = tau

    xi1_hist = np.stack(xi1_list, axis=0) if xi1_list else np.empty((0,0))
    xi2_hist = np.stack(xi2_list, axis=0) if xi2_list else np.empty((0,0))
    M_hist   = np.stack(M_list,   axis=0) if M_list   else np.empty((0,0))

    return PathResult(
        steps=np.array(steps,int), tau=np.array(taus,float), R_plus=np.array(Rps,int),
        R_minus=np.array(Rms,int), FDPhat=np.array(FDPs,float),
        FDR_true=np.array(FDRs,float) if (true_idx is not None and len(true_idx)>0) else None,
        TypeII=np.array(T2s,float) if (true_idx is not None and len(true_idx)>0) else None,
        last_xi1=last_xi1, last_xi2=last_xi2, last_M=last_M, last_selected=last_selected, last_tau=float(last_tau),
        loss_steps=np.array(loss_steps,int), loss_values=np.array(loss_values,float),
        xi1_hist=xi1_hist, xi2_hist=xi2_hist, M_hist=M_hist
    )

# ----------------------------
# Simple wrappers (MLP/CNN/RNN/Transformer)
# ----------------------------
def torch_nn_feature_selection_path(
    X: Array, y: Array, alpha: float, T: int,
    true_idx: Optional[Array] = None,
    hidden_dims: List[int] = [64, 64],
    num_classes: int = 3,
    batch_size: int = 128, lr: float = 5e-3, weight_decay: float = 0.0,
    loss: str = "ce", psi_method: str = "min",
    seed_split: int = 2025, seed_model1: int = 11, seed_model2: int = 22,
    compute_every: int = 10, xi_batch: int = 2048, xi_mode: str = "sumgrad",
    activation: str = "relu", init_mode: str = "he", xi_scalar: str = "logsumexp",
) -> PathResult:
    n = X.shape[1]
    def _build(n_in: int) -> nn.Module:
        return TorchMLP(n_in, hidden_dims, num_classes, bias_init=0.0, init_mode=init_mode, activation=activation)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar
    )

def torch_cnn1d_feature_selection_path(
    X, y, alpha, T, true_idx=None,
    num_classes: int = 3, lift_dim: int | None = None, lift_init_mode: str = "he", lift_fixed_sigma: float = 0.01,
    conv_channels=[64,64], kernel_sizes=[11,7], strides=[1,1], fc_dims=[128],
    activation="relu", init_mode="he", bias_init=0.0, dropout=0.0, use_bn=False,
    batch_size=128, lr=5e-3, weight_decay=0.0, loss="ce", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=2048, xi_mode="sumgrad",
    xi_scalar: str = "logsumexp",
):
    n = X.shape[1]; q = n if lift_dim is None else int(lift_dim)
    def _build(n_in: int) -> nn.Module:
        return TorchCNN1D_Lifted(n_in, q, num_classes,
                                 conv_channels, kernel_sizes, strides, fc_dims,
                                 activation, init_mode, bias_init, dropout, use_bn,
                                 lift_init_mode, lift_fixed_sigma)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split, seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar
    )

def torch_rnn_feature_selection_path(
    X, y, alpha, T, true_idx=None,
    num_classes: int = 3, lift_dim: int | None = None, lift_init_mode: str = "xavier", lift_fixed_sigma: float = 0.01,
    rnn_type="lstm", hidden_size=128, num_layers=1, bidirectional=True,
    input_proj_dim=4, fc_dims=[128], activation="relu", init_mode="xavier", bias_init=0.0, dropout=0.0,
    batch_size=128, lr=5e-3, weight_decay=0.0, loss="ce", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=2048, xi_mode="sumgrad",
    xi_scalar: str = "logsumexp",
):
    n = X.shape[1]; q = n if lift_dim is None else int(lift_dim)
    def _build(n_in: int) -> nn.Module:
        return TorchRNN_Lifted(n_in, q, num_classes,
                               rnn_type, hidden_size, num_layers, bidirectional,
                               input_proj_dim, fc_dims, activation, init_mode, bias_init, dropout,
                               lift_init_mode, lift_fixed_sigma)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar
    )

def torch_transformer_feature_selection_path(
    X, y, alpha, T, true_idx=None,
    num_classes: int = 3, lift_dim: int | None = None, lift_init_mode: str = "he", lift_fixed_sigma: float = 0.01,
    d_model=128, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1,
    activation="relu", init_mode="he", bias_init=0.0,
    pool="mean", use_sinusoidal_pos=True, fc_dims=[128],
    batch_size=128, lr=5e-3, weight_decay=0.0, loss="ce", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=2048, xi_mode="sumgrad",
    xi_scalar: str = "logsumexp",
):
    n = X.shape[1]; q = n if lift_dim is None else int(lift_dim)
    def _build(n_in: int) -> nn.Module:
        return TorchTransformer1D_Lifted(n_in, q, num_classes,
                                         d_model, nhead, num_layers, dim_feedforward, dropout,
                                         activation, init_mode, bias_init, pool, use_sinusoidal_pos, fc_dims,
                                         lift_init_mode, lift_fixed_sigma)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split, seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar
    )

# ============================================================
# Multi-class synthetic data (trigonometric softmax)
# ============================================================
def generate_trig_softmax_data(
    m: int, n: int, K: int = 3, seed: int = 0, tau: float = 1.0,
    alpha=1.0, beta=0.8, gamma=0.6, omega=1.0, nu=1.3, bias=0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-class softmax (trigonometric multi-index). Half of features are truly related.
    Returns: X (m,n), y (m,), true_idx (n/2,), B (n,3)
    """
    assert n % 2 == 0, "n should be even."
    rng = np.random.default_rng(seed)
    half = n // 2
    true_idx = np.arange(half, dtype=int)
    X = rng.standard_normal((m, n))
    G = rng.standard_normal((half, 3)); Q, _ = np.linalg.qr(G)  # (half,3)
    B = np.zeros((n, 3)); B[:half, :] = Q
    B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-12

    def to_vec(x):
        x = np.asarray(x)
        if x.ndim == 0: return np.full(K, float(x))
        if x.shape == (K,): return x.astype(float)
        raise ValueError("alpha/beta/gamma/omega/nu/bias must be scalar or shape=(K,)")

    alpha = to_vec(alpha); beta = to_vec(beta); gamma = to_vec(gamma)
    omega = to_vec(omega); nu   = to_vec(nu);   bias  = to_vec(bias)

    t1 = X @ B[:,0]; t2 = X @ B[:,1]; t3 = X @ B[:,2]  # (m,)
    H = np.empty((m, K), dtype=float)
    for k in range(K):
        H[:,k] = alpha[k]*np.sin(omega[k]*t1) + beta[k]*np.cos(nu[k]*t2) + gamma[k]*t3 + bias[k]
    Z = H / max(tau, 1e-12); Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=P[i]) for i in range(m)], dtype=int)
    return X, y, true_idx, B
