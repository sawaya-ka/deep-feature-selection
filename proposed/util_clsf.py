# ============================================================
# PyTorch-based NN Feature Selection with Cross-fitting
# (Multi-class classification, Cross-Entropy loss)
# - Fully-connected MLP / CNN1D / RNN(LSTM/GRU) / Transformer
# - Output logits of size K; train with cross-entropy
# - Constant LR, He/Xavier init, original input coordinates (no standardization)
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
from contextlib import nullcontext

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
    mode: 'logsumexp' (default) | 'sum' | 'max' | 'l2' | 'contrast'
    'contrast' is the fixed last-minus-first logit difference and is invariant
    to adding the same input-dependent function to all class logits.
    """
    mode = mode.lower()
    if mode not in {"logsumexp", "sum", "max", "l2", "contrast"}:
        raise ValueError(f"unknown scalarization mode: {mode}")
    if mode == "contrast":
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("contrast requires logits of shape (batch, K), K >= 2.")
        return logits[:, -1] - logits[:, 0]
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
# xi estimators (original training coordinates)
# ----------------------------
def sum_input_gradients(model, X, batch_size=2048, scalarize=None):
    """Compute sum_i grad_x f_model(X_i) and return a float64 NumPy vector.

    Use the same block that trained model. X must use exactly the coordinates
    used for training. This function does not center, standardize, average,
    or divide by a simulation-specific projection radius.

    By default, model output must have shape (batch,) or (batch, 1).
    For classification, explicitly supply a fixed, label-free scalarization,
    for example scalarize=lambda logits: logits.logsumexp(dim=-1).
    scalarize must return one scalar per input row.

    The predictor must act separately on input rows in evaluation mode.
    Evaluation mode makes the differentiated predictor deterministic. All
    modules' original training flags are restored, including mixed modes.
    CUDA recurrent modules use the non-cuDNN path because cuDNN evaluation
    forward passes do not support this input-gradient backward operation.
    Parameter requires_grad flags and accumulated parameter gradients are
    not changed. PyTorch is imported only when this function is called.
    """
    import torch

    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, bool):
        raise ValueError("batch_size must be a positive integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Input gradients cannot be evaluated inside inference_mode.")

    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    device = reference.device if reference is not None else torch.device("cpu")
    dtype = reference.dtype if reference is not None and reference.is_floating_point() else torch.float32
    X_tensor = torch.as_tensor(X, dtype=dtype, device=device).detach()
    if X_tensor.ndim != 2 or X_tensor.shape[0] == 0 or X_tensor.shape[1] == 0:
        raise ValueError("X must be a nonempty matrix of shape (samples, features).")
    if not bool(torch.isfinite(X_tensor).all()):
        raise ValueError("X must contain only finite values in the model dtype.")

    modules = tuple(model.modules())
    original_modes = tuple(module.training for module in modules)
    uses_cuda_rnn = device.type == "cuda" and any(
        isinstance(module, torch.nn.RNNBase) for module in modules
    )
    backend_context = torch.backends.cudnn.flags(enabled=False) if uses_cuda_rnn else nullcontext()
    total = torch.zeros(X_tensor.shape[1], dtype=torch.float64, device=device)
    try:
        model.eval()
        with backend_context, torch.enable_grad():
            for start in range(0, X_tensor.shape[0], int(batch_size)):
                batch = X_tensor[start:start + int(batch_size)].detach().clone()
                batch.requires_grad_(True)
                output = model(batch)
                values = scalarize(output) if scalarize is not None else output
                if not isinstance(values, torch.Tensor):
                    raise TypeError("The scalarized model output must be a Tensor.")
                if values.ndim == 2 and values.shape[1] == 1:
                    values = values[:, 0]
                if values.ndim != 1 or values.shape[0] != batch.shape[0]:
                    raise ValueError("The scalarized output must contain one value per input row.")
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("The scalarized model output is nonfinite.")
                if values.requires_grad:
                    gradient = torch.autograd.grad(
                        values.sum(), batch,
                        create_graph=False, retain_graph=False, allow_unused=True,
                    )[0]
                    if gradient is not None:
                        if not bool(torch.isfinite(gradient).all()):
                            raise FloatingPointError("An input gradient is nonfinite.")
                        total += gradient.detach().to(torch.float64).sum(dim=0)
    finally:
        for module, training in zip(modules, original_modes):
            module.training = training

    result = total.cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("The accumulated sensitivity is nonfinite.")
    return result


def steinized_input_direction(
    model: nn.Module,
    Xn: torch.Tensor,
    batch_size: int = 2048,
    xi_scalar: str = "logsumexp",
) -> np.ndarray:
    """Reserved legacy API. This statistic is unsupported by the FDR path."""
    raise NotImplementedError(
        "The coordinatewise X * grad statistic is not part of the signed-gradient "
        "feature-selection path: it is not rotation equivariant and it does not "
        "have the required coordinate sign-flip behavior. Use xi_mode='sumgrad'."
    )

# ----------------------------
# Aggregation psi and strict mirror-FDP threshold
# ----------------------------
def psi_aggregate(a: Array, b: Array, method: str = "mean", eps: float = 1e-12) -> Array:
    if method == "min":
        return np.minimum(a, b)
    if method == "mean":
        return 0.5 * a + 0.5 * b
    if method == "geomean":
        return np.sqrt(a) * np.sqrt(b)
    if method == "harmmean":
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        ratio = np.divide(lo, hi, out=np.zeros_like(lo), where=hi > 0.0)
        return lo / (0.5 + 0.5 * ratio)
    raise ValueError("Unknown psi method")

def _strict_threshold_candidates(M):
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 1 or not np.isfinite(M).all():
        raise ValueError("M must be a finite one-dimensional array.")
    return np.unique(np.r_[0.0, np.abs(M)])


def strict_mirror_fdp_threshold(M, alpha):
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    M = np.asarray(M, dtype=np.float64)
    for u in _strict_threshold_candidates(M):
        Rp = int(np.count_nonzero(M > u))
        Rm = int(np.count_nonzero(M < -u))
        fdp = Rm / max(Rp, 1)
        if fdp <= alpha:
            return float(u), Rp, Rm, float(fdp)
    raise RuntimeError("No threshold found for finite scores.")

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
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
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
                 use_sinusoidal_pos=True, fc_dims=[128], lift_init_mode="he", lift_fixed_sigma=0.01,
                 norm_first: bool = False):
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
        # Toggle after initialization to keep the seeded weights identical.
        # No final encoder LayerNorm is added.
        self.norm_first = bool(norm_first)
        if self.norm_first:
            for layer in self.encoder.layers:
                layer.norm_first = True
            self.encoder.use_nested_tensor = False

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
def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _validate_loss_output(logits, y, loss_type):
    """Validate output/target shapes before any broadcasting or CE casting."""
    if not isinstance(logits, torch.Tensor):
        raise TypeError("The model output must be a Tensor.")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite output during training.")
    if y.ndim != 1 or logits.ndim < 1 or logits.shape[0] != y.shape[0]:
        raise ValueError("Targets must be one-dimensional and match the output batch size.")
    if loss_type in ("ce", "crossentropy", "cross-entropy"):
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("Cross-entropy requires logits of shape (batch, K), K >= 2.")
        if y.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("Cross-entropy targets must have an integer dtype.")
        if bool(((y < 0) | (y >= logits.shape[1])).any()):
            raise ValueError("Cross-entropy labels must lie in [0, K-1].")
    elif loss_type in ("mse", "logistic"):
        if not (logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1)):
            raise ValueError(f"{loss_type} requires scalar output per row, not K-class logits.")
        if not bool(torch.isfinite(y).all()):
            raise ValueError("Targets must be finite.")
        if loss_type == "logistic" and bool(((y < 0) | (y > 1)).any()):
            raise ValueError("Binary logistic targets must lie in [0, 1].")
    else:
        raise ValueError("loss_type must be 'ce'|'mse'|'logistic'")


def _check_model_finite(model, gradients=False):
    tensors = [p.grad for p in model.parameters() if p.grad is not None] if gradients else list(model.parameters())
    if tensors and not bool(torch.stack([torch.isfinite(p).all() for p in tensors]).all()):
        kind = "gradient" if gradients else "parameter after optimizer update"
        raise FloatingPointError(f"Nonfinite {kind} during training.")


def _numerical_context(model, X, optimizer, iteration, split):
    maximum = 0.0
    nonfinite = []
    for name, param in model.named_parameters():
        values = param.detach()
        finite = torch.isfinite(values)
        if not bool(finite.all()):
            nonfinite.append(name)
        if bool(finite.any()):
            maximum = max(maximum, float(values[finite].abs().max().item()))
    rates = [float(group["lr"]) for group in optimizer.param_groups]
    return (f"{type(model).__name__}, iteration={iteration}, split={split}, "
            f"lr={rates}, max_abs_input={float(X.abs().max().item()):.3e}, "
            f"max_abs_finite_parameter={maximum:.3e}, nonfinite_parameters={nonfinite}")


def one_sgd_step(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_type: str,
    batch_size: int,
    g: torch.Generator
) -> float:
    """One minibatch optimizer step (legacy name, also supports Adam)."""
    batch_size = _positive_int("batch_size", batch_size)
    if X.ndim != 2 or X.shape[0] < 1 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("X and one-dimensional y must contain the same nonzero number of rows.")
    m = X.shape[0]
    idx = torch.randint(low=0, high=m, size=(min(batch_size, m),), generator=g, device=X.device)
    Xb = X.index_select(0, idx)
    yb = y.index_select(0, idx)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(Xb)
    loss_low = loss_type.lower()
    _validate_loss_output(logits, yb, loss_low)
    if loss_low in ("ce", "crossentropy", "cross-entropy"):
        loss = F.cross_entropy(logits, yb.long())
    else:
        scores = logits.reshape(-1)
        targets = yb.to(dtype=scores.dtype)
        if loss_low == "logistic":
            loss = F.binary_cross_entropy_with_logits(scores, targets)
        else:
            loss = 0.5 * torch.mean((scores - targets)**2)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite loss during training.")
    loss.backward()
    _check_model_finite(model, gradients=True)
    optimizer.step()
    _check_model_finite(model)
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
    xi_mode: str = "sumgrad",           # only "sumgrad" is supported by this path
    xi_scalar: str = "logsumexp",       # fixed, label-free logits -> scalar
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
) -> PathResult:
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    T = _positive_int("T", T)
    batch_size = _positive_int("batch_size", batch_size)
    compute_every = _positive_int("compute_every", compute_every)
    xi_batch = _positive_int("xi_batch", xi_batch)
    if xi_mode != "sumgrad":
        raise ValueError("This path requires xi_mode='sumgrad'.")
    if xi_scalar not in {"logsumexp", "sum", "max", "l2", "contrast"}:
        raise ValueError(f"unknown scalarization mode: {xi_scalar}")
    if psi_method not in {"min", "mean", "geomean", "harmmean"}:
        raise ValueError("Unknown psi method")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay nonnegative, both finite.")
    Xn = np.asarray(X, dtype=np.float32)
    if Xn.ndim != 2 or not np.isfinite(Xn).all():
        raise ValueError("X must be a finite matrix.")
    m, n = Xn.shape
    if m < 2 or m % 2 != 0 or n < 1:
        raise ValueError("Use an even sample size m >= 2 and n >= 1.")
    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != m:
        raise ValueError("y must have shape (m,) and match the number of rows in X.")
    loss_low = loss.lower()
    if loss_low in ("ce", "crossentropy", "cross-entropy"):
        if not np.issubdtype(y.dtype, np.integer) or np.issubdtype(y.dtype, np.bool_):
            raise ValueError("Cross-entropy targets must have an integer dtype.")
        if np.any(y < 0) or np.any(y > np.iinfo(np.int64).max):
            raise ValueError("Cross-entropy labels must be nonnegative int64 values.")
    elif loss_low in ("mse", "logistic"):
        y = np.asarray(y, dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("Targets must be finite in float32.")
        if loss_low == "logistic" and np.any((y < 0) | (y > 1)):
            raise ValueError("Binary logistic targets must lie in [0, 1].")
    else:
        raise ValueError("loss must be 'ce'|'mse'|'logistic'.")
    if true_idx is not None:
        true_idx = np.asarray(true_idx)
        if true_idx.ndim == 1 and true_idx.size == 0:
            true_idx = true_idx.astype(np.int64)
        if (true_idx.ndim != 1 or not np.issubdtype(true_idx.dtype, np.integer)
                or np.any((true_idx < 0) | (true_idx >= n))
                or np.unique(true_idx).size != true_idx.size):
            raise ValueError("true_idx must contain distinct integer feature indices in [0, n-1].")
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

    # Validate all labels and the model/loss interface before starting updates.
    # Evaluation mode avoids batch-statistic updates and active dropout.
    for model, Xpart, ypart in ((model1, X1n, y1), (model2, X2n, y2)):
        modes = [(module, module.training) for module in model.modules()]
        try:
            model.eval()
            with torch.no_grad():
                output = model(Xpart[:1])
            _validate_loss_output(output, ypart[:1], loss_low)
            if loss_low in ("ce", "crossentropy", "cross-entropy") and bool((ypart >= output.shape[1]).any()):
                raise ValueError("Cross-entropy labels must lie in [0, K-1].")
            if xi_scalar == "contrast" and (output.ndim != 2 or output.shape[1] < 2):
                raise ValueError("contrast requires logits of shape (batch, K), K >= 2.")
            _check_model_finite(model)
        finally:
            for module, training in modes:
                module.training = training

    if optimizer_factory is None:
        opt1 = torch.optim.SGD(model1.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.0, nesterov=False)
        opt2 = torch.optim.SGD(model2.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.0, nesterov=False)
    else:
        opt1 = optimizer_factory(model1)
        opt2 = optimizer_factory(model2)
    scalarize = lambda logits: _scalarize_logits(logits, xi_scalar)

    g1 = torch.Generator(device=device).manual_seed(seed_model1 + 1)
    g2 = torch.Generator(device=device).manual_seed(seed_model2 + 1)

    steps, taus, Rps, Rms, FDPs = [], [], [], [], []
    FDRs, T2s = [], []
    loss_steps, loss_values = [], []; loss_accum, loss_count = 0.0, 0
    xi1_list, xi2_list, M_list = [], [], []
    last_xi1 = last_xi2 = last_M = None; last_selected = np.array([], dtype=int); last_tau = np.inf

    for t in range(1, T + 1):
        losses = []
        for split, model, Xpart, ypart, opt, gen in (
            (1, model1, X1n, y1, opt1, g1), (2, model2, X2n, y2, opt2, g2)
        ):
            try:
                losses.append(one_sgd_step(model, Xpart, ypart, opt, loss, batch_size, gen))
            except FloatingPointError as exc:
                raise FloatingPointError(f"{_numerical_context(model, Xpart, opt, t, split)}: {exc}") from exc
        l1, l2 = losses
        if t == 1 or t % compute_every == 0 or t == T:
            print(f"[{type(model1).__name__}] iteration={t}, loss1={l1:.6e}, loss2={l2:.6e}", flush=True)
        loss_accum += 0.5*(l1 + l2); loss_count += 1

        if (t % compute_every) == 0 or (t == T):
            loss_steps.append(t); loss_values.append(loss_accum / max(loss_count, 1))
            loss_accum, loss_count = 0.0, 0

            sensitivities = []
            for split, model, Xpart, opt in (
                (1, model1, X1n, opt1), (2, model2, X2n, opt2)
            ):
                try:
                    sensitivities.append(sum_input_gradients(
                        model, Xpart, batch_size=xi_batch, scalarize=scalarize))
                except FloatingPointError as exc:
                    raise FloatingPointError(
                        f"{_numerical_context(model, Xpart, opt, t, split)}, sensitivity evaluation: {exc}"
                    ) from exc
            xi1, xi2 = sensitivities

            sgn = np.sign(xi1) * np.sign(xi2); mag = psi_aggregate(np.abs(xi1), np.abs(xi2), method=psi_method)
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
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
) -> PathResult:
    def _build(n_in: int) -> nn.Module:
        return TorchMLP(n_in, hidden_dims, num_classes, bias_init=0.0, init_mode=init_mode, activation=activation)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar,
        optimizer_factory=optimizer_factory
    )

def torch_cnn1d_feature_selection_path(
    X, y, alpha, T, true_idx=None,
    num_classes: int = 3, lift_dim: int | None = None, lift_init_mode: str = "he", lift_fixed_sigma: float = 0.01,
    conv_channels=[64,64], kernel_sizes=[11,7], strides=[1,1], fc_dims=[128],
    activation="relu", init_mode="he", bias_init=0.0, dropout=0.0, use_bn=False,
    batch_size=128, lr=5e-3, weight_decay=0.0, loss="ce", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=2048, xi_mode="sumgrad",
    xi_scalar: str = "logsumexp",
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
):
    def _build(n_in: int) -> nn.Module:
        q = n_in if lift_dim is None else _positive_int("lift_dim", lift_dim)
        return TorchCNN1D_Lifted(n_in, q, num_classes,
                                 conv_channels, kernel_sizes, strides, fc_dims,
                                 activation, init_mode, bias_init, dropout, use_bn,
                                 lift_init_mode, lift_fixed_sigma)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split, seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar,
        optimizer_factory=optimizer_factory
    )

def torch_rnn_feature_selection_path(
    X, y, alpha, T, true_idx=None,
    num_classes: int = 3, lift_dim: int | None = None, lift_init_mode: str = "xavier", lift_fixed_sigma: float = 0.01,
    rnn_type="lstm", hidden_size=128, num_layers=1, bidirectional=True,
    input_proj_dim=4, fc_dims=[128], activation="relu", init_mode="xavier", bias_init=0.0, dropout=0.0,
    batch_size=128, lr=5e-3, weight_decay=0.0, loss="ce", psi_method="min",
    seed_split=2025, seed_model1=11, seed_model2=22, compute_every=10, xi_batch=2048, xi_mode="sumgrad",
    xi_scalar: str = "logsumexp",
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
):
    def _build(n_in: int) -> nn.Module:
        q = n_in if lift_dim is None else _positive_int("lift_dim", lift_dim)
        return TorchRNN_Lifted(n_in, q, num_classes,
                               rnn_type, hidden_size, num_layers, bidirectional,
                               input_proj_dim, fc_dims, activation, init_mode, bias_init, dropout,
                               lift_init_mode, lift_fixed_sigma)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split,
        seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar,
        optimizer_factory=optimizer_factory
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
    norm_first: bool = False,
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
):
    def _build(n_in: int) -> nn.Module:
        q = n_in if lift_dim is None else _positive_int("lift_dim", lift_dim)
        return TorchTransformer1D_Lifted(n_in, q, num_classes,
                                         d_model, nhead, num_layers, dim_feedforward, dropout,
                                         activation, init_mode, bias_init, pool, use_sinusoidal_pos, fc_dims,
                                         lift_init_mode, lift_fixed_sigma, norm_first=norm_first)
    return torch_nn_feature_selection_path_generic(
        X, y, alpha, T, build_model=_build, true_idx=true_idx,
        batch_size=batch_size, lr=lr, weight_decay=weight_decay, loss=loss,
        psi_method=psi_method, seed_split=seed_split, seed_model1=seed_model1, seed_model2=seed_model2,
        compute_every=compute_every, xi_batch=xi_batch, xi_mode=xi_mode, xi_scalar=xi_scalar,
        optimizer_factory=optimizer_factory
    )

# ============================================================
# Multi-class synthetic data
# ============================================================
def generate_rotated_softmax_data(
    m: int, n: int, K: int = 3, seed: int = 0,
    twist: float = 0.5, tau: float = 0.1, basis_seed: int = 314159,
    label_mode: str = "softmax", return_details: bool = False,
):
    """Balanced three-class teacher with nonlinear, rotating boundaries.

    X ~ N(0, I), Z = X @ B, and theta = twist * Z[:, 2].
    U = cos(theta)*Z1 - sin(theta)*Z2, V = sin(theta)*Z1 + cos(theta)*Z2.
    The logits are (-U, sqrt(3)*V, U) / tau.

    B is fixed across data seeds. Its first column has equal absolute entries
    on S = {0, ..., n/2-1}; the other columns are orthogonal on the same S.
    All three columns vanish outside S. Population class proportions are 1/3.
    For softmax labels, E grad(log(p2/p0)) = 2*exp(-twist**2/2)*B[:,0]/tau.
    Argmax labels are deterministic; finite reference logits then are NOT the
    true log-odds, and their gradient is only a diagnostic reference.

    Returns (X, y, true_idx, B), plus a details dict if requested.
    Uses only NumPy and never standardizes using the generated sample.
    """
    m = _positive_int("m", m)
    n = _positive_int("n", n)
    K = _positive_int("K", K)
    if m < 2 or m % 2 or n < 6 or n % 2:
        raise ValueError("Use even m >= 2 and even n >= 6.")
    if K != 3:
        raise ValueError("The rotating-boundary teacher requires K=3.")
    if not np.isscalar(twist) or not np.isfinite(twist) or twist < 0:
        raise ValueError("twist must be finite and nonnegative.")
    if not np.isscalar(tau) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and strictly positive, including argmax mode.")
    if (isinstance(basis_seed, (bool, np.bool_))
            or not isinstance(basis_seed, (int, np.integer)) or basis_seed < 0):
        raise ValueError("basis_seed must be a nonnegative integer.")
    if label_mode not in ("softmax", "argmax"):
        raise ValueError("label_mode must be 'softmax' or 'argmax'.")

    s = n // 2
    true_idx = np.arange(s, dtype=int)
    rng_B = np.random.default_rng(basis_seed)
    b1 = rng_B.choice(np.array([-1.0, 1.0]), size=s) / np.sqrt(s)
    # QR preserves the first direction up to sign; explicitly orient it as b1.
    Q, R = np.linalg.qr(np.column_stack((b1, rng_B.standard_normal((s, 2)))))
    if np.any(np.abs(np.diag(R)) < 1e-12):
        raise ValueError("Degenerate teacher basis; choose another basis_seed.")
    Q[:, 0] *= 1.0 if np.dot(Q[:, 0], b1) >= 0 else -1.0
    B = np.zeros((n, 3), dtype=float)
    B[:s, :] = Q

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, n))
    Z = X @ B
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        theta = float(twist) * Z[:, 2]
        U = np.cos(theta) * Z[:, 0] - np.sin(theta) * Z[:, 1]
        V = np.sin(theta) * Z[:, 0] + np.cos(theta) * Z[:, 1]
        logits = np.column_stack((-U, np.sqrt(3.0) * V, U)) / float(tau)
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite teacher logits. Check twist and tau.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(shifted)
    P /= P.sum(axis=1, keepdims=True)
    if label_mode == "softmax":
        cdf = np.cumsum(P, axis=1)
        cdf[:, -1] = 1.0
        y = (rng.random(m)[:, None] >= cdf).sum(axis=1).astype(np.int64)
        probabilities = P
    else:
        y = logits.argmax(axis=1).astype(np.int64)
        probabilities = np.eye(K)[y]

    if not return_details:
        return X, y, true_idx, B
    log_probabilities = np.log(np.maximum(probabilities, np.finfo(float).tiny))
    details = dict(
        reference_logits=logits,
        probabilities=probabilities,
        reference_contrast_mean_gradient=(
            2.0 * np.exp(-0.5 * float(twist) * float(twist)) * B[:, 0] / float(tau)
        ),
        class_counts=np.bincount(y, minlength=K),
        bayes_accuracy_on_X=float(probabilities.max(axis=1).mean()),
        bayes_ce_on_X=float(-(probabilities * log_probabilities).sum(axis=1).mean()),
    )
    return X, y, true_idx, B, details


def generate_trig_softmax_data(
    m: int, n: int, K: int = 3, seed: int = 0, tau: float = 1.0,
    alpha=None, beta=None, gamma=None, omega=1.0, nu=1.3, bias=0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multi-class softmax (trigonometric multi-index), without standardization.
    Defaults use class weights linspace(-1, 1, K) with coefficients 1, .8, .6.
    Explicit scalar coefficients are broadcast for compatibility, but a teacher
    with class-constant nonconstant terms is rejected because its probabilities
    do not depend on X and its declared nonempty true support would be false.
    Returns: X (m,n), y (m,), true_idx (n/2,), B (n,3).
    """
    m = _positive_int("m", m)
    n = _positive_int("n", n)
    K = _positive_int("K", K)
    if m < 2 or m % 2:
        raise ValueError("m must be even and at least 2.")
    if n < 6 or n % 2:
        raise ValueError("n must be even and at least 6 for the rank-three teacher.")
    if K < 2:
        raise ValueError("K must be at least 2.")
    if not np.isscalar(tau) or not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite.")
    weights = np.linspace(-1.0, 1.0, K)
    alpha = weights if alpha is None else alpha
    beta = 0.8 * weights if beta is None else beta
    gamma = 0.6 * weights if gamma is None else gamma
    rng = np.random.default_rng(seed)
    half = n // 2
    true_idx = np.arange(half, dtype=int)
    X = rng.standard_normal((m, n))
    rng_B = np.random.default_rng(314159)
    G = rng_B.standard_normal((half, 3))
    Q, _ = np.linalg.qr(G)

    B = np.zeros((n, 3))
    B[:half, :] = Q

    def to_vec(x):
        x = np.asarray(x, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Teacher coefficients and frequencies must be finite.")
        if x.ndim == 0: return np.full(K, float(x))
        if x.shape == (K,): return x.astype(float)
        raise ValueError("alpha/beta/gamma/omega/nu/bias must be scalar or shape=(K,)")

    alpha = to_vec(alpha); beta = to_vec(beta); gamma = to_vec(gamma)
    omega = to_vec(omega); nu   = to_vec(nu);   bias  = to_vec(bias)

    # Compare the variable parts analytically. A common input-dependent logit
    # cancels from softmax even if the class biases differ.
    variable_parts = []
    for k in range(K):
        sine = (float(abs(omega[k])), float(alpha[k] * np.sign(omega[k]))) if alpha[k] != 0 and omega[k] != 0 else (0.0, 0.0)
        cosine = (float(abs(nu[k])), float(beta[k])) if beta[k] != 0 and nu[k] != 0 else (0.0, 0.0)
        variable_parts.append((sine, cosine, float(gamma[k])))
    if all(part == variable_parts[0] for part in variable_parts[1:]):
        raise ValueError("All class logits have the same variable part: class probabilities are independent of X, so the declared true support is invalid.")

    t1 = X @ B[:,0]; t2 = X @ B[:,1]; t3 = X @ B[:,2]  # (m,)
    H = np.empty((m, K), dtype=float)
    for k in range(K):
        H[:,k] = alpha[k]*np.sin(omega[k]*t1) + beta[k]*np.cos(nu[k]*t2) + gamma[k]*t3 + bias[k]
    Z = H / tau
    if not np.isfinite(Z).all():
        raise FloatingPointError("Teacher logits are nonfinite. Check coefficients and tau.")
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=P[i]) for i in range(m)], dtype=int)
    return X, y, true_idx, B
