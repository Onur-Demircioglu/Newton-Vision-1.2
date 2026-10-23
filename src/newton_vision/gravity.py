"""newton_vision.gravity - Loss Fonksiyonları & Optimizer Engine (Gravity Engine)."""

from __future__ import annotations
import numpy as np
from newton_vision.tensor import Tensor
from newton_vision.module import Module


class MSELoss(Module):
    """Ortalama Karesel Hata (Mean Squared Error) Kayıp Fonksiyonu."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), f"Geçersiz reduction: {reduction}"
        self.reduction = reduction

    def forward(self, pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> Tensor:
        p = pred if isinstance(pred, Tensor) else Tensor(pred)
        t = target if isinstance(target, Tensor) else Tensor(target)
        diff = p - t
        sq_diff = diff * diff
        if self.reduction == "mean":
            return sq_diff.mean()
        elif self.reduction == "sum":
            return sq_diff.sum()
        return sq_diff


class CrossEntropyLoss(Module):
    """Softmax ile Birleşik Çapraz Entropi (Cross-Entropy) Kayıp Fonksiyonu."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), f"Geçersiz reduction: {reduction}"
        self.reduction = reduction

    def forward(self, logits: Tensor | np.ndarray, target: Tensor | np.ndarray) -> Tensor:
        l_tensor = logits if isinstance(logits, Tensor) else Tensor(logits)
        l_data = l_tensor.data
        t_data = target.data if isinstance(target, Tensor) else np.asarray(target)

        # Numerik kararli log-softmax: log p = (z - max) - log(sum(exp(z - max)))
        # log(p + eps) KULLANILMAZ: kaybi kirpar ve (p - onehot)/N gradyaniyla tutarsizdir.
        shifted = l_data - np.max(l_data, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        probs = np.exp(log_probs)

        N = probs.shape[0]

        # Target 1D sınıf indeksleri veya 2D One-Hot olabilir
        if t_data.ndim == 1:
            target_indices = t_data.astype(np.int64)
            raw_loss = -log_probs[np.arange(N), target_indices]
        else:
            # One-Hot formatı
            raw_loss = -np.sum(t_data * log_probs, axis=-1)

        if self.reduction == "mean":
            loss_val = np.mean(raw_loss)
        elif self.reduction == "sum":
            loss_val = np.sum(raw_loss)
        else:
            loss_val = raw_loss

        req_grad = l_tensor.requires_grad
        children = (l_tensor,) if req_grad else ()
        out = Tensor(loss_val, requires_grad=req_grad, device=l_tensor.device, _children=children)

        if req_grad:
            def _backward():
                grad = probs.copy()
                if t_data.ndim == 1:
                    grad[np.arange(N), target_indices] -= 1.0
                else:
                    grad -= t_data

                if self.reduction == "mean":
                    grad /= N
                elif self.reduction == "none":
                    grad = grad * out.grad[:, None]
                    l_tensor._add_grad(grad)
                    return

                g = grad * out.grad
                l_tensor._add_grad(g)

            out._backward = _backward

        return out


def _elementwise_loss(pred: Tensor, loss_data: np.ndarray, local_grad: np.ndarray, reduction: str) -> Tensor:
    """Eleman bazlı kayıp için indirgeme (mean/sum/none) ve backward bağlantısını kurar.

    local_grad: d(loss_i)/d(pred_i), loss_data ile aynı şekilde.
    """
    if reduction == "mean":
        value, scale = np.mean(loss_data), 1.0 / loss_data.size
    elif reduction == "sum":
        value, scale = np.sum(loss_data), 1.0
    else:
        value, scale = loss_data, None

    out = Tensor(value, requires_grad=pred.requires_grad, device=pred.device, _children=(pred,) if pred.requires_grad else ())
    if pred.requires_grad:
        def _backward():
            if scale is None:
                pred._add_grad(local_grad * out.grad)
            else:
                pred._add_grad(local_grad * (scale * out.grad))

        out._backward = _backward
    return out


def _check_same_shape(name: str, pred: np.ndarray, target: np.ndarray):
    if pred.shape != target.shape:
        raise ValueError(f"{name}: tahmin {pred.shape} ve hedef {target.shape} aynı şekilde olmalı")


class L1Loss(Module):
    """Ortalama mutlak hata |pred - target| (torch.nn.L1Loss ile aynı; sign(0) = 0)."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), f"Geçersiz reduction: {reduction}"
        self.reduction = reduction

    def forward(self, pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> Tensor:
        p = pred if isinstance(pred, Tensor) else Tensor(pred)
        t = target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32)
        _check_same_shape("L1Loss", p.data, t)
        diff = p.data - t
        return _elementwise_loss(p, np.abs(diff), np.sign(diff).astype(np.float32), self.reduction)


class BCELoss(Module):
    """Olasılıklar üzerinde ikili çapraz entropi (torch.nn.BCELoss ile aynı).

    Girdi [0, 1] aralığında olmalıdır. PyTorch gibi log değerleri -100'de kırpılır.
    Sayısal kararlılık için BCEWithLogitsLoss tercih edilmelidir.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), f"Geçersiz reduction: {reduction}"
        self.reduction = reduction

    def forward(self, pred: Tensor | np.ndarray, target: Tensor | np.ndarray) -> Tensor:
        p = pred if isinstance(pred, Tensor) else Tensor(pred)
        y = target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32)
        _check_same_shape("BCELoss", p.data, y)
        if np.any(p.data < 0.0) or np.any(p.data > 1.0):
            raise ValueError("BCELoss girdisi [0, 1] aralığında olmalıdır (logit için BCEWithLogitsLoss kullanın)")
        prob = p.data.astype(np.float64)
        with np.errstate(divide="ignore"):
            log_p = np.maximum(np.log(prob), -100.0)
            log_1mp = np.maximum(np.log1p(-prob), -100.0)
        loss = -(y * log_p + (1.0 - y) * log_1mp)
        grad = (prob - y) / np.maximum(prob * (1.0 - prob), 1e-12)
        return _elementwise_loss(p, loss.astype(np.float32), grad.astype(np.float32), self.reduction)


class BCEWithLogitsLoss(Module):
    """Sigmoid + ikili çapraz entropi, sayısal kararlı (torch.nn.BCEWithLogitsLoss ile aynı).

    loss = max(x, 0) - x * y + log(1 + exp(-|x|)),  d/dx = sigmoid(x) - y
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), f"Geçersiz reduction: {reduction}"
        self.reduction = reduction

    def forward(self, logits: Tensor | np.ndarray, target: Tensor | np.ndarray) -> Tensor:
        x_t = logits if isinstance(logits, Tensor) else Tensor(logits)
        y = target.data if isinstance(target, Tensor) else np.asarray(target, dtype=np.float32)
        _check_same_shape("BCEWithLogitsLoss", x_t.data, y)
        x = x_t.data.astype(np.float64)
        loss = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
        sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))
        return _elementwise_loss(x_t, loss.astype(np.float32), (sig - y).astype(np.float32), self.reduction)


def _check_state_buffers(name: str, buffers: list[np.ndarray], params: list[Tensor]):
    if len(buffers) != len(params):
        raise ValueError(f"{name}: durumda {len(buffers)} parametre var, optimizer'da {len(params)}")
    for i, (b, p) in enumerate(zip(buffers, params)):
        if np.shape(b) != p.data.shape:
            raise ValueError(f"{name}: {i}. parametre şekli uyuşmuyor: durum {np.shape(b)}, parametre {p.data.shape}")


class SGD:
    def __init__(self, params: list[Tensor], lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def state_dict(self) -> dict:
        """Eğitime devam etmek için optimizer durumu (parametre sırasına göre)."""
        return {
            "type": "SGD",
            "lr": self.lr, "momentum": self.momentum, "weight_decay": self.weight_decay,
            "velocities": [v.copy() for v in self.velocities],
        }

    def load_state_dict(self, state: dict):
        if state.get("type") != "SGD":
            raise ValueError(f"SGD durumu bekleniyordu, alınan: {state.get('type')}")
        _check_state_buffers("SGD.load_state_dict", state["velocities"], self.params)
        self.lr, self.momentum, self.weight_decay = state["lr"], state["momentum"], state["weight_decay"]
        self.velocities = [np.asarray(v, dtype=np.float32).copy() for v in state["velocities"]]

    def step(self):
        for p, v in zip(self.params, self.velocities):
            if p.grad is not None:
                d_p = p.grad
                if self.weight_decay != 0.0:
                    d_p = d_p + self.weight_decay * p.data
                if self.momentum > 0:
                    v[...] = self.momentum * v + d_p
                    p.data -= self.lr * v
                else:
                    p.data -= self.lr * d_p


class Adam:
    def __init__(self, params: list[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def state_dict(self) -> dict:
        """Eğitime devam etmek için optimizer durumu (adım sayısı + moment tamponları)."""
        return {
            "type": type(self).__name__,
            "lr": self.lr, "beta1": self.beta1, "beta2": self.beta2, "eps": self.eps,
            "weight_decay": self.weight_decay, "t": self.t,
            "m": [m.copy() for m in self.m], "v": [v.copy() for v in self.v],
        }

    def load_state_dict(self, state: dict):
        if state.get("type") != type(self).__name__:
            raise ValueError(f"{type(self).__name__} durumu bekleniyordu, alınan: {state.get('type')}")
        _check_state_buffers("Adam.load_state_dict (m)", state["m"], self.params)
        _check_state_buffers("Adam.load_state_dict (v)", state["v"], self.params)
        self.lr, self.beta1, self.beta2 = state["lr"], state["beta1"], state["beta2"]
        self.eps, self.weight_decay, self.t = state["eps"], state["weight_decay"], int(state["t"])
        self.m = [np.asarray(m, dtype=np.float32).copy() for m in state["m"]]
        self.v = [np.asarray(v, dtype=np.float32).copy() for v in state["v"]]

    def step(self):
        self.t += 1
        for p, m_t, v_t in zip(self.params, self.m, self.v):
            if p.grad is not None:
                grad = p.grad
                if self.weight_decay != 0.0:
                    grad = grad + self.weight_decay * p.data
                m_t[...] = self.beta1 * m_t + (1.0 - self.beta1) * grad
                v_t[...] = self.beta2 * v_t + (1.0 - self.beta2) * (grad ** 2)
                m_hat = m_t / (1.0 - self.beta1 ** self.t)
                v_hat = v_t / (1.0 - self.beta2 ** self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Ayrık ağırlık azaltmalı Adam (torch.optim.AdamW ile aynı).

    Adam'daki weight_decay gradyana L2 cezası ekler; AdamW ise ağırlığı doğrudan
    p <- p - lr * weight_decay * p ile küçültür ve moment kestirimini cezadan ayırır.
    """
    def __init__(self, params: list[Tensor], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)

    def step(self):
        self.t += 1
        for p, m_t, v_t in zip(self.params, self.m, self.v):
            if p.grad is not None:
                if self.weight_decay != 0.0:
                    p.data *= np.float32(1.0 - self.lr * self.weight_decay)
                grad = p.grad
                m_t[...] = self.beta1 * m_t + (1.0 - self.beta1) * grad
                v_t[...] = self.beta2 * v_t + (1.0 - self.beta2) * (grad ** 2)
                m_hat = m_t / (1.0 - self.beta1 ** self.t)
                v_hat = v_t / (1.0 - self.beta2 ** self.t)
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_grad_norm_(params: list[Tensor], max_norm: float, norm_type: float = 2.0) -> float:
    """Tüm gradyanların toplam normunu max_norm ile sınırlar (torch.nn.utils.clip_grad_norm_ ile aynı).

    Gradyanları yerinde ölçekler ve kırpmadan önceki toplam normu döndürür.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        total = max(float(np.max(np.abs(g))) for g in grads)
    else:
        total = float(np.sum([np.sum(np.abs(g.astype(np.float64)) ** norm_type) for g in grads]) ** (1.0 / norm_type))
    if not np.isfinite(total):
        raise RuntimeError(f"clip_grad_norm_: gradyan normu sonlu değil ({total})")
    coef = min(max_norm / (total + 1e-6), 1.0)
    for g in grads:
        g *= np.float32(coef)
    return total


class _LRScheduler:
    """Öğrenme oranı zamanlayıcısı tabanı: her `step()` çağrısı bir epoch ilerler ve optimizer.lr'yi günceller."""
    def __init__(self, optimizer):
        if not hasattr(optimizer, "lr"):
            raise TypeError("Scheduler 'lr' niteliği olan bir optimizer bekler")
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.last_epoch = 0

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_last_lr(self) -> list[float]:
        return [self.optimizer.lr]

    def state_dict(self) -> dict:
        return {"base_lr": self.base_lr, "last_epoch": self.last_epoch}

    def load_state_dict(self, state: dict):
        self.base_lr = state["base_lr"]
        self.last_epoch = int(state["last_epoch"])
        self.optimizer.lr = self.get_lr()


class StepLR(_LRScheduler):
    """Her step_size epoch'ta lr'yi gamma ile çarpar (torch.optim.lr_scheduler.StepLR ile aynı)."""
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size < 1:
            raise ValueError(f"step_size >= 1 olmalı, alınan: {step_size}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class CosineAnnealingLR(_LRScheduler):
    """Kosinüs tavlama: lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * t / T_max)) / 2
    (torch.optim.lr_scheduler.CosineAnnealingLR'nin kapalı form karşılığı)."""
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        if T_max < 1:
            raise ValueError(f"T_max >= 1 olmalı, alınan: {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (1.0 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2.0
