"""
tutorials/02_mnist_cnn_vs_pytorch.py - MNIST CNN: newton-vision ve PyTorch, aynı koşullarda.
=============================================================================
Aynı CNN'i newton-vision ve PyTorch ile AYNI başlangıç ağırlıkları, AYNI karıştırma sırası ve
AYNI veri artırmasıyla eğitir; iki framework'ün öğrenme eğrilerini, test sonucunu ve süresini karşılaştırır.
Ardından newton-vision modelinin INT8 kuantize halini ve ONNX dışa aktarımını test setinde ölçer.

Değerlendirme dürüstlüğü:
- Eğitim kümesinin son 5000 görseli doğrulama (val) olarak ayrılır.
- Test seti (10.000 görsel) yalnızca en sonda bir kez ölçülür.
- Dropout kullanılmaz: iki framework'ün rastgele maskeleri farklı olacağından karşılaştırmayı bozardı.

Kullanım:
    python tutorials/02_mnist_cnn_vs_pytorch.py                 # 10 epoch, tam veri (~16 dk bu makinede)
    python tutorials/02_mnist_cnn_vs_pytorch.py --epochs 1 --n-train 8000   # hızlı deneme

Gereksinimler: MNIST dosyaları tutorials/data/ altında (yoksa önce tutorials/train_and_eval_lenet.py indirir).
PyTorch ve onnxruntime opsiyoneldir; yoksa ilgili karşılaştırmalar atlanır.
Çıktılar: tutorials/output/ (results.json, model.nv, model.onnx; git'e dahil değil).
"""

import argparse
import copy
import gzip
import json
import os
import time

import numpy as np
import newton_vision as nv

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import onnxruntime as ort
    import onnx  # noqa: F401  (nv.export_onnx için)
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN, STD = 0.1307, 0.3081

parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--n-train", type=int, default=60000, help="kullanılacak eğitim görseli (son 5000'i val)")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--clip", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=os.path.join(HERE, "output"))
parser.add_argument("--no-torch", action="store_true", help="PyTorch karşılaştırmasını atla")
args = parser.parse_args()
use_torch = HAS_TORCH and not args.no_torch
N_VAL = 5000
if args.n_train <= N_VAL:
    parser.error(f"--n-train {N_VAL}'den büyük olmalı (son {N_VAL} görsel doğrulama için ayrılır)")


def read_idx(fname, offset):
    path = os.path.join(HERE, "data", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} yok. İndirmek için önce `python tutorials/train_and_eval_lenet.py` çalıştırın.")
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def normalize(x):
    return ((x.astype(np.float32) / 255.0) - MEAN) / STD


x_all = normalize(read_idx("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)[:args.n_train])
y_all = read_idx("train-labels-idx1-ubyte.gz", 8)[:args.n_train].astype(np.int64)
x_tr, y_tr, x_val, y_val = x_all[:-N_VAL], y_all[:-N_VAL], x_all[-N_VAL:], y_all[-N_VAL:]
x_te = normalize(read_idx("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28))
y_te = read_idx("t10k-labels-idx1-ubyte.gz", 8).astype(np.int64)
PAD_VALUE = (0.0 - MEAN) / STD   # siyah arka plan (normalize edilmiş)


def augment(xb, rng):
    """RandomCrop(28, padding=2): her örnek ±2 piksel kaydırılır. Batch iki framework'e aynen verilir."""
    padded = np.pad(xb, ((0, 0), (0, 0), (2, 2), (2, 2)), constant_values=PAD_VALUE)
    out = np.empty_like(xb)
    for i, (dy, dx) in enumerate(rng.integers(0, 5, size=(len(xb), 2))):
        out[i] = padded[i, :, dy:dy + 28, dx:dx + 28]
    return out


# ------------------------------------------------------------------ modeller (aynı mimari, aynı başlangıç)
np.random.seed(args.seed)
nv_model = nv.Sequential(
    nv.Conv2d(1, 16, 3, padding=1), nv.BatchNorm2d(16), nv.ReLU(), nv.MaxPool2d(2),
    nv.Conv2d(16, 32, 3, padding=1), nv.BatchNorm2d(32), nv.ReLU(), nv.MaxPool2d(2),
    nv.Flatten(),
    nv.Linear(32 * 7 * 7, 128), nv.BatchNorm1d(128), nv.ReLU(),
    nv.Linear(128, 10),
)
n_params = int(sum(p.data.size for p in nv_model.parameters()))
nv_opt = nv.AdamW(nv_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
nv_sched = nv.CosineAnnealingLR(nv_opt, T_max=args.epochs)
nv_loss_fn = nv.CrossEntropyLoss()

if use_torch:
    t_model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 7 * 7, 128), torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )
    with torch.no_grad():   # newton-vision başlangıç ağırlıklarını kopyala (Linear: (in,out) -> (out,in))
        for nv_l, t_l in zip(nv_model, t_model):
            if isinstance(nv_l, (nv.Conv2d, nv.Linear)):
                w = nv_l.w.data.T if isinstance(nv_l, nv.Linear) else nv_l.w.data
                t_l.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
                t_l.bias.copy_(torch.from_numpy(nv_l.bias.data))
    if sum(p.numel() for p in t_model.parameters()) != n_params:
        raise RuntimeError("PyTorch ve newton-vision modellerinin parametre sayıları farklı")
    t_opt = torch.optim.AdamW(t_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_sched = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, T_max=args.epochs)


def predict(model, x):
    model.eval()
    with nv.no_grad():
        return np.concatenate([model(nv.Tensor(x[i:i + 1000])).data for i in range(0, len(x), 1000)])


def t_predict(x):
    t_model.eval()
    with torch.no_grad():
        return np.concatenate([t_model(torch.from_numpy(x[i:i + 1000])).numpy() for i in range(0, len(x), 1000)])


# ------------------------------------------------------------------ eğitim
os.makedirs(args.out, exist_ok=True)
log = {"config": {**vars(args), "n_train_used": len(x_tr), "n_val": N_VAL, "n_test": len(x_te), "params": n_params,
                  "pytorch": torch.__version__ if use_torch else None}, "epochs": []}
print(f"Parametre: {n_params:,} | train {len(x_tr)} / val {N_VAL} / test {len(x_te)} | "
      f"PyTorch karşılaştırması: {'açık' if use_torch else 'kapalı'}", flush=True)
conv1_w0 = nv_model[0].w.data.copy()
nv_time = t_time = 0.0

for ep in range(1, args.epochs + 1):
    rng = np.random.default_rng(args.seed + ep)
    order = rng.permutation(len(x_tr))
    nv_model.train()
    if use_torch:
        t_model.train()
    s = dict(nv_loss=0.0, t_loss=0.0, nv_correct=0, t_correct=0)
    for b in range(0, len(order), args.batch):
        idx = order[b:b + args.batch]
        xb, yb = augment(x_tr[idx], rng), y_tr[idx]

        t0 = time.perf_counter()
        logits = nv_model(nv.Tensor(xb))
        loss = nv_loss_fn(logits, yb)
        nv_opt.zero_grad()
        loss.backward()
        nv.clip_grad_norm_(nv_model.parameters(), args.clip)
        nv_opt.step()
        nv_time += time.perf_counter() - t0
        s["nv_loss"] += loss.item() * len(idx)
        s["nv_correct"] += int((logits.data.argmax(1) == yb).sum())

        if use_torch:
            t0 = time.perf_counter()
            t_logits = t_model(torch.from_numpy(xb))
            t_loss = torch.nn.functional.cross_entropy(t_logits, torch.from_numpy(yb))
            t_opt.zero_grad()
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.clip)
            t_opt.step()
            t_time += time.perf_counter() - t0
            s["t_loss"] += t_loss.item() * len(idx)
            s["t_correct"] += int((t_logits.detach().numpy().argmax(1) == yb).sum())
    nv_sched.step()
    if use_torch:
        t_sched.step()

    n = len(x_tr)
    row = dict(epoch=ep, nv_train_loss=s["nv_loss"] / n, nv_train_acc=s["nv_correct"] / n,
               nv_val_acc=float((predict(nv_model, x_val).argmax(1) == y_val).mean()), nv_cum_time_s=nv_time)
    if use_torch:
        row.update(t_train_loss=s["t_loss"] / n, t_train_acc=s["t_correct"] / n,
                   t_val_acc=float((t_predict(x_val).argmax(1) == y_val).mean()), t_cum_time_s=t_time)
    log["epochs"].append(row)
    msg = f"Epoch {ep:2d}/{args.epochs} | nv loss {row['nv_train_loss']:.4f} val %{100 * row['nv_val_acc']:.2f}"
    if use_torch:
        msg += f" | torch loss {row['t_train_loss']:.4f} val %{100 * row['t_val_acc']:.2f}"
    print(msg + f" | nv süre {nv_time:.0f} sn", flush=True)

conv_change = float(np.abs(nv_model[0].w.data - conv1_w0).max())
if conv_change == 0.0:
    raise RuntimeError("İlk evrişim katmanının ağırlıkları değişmedi: gradyan grafı kopuk!")

# ------------------------------------------------------------------ test (yalnızca bir kez)
nv_logits = predict(nv_model, x_te)
nv_pred = nv_logits.argmax(1)
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_te, nv_pred):
    cm[t, p] += 1
test = dict(nv_acc=float((nv_pred == y_te).mean()), errors=int((nv_pred != y_te).sum()),
            conv1_weight_change=conv_change, confusion_matrix=cm.tolist(),
            precision=(np.diag(cm) / np.maximum(cm.sum(0), 1)).tolist(),
            recall=(np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist())

if use_torch:
    t_logits_te = t_predict(x_te)
    test.update(t_acc=float((t_logits_te.argmax(1) == y_te).mean()),
                pred_agreement_nv_vs_torch=float((t_logits_te.argmax(1) == nv_pred).mean()))

nv_int8 = copy.deepcopy(nv_model)
qstats = nv.quantize_model(nv_int8, per_channel=True)
test.update(int8_acc=float((predict(nv_int8, x_te).argmax(1) == y_te).mean()),
            fp32_weight_kb=qstats["fp32_memory_kb"], int8_weight_kb=qstats["int8_memory_kb"])

nv.save_model(nv_model, os.path.join(args.out, "model.nv"))
if HAS_ONNX:
    onnx_path = os.path.join(args.out, "model.onnx")
    nv.export_onnx(nv_model, x_te[:1000], onnx_path)   # sabit batch 1000
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_logits = np.concatenate([sess.run(None, {"input": x_te[i:i + 1000]})[0] for i in range(0, len(x_te), 1000)])
    test.update(onnx_acc=float((onnx_logits.argmax(1) == y_te).mean()),
                onnx_pred_agreement=float((onnx_logits.argmax(1) == nv_pred).mean()),
                onnx_max_logit_diff=float(np.abs(onnx_logits - nv_logits).max()))

log["test"] = test
with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
    json.dump(log, f, indent=1)

print("\n================ SONUÇ (test seti, 10.000 görsel) ================")
print(f"  newton-vision : %{100 * test['nv_acc']:.2f} ({test['errors']} hata) | eğitim {nv_time:.0f} sn")
if use_torch:
    print(f"  PyTorch       : %{100 * test['t_acc']:.2f} | eğitim {t_time:.0f} sn | "
          f"tahmin uyumu %{100 * test['pred_agreement_nv_vs_torch']:.2f}")
print(f"  INT8          : %{100 * test['int8_acc']:.2f} | ağırlık {test['fp32_weight_kb']:.0f} KB -> {test['int8_weight_kb']:.0f} KB")
if HAS_ONNX:
    print(f"  ONNX          : %{100 * test['onnx_acc']:.2f} | tahmin uyumu %{100 * test['onnx_pred_agreement']:.2f} "
          f"| en büyük çıktı farkı {test['onnx_max_logit_diff']:.1e}")
print(f"  Ayrıntılar    : {os.path.join(args.out, 'results.json')}")
