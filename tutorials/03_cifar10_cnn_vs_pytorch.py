"""
tutorials/03_cifar10_cnn_vs_pytorch.py - CIFAR-10 CNN: newton-vision ve PyTorch, aynı koşullarda.
=============================================================================
MNIST'ten daha zor bir görevde (CIFAR-10, 32x32 renkli görüntü, 10 sınıf) aynı CNN'i newton-vision ve
PyTorch ile AYNI başlangıç ağırlıkları, AYNI karıştırma sırası ve AYNI veri artırmasıyla eğitir.
Ardından newton-vision modelinin INT8 kuantize halini ve ONNX dışa aktarımını test setinde ölçer.

Veri:
- CIFAR-10 binary sürümü (https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz, 170 MB) yoksa indirilir.
  Kesilen indirme kaldığı yerden devam eder; arşiv MD5 ile doğrulanır, yollar açılmadan önce kontrol edilir.
- Python/pickle sürümü bilinçli olarak kullanılmaz (dış kaynaklı pickle kod çalıştırabilir); ham baytlar NumPy ile okunur.

Değerlendirme dürüstlüğü:
- 50.000 eğitim görselinin son 5.000'i doğrulama (val) olarak ayrılır; test seti (10.000) yalnızca sonda ölçülür.
- Normalizasyon istatistikleri yalnızca eğitim bölümünden hesaplanır.
- Dropout kullanılmaz: iki framework'ün rastgele maskeleri farklı olacağından karşılaştırmayı bozardı.

Kullanım:
    python tutorials/03_cifar10_cnn_vs_pytorch.py                       # 15 epoch (~1.6 saat bu makinede)
    python tutorials/03_cifar10_cnn_vs_pytorch.py --epochs 1 --n-train 7000   # hızlı deneme

PyTorch ve onnxruntime opsiyoneldir; yoksa ilgili karşılaştırmalar atlanır.
Çıktılar: tutorials/output/cifar10/ (results.json UTF-8, model.nv, model.onnx; git'e dahil değil).
"""

import argparse
import copy
import hashlib
import json
import os
import tarfile
import time
import urllib.request

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
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_MD5 = "c32a1d4ab5d03f1284b67883e8d87530"   # CIFAR-10 sayfasında yayımlanan değer (binary sürüm)
CLASSES = ["uçak", "otomobil", "kuş", "kedi", "geyik", "köpek", "kurbağa", "at", "gemi", "kamyon"]
N_VAL = 5000


# ------------------------------------------------------------------ veri: indirme, doğrulama, okuma
def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, retries=5):
    """Kaldığı yerden devam eden indirme (HTTP Range); yavaş sunucuda kesilirse yeniden dener."""
    for attempt in range(1, retries + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resumed = have and resp.status == 206
                with open(path, "ab" if resumed else "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
            return
        except OSError as e:
            print(f"  indirme kesildi ({e}); deneme {attempt}/{retries}", flush=True)
    raise RuntimeError(f"{url} indirilemedi")


def safe_extract(archive, dest):
    with tarfile.open(archive) as tar:
        for m in tar.getmembers():
            parts = m.name.replace("\\", "/").split("/")
            if m.name.startswith(("/", "\\")) or ".." in parts or not (m.isfile() or m.isdir()):
                raise RuntimeError(f"Beklenmeyen arşiv üyesi: {m.name}")
        tar.extractall(dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def prepare_cifar10(data_dir, url=CIFAR_URL):
    batches = os.path.join(data_dir, "cifar-10-batches-bin")
    if os.path.isdir(batches):
        return batches
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "cifar-10-binary.tar.gz")
    if not (os.path.exists(archive) and md5sum(archive) == CIFAR_MD5):
        print(f"CIFAR-10 indiriliyor: {url} (170 MB) -> {archive}", flush=True)
        download(url, archive)
    got = md5sum(archive)
    if got != CIFAR_MD5:
        raise RuntimeError(f"MD5 uyuşmuyor: {got} != {CIFAR_MD5}. Dosyayı silip yeniden deneyin: {archive}")
    safe_extract(archive, data_dir)
    return batches


def load_batches(root, names):
    raw = np.concatenate([np.fromfile(os.path.join(root, n), dtype=np.uint8) for n in names]).reshape(-1, 3073)
    return raw[:, 1:].reshape(-1, 3, 32, 32), raw[:, 0].astype(np.int64)


def augment(xb, rng, pad_val):
    """RandomCrop(32, padding=4, siyah dolgu) + RandomHorizontalFlip(0.5); batch iki framework'e aynen verilir."""
    padded = np.empty((len(xb), 3, 40, 40), dtype=np.float32)
    padded[:] = pad_val[None, :, None, None]
    padded[:, :, 4:36, 4:36] = xb
    offs = rng.integers(0, 9, size=(len(xb), 2))
    flips = rng.random(len(xb)) < 0.5
    out = np.empty_like(xb)
    for i, ((dy, dx), f) in enumerate(zip(offs, flips)):
        crop = padded[i, :, dy:dy + 32, dx:dx + 32]
        out[i] = crop[:, :, ::-1] if f else crop
    return out


# ------------------------------------------------------------------ modeller
def build_nv():
    def block(cin, cout):
        return [nv.Conv2d(cin, cout, 3, padding=1), nv.BatchNorm2d(cout), nv.ReLU()]
    return nv.Sequential(
        *block(3, 32), nv.MaxPool2d(2),
        *block(32, 64), *block(64, 64), nv.MaxPool2d(2),
        *block(64, 128), *block(128, 128), nv.MaxPool2d(2),
        nv.Flatten(), nv.Linear(128 * 4 * 4, 256), nv.BatchNorm1d(256), nv.ReLU(), nv.Linear(256, 10))


def build_torch_like(nv_model):
    """Aynı mimari; newton-vision başlangıç ağırlıkları kopyalanır (Linear: (in,out) -> (out,in))."""
    def block(cin, cout):
        return [torch.nn.Conv2d(cin, cout, 3, padding=1), torch.nn.BatchNorm2d(cout), torch.nn.ReLU()]
    t_model = torch.nn.Sequential(
        *block(3, 32), torch.nn.MaxPool2d(2),
        *block(32, 64), *block(64, 64), torch.nn.MaxPool2d(2),
        *block(64, 128), *block(128, 128), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(128 * 4 * 4, 256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(),
        torch.nn.Linear(256, 10))
    with torch.no_grad():
        for nv_l, t_l in zip(nv_model, t_model):
            if isinstance(nv_l, (nv.Conv2d, nv.Linear)):
                w = nv_l.w.data.T if isinstance(nv_l, nv.Linear) else nv_l.w.data
                t_l.weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))
                t_l.bias.copy_(torch.from_numpy(nv_l.bias.data))
    return t_model


def nv_predict(model, x):
    model.eval()
    with nv.no_grad():
        return np.concatenate([model(nv.Tensor(x[i:i + 500])).data for i in range(0, len(x), 500)])


def t_predict(t_model, x):
    t_model.eval()
    with torch.no_grad():
        return np.concatenate([t_model(torch.from_numpy(x[i:i + 500])).numpy() for i in range(0, len(x), 500)])


def main():
    p = argparse.ArgumentParser(description="CIFAR-10 CNN: newton-vision ve PyTorch, aynı koşullarda")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-train", type=int, default=50000, help="kullanılacak eğitim görseli (son 5000'i val)")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default=os.path.join(HERE, "data", "cifar10"))
    p.add_argument("--out", default=os.path.join(HERE, "output", "cifar10"))
    p.add_argument("--no-torch", action="store_true", help="PyTorch karşılaştırmasını atla")
    args = p.parse_args()
    use_torch = HAS_TORCH and not args.no_torch
    if args.n_train <= N_VAL:
        p.error(f"--n-train {N_VAL}'den büyük olmalı (son {N_VAL} görsel doğrulama için ayrılır)")

    root = prepare_cifar10(args.data)
    x_all, y_all = load_batches(root, [f"data_batch_{i}.bin" for i in range(1, 6)])
    x_all, y_all = x_all[:args.n_train], y_all[:args.n_train]
    x_te_u8, y_te = load_batches(root, ["test_batch.bin"])
    x_tr_u8, y_tr, x_val_u8, y_val = x_all[:-N_VAL], y_all[:-N_VAL], x_all[-N_VAL:], y_all[-N_VAL:]

    # Normalizasyon istatistikleri yalnızca eğitim bölümünden
    mean = (x_tr_u8.astype(np.float64) / 255.0).mean(axis=(0, 2, 3)).astype(np.float32).reshape(1, 3, 1, 1)
    std = (x_tr_u8.astype(np.float64) / 255.0).std(axis=(0, 2, 3)).astype(np.float32).reshape(1, 3, 1, 1)
    norm = lambda x: ((x.astype(np.float32) / 255.0) - mean) / std
    x_tr, x_val, x_te = norm(x_tr_u8), norm(x_val_u8), norm(x_te_u8)
    pad_val = (-mean / std).reshape(3)

    np.random.seed(args.seed)
    nv_model = build_nv()
    n_params = int(sum(q.data.size for q in nv_model.parameters()))
    nv_opt = nv.AdamW(nv_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    nv_sched = nv.CosineAnnealingLR(nv_opt, T_max=args.epochs)
    nv_loss_fn = nv.CrossEntropyLoss()
    if use_torch:
        t_model = build_torch_like(nv_model)
        if sum(q.numel() for q in t_model.parameters()) != n_params:
            raise RuntimeError("PyTorch ve newton-vision modellerinin parametre sayıları farklı")
        t_opt = torch.optim.AdamW(t_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        t_sched = torch.optim.lr_scheduler.CosineAnnealingLR(t_opt, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "results.json")
    log = {"config": {**vars(args), "n_train_used": len(x_tr), "n_val": N_VAL, "n_test": len(x_te), "params": n_params,
                      "mean": mean.ravel().tolist(), "std": std.ravel().tolist(),
                      "pytorch": torch.__version__ if use_torch else None}, "epochs": []}

    def save_log():
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1, ensure_ascii=False)

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
        s = dict(nv_loss=0.0, t_loss=0.0, nv_c=0, t_c=0)
        for b in range(0, len(order), args.batch):
            idx = order[b:b + args.batch]
            xb, yb = augment(x_tr[idx], rng, pad_val), y_tr[idx]

            t0 = time.perf_counter()
            logits = nv_model(nv.Tensor(xb))
            loss = nv_loss_fn(logits, yb)
            nv_opt.zero_grad()
            loss.backward()
            nv.clip_grad_norm_(nv_model.parameters(), args.clip)
            nv_opt.step()
            nv_time += time.perf_counter() - t0
            s["nv_loss"] += loss.item() * len(idx)
            s["nv_c"] += int((logits.data.argmax(1) == yb).sum())

            if use_torch:
                t0 = time.perf_counter()
                tl = t_model(torch.from_numpy(xb))
                tloss = torch.nn.functional.cross_entropy(tl, torch.from_numpy(yb))
                t_opt.zero_grad()
                tloss.backward()
                torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.clip)
                t_opt.step()
                t_time += time.perf_counter() - t0
                s["t_loss"] += tloss.item() * len(idx)
                s["t_c"] += int((tl.detach().numpy().argmax(1) == yb).sum())
        nv_sched.step()
        if use_torch:
            t_sched.step()

        n = len(x_tr)
        row = dict(epoch=ep, nv_train_loss=s["nv_loss"] / n, nv_train_acc=s["nv_c"] / n,
                   nv_val_acc=float((nv_predict(nv_model, x_val).argmax(1) == y_val).mean()), nv_cum_time_s=nv_time)
        if use_torch:
            row.update(t_train_loss=s["t_loss"] / n, t_train_acc=s["t_c"] / n,
                       t_val_acc=float((t_predict(t_model, x_val).argmax(1) == y_val).mean()), t_cum_time_s=t_time)
        log["epochs"].append(row)
        save_log()
        msg = f"Epoch {ep:2d}/{args.epochs} | nv loss {row['nv_train_loss']:.4f} val %{100 * row['nv_val_acc']:.2f}"
        if use_torch:
            msg += f" | torch loss {row['t_train_loss']:.4f} val %{100 * row['t_val_acc']:.2f}"
        print(msg + f" | nv süre {nv_time / 60:.1f} dk", flush=True)

    conv_change = float(np.abs(nv_model[0].w.data - conv1_w0).max())
    if conv_change == 0.0:
        raise RuntimeError("İlk evrişim katmanının ağırlıkları değişmedi: gradyan grafı kopuk!")

    # -------------------------------------------------------------- test (yalnızca bir kez)
    nv_logits = nv_predict(nv_model, x_te)
    nv_pred = nv_logits.argmax(1)
    cm = np.zeros((10, 10), dtype=int)
    for a, b in zip(y_te, nv_pred):
        cm[a, b] += 1
    test = dict(nv_acc=float((nv_pred == y_te).mean()), errors=int((nv_pred != y_te).sum()),
                per_class_acc={c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASSES)},
                confusion_matrix=cm.tolist(), conv1_weight_change=conv_change)
    if use_torch:
        t_pred = t_predict(t_model, x_te).argmax(1)
        test.update(t_acc=float((t_pred == y_te).mean()), pred_agreement_nv_vs_torch=float((t_pred == nv_pred).mean()),
                    t_per_class_acc={c: float(((t_pred == i) & (y_te == i)).sum() / (y_te == i).sum())
                                     for i, c in enumerate(CLASSES)})

    nv_int8 = copy.deepcopy(nv_model)
    q = nv.quantize_model(nv_int8, per_channel=True)
    test.update(int8_acc=float((nv_predict(nv_int8, x_te).argmax(1) == y_te).mean()),
                fp32_weight_kb=q["fp32_memory_kb"], int8_weight_kb=q["int8_memory_kb"])

    nv.save_model(nv_model, os.path.join(args.out, "model.nv"))
    if HAS_ONNX:
        onnx_path = os.path.join(args.out, "model.onnx")
        nv.export_onnx(nv_model, x_te[:500], onnx_path)   # sabit batch 500
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_logits = np.concatenate([sess.run(None, {"input": x_te[i:i + 500]})[0] for i in range(0, len(x_te), 500)])
        test.update(onnx_acc=float((onnx_logits.argmax(1) == y_te).mean()),
                    onnx_pred_agreement=float((onnx_logits.argmax(1) == nv_pred).mean()),
                    onnx_max_logit_diff=float(np.abs(onnx_logits - nv_logits).max()))

    log["test"] = test
    save_log()

    off = sorted([(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(10) for j in range(10) if i != j], reverse=True)[:3]
    print("\n================ SONUÇ (test seti, 10.000 görsel) ================")
    print(f"  newton-vision : %{100 * test['nv_acc']:.2f} ({test['errors']} hata) | eğitim {nv_time / 60:.1f} dk")
    if use_torch:
        print(f"  PyTorch       : %{100 * test['t_acc']:.2f} | eğitim {t_time / 60:.1f} dk | "
              f"tahmin uyumu %{100 * test['pred_agreement_nv_vs_torch']:.2f}")
    print(f"  INT8          : %{100 * test['int8_acc']:.2f} | ağırlık {test['fp32_weight_kb']:.0f} KB -> {test['int8_weight_kb']:.0f} KB")
    if HAS_ONNX:
        print(f"  ONNX          : %{100 * test['onnx_acc']:.2f} | tahmin uyumu %{100 * test['onnx_pred_agreement']:.2f} "
              f"| en büyük çıktı farkı {test['onnx_max_logit_diff']:.1e}")
    print("  Sınıf bazında : " + ", ".join(f"{c} %{100 * a:.1f}" for c, a in test["per_class_acc"].items()))
    print("  En çok karışan: " + ", ".join(f"{a}→{b} ({n})" for n, a, b in off))
    print(f"  Ayrıntılar    : {results_path}")


if __name__ == "__main__":
    main()
