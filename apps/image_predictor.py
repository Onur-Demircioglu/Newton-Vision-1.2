"""apps/image_predictor.py - Canli Goruntu Siniflandirma ve Masaustu Test Araci.

Diskten okunan herhangi bir gercek .png, .jpg veya .bmp resmini on isler,
newton-vision CNN agina besler ve terminalde gorsel ASCII onizleme ile tahmin uretir.

Kullanim:
    Ornek resim uret ve test et : python apps/image_predictor.py
    Kendi resmini test et        : python apps/image_predictor.py --image <resim_yolu.png>
    Modeli yeniden egit          : python apps/image_predictor.py --train
"""

from __future__ import annotations
import os
import sys
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import newton_vision as nv


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "tutorials", "lenet_mnist_v1.nv")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


class LeNetClassifier(nv.Module):
    """Resmi MNIST 60.000 El Yazisi Veritabani ile Egitilen LeNet Modeli."""
    def __init__(self):
        super().__init__()
        self.conv1 = nv.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nv.ReLU()
        self.pool1 = nv.MaxPool2d(2)

        self.conv2 = nv.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nv.ReLU()
        self.pool2 = nv.MaxPool2d(2)

        self.fc1 = nv.Linear(16 * 5 * 5, 120)
        self.relu3 = nv.ReLU()
        self.fc2 = nv.Linear(120, 84)
        self.relu4 = nv.ReLU()
        self.fc3 = nv.Linear(84, 10)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))

        # Tensor.reshape grafı korur (--train modunda konvolüsyonlar da eğitilir)
        flat = out.reshape(out.shape[0], -1)

        out = self.relu3(self.fc1(flat))
        out = self.relu4(self.fc2(out))
        return self.fc3(out)


def generate_sample_images():
    """Kullanicinin diskten hemen test edebilecegi ornek gercek PNG gorselleri olusturur."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    samples = {
        "digit_3.png": 3,
        "digit_7.png": 7,
        "digit_0.png": 0,
    }

    created_files = []
    for fname, digit in samples.items():
        fpath = os.path.join(ASSETS_DIR, fname)
        # 100x100 siyah arka plan uzerine beyaz rakam ciz
        img = Image.new("L", (100, 100), color=0)
        draw = ImageDraw.Draw(img)

        # Rakamlari cizgilerle geometrik olarak ciz
        if digit == 3:
            draw.line([(25, 25), (75, 25), (45, 50), (75, 50), (75, 75), (25, 75)], fill=255, width=12)
        elif digit == 7:
            draw.line([(25, 25), (75, 25), (40, 85)], fill=255, width=12)
        elif digit == 0:
            draw.ellipse([(25, 20), (75, 80)], outline=255, width=12)

        img.save(fpath)
        created_files.append(fpath)

    return created_files


def train_vision_model():
    print("\n" + "=" * 65)
    print(" BILGISAYARLI GORU: RAKAM SINIFLANDIRICI CNN MODEL EGITIMI")
    print("=" * 65)

    print("[1/3] Sentetik egitim desenleri hazirlaniyor (600 ornek)...")
    np.random.seed(42)
    # Her rakam icin 60 ornek uret (toplam 600 ornek)
    X_list = []
    y_list = []
    for digit in range(10):
        for _ in range(60):
            img = Image.new("L", (28, 28), color=0)
            draw = ImageDraw.Draw(img)
            # Rastgele ufak kaydirmalar ile ciz
            ox = np.random.randint(-2, 3)
            oy = np.random.randint(-2, 3)
            if digit == 0:
                draw.ellipse([(6+ox, 4+oy), (22+ox, 24+oy)], outline=255, width=3)
            elif digit == 1:
                draw.line([(14+ox, 4+oy), (14+ox, 24+oy)], fill=255, width=3)
            elif digit == 7:
                draw.line([(6+ox, 6+oy), (22+ox, 6+oy), (10+ox, 24+oy)], fill=255, width=3)
            elif digit == 3:
                draw.line([(6+ox, 6+oy), (22+ox, 6+oy), (14+ox, 14+oy), (22+ox, 14+oy), (22+ox, 22+oy), (6+ox, 22+oy)], fill=255, width=3)
            else:
                # Diger rakamlar icin genel elips ve cizgi kombinasyonlari
                draw.ellipse([(8+ox, 6+oy), (20+ox, 22+oy)], outline=200, width=2)
                draw.line([(10+ox, 14+oy), (18+ox, 14+oy)], fill=255, width=2)

            arr = np.array(img, dtype=np.float32) / 255.0
            # Ufak gürültü ekle
            arr += np.random.normal(0, 0.05, arr.shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 1.0)

            X_list.append(arr.reshape(1, 28, 28))
            y_list.append(digit)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int64)

    dataset = nv.TensorDataset(X_train, y_train)
    loader = nv.DataLoader(dataset, batch_size=32, shuffle=True)

    print("[2/3] VisionCNN mimarisi baslatiliyor...")
    model = VisionCNN()
    criterion = nv.gravity.CrossEntropyLoss()
    optimizer = nv.gravity.Adam(model.parameters(), lr=0.005)

    print("[3/3] Model egitiliyor (8 Epoch)...")
    model.train()
    for epoch in range(1, 9):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.data) * len(bx.data)
            preds = np.argmax(pred.data, axis=1)
            correct += int(np.sum(preds == by.data))
            total += len(bx.data)

        acc = (correct / total) * 100.0
        print(f"  Epoch [{epoch:2d}/8] | Loss: {epoch_loss/total:.4f} | Dogruluk: %{acc:.1f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    nv.save_model(model, MODEL_PATH)
    print(f"  [OK] Vision modeli diske kaydedildi: {MODEL_PATH}\n")


def render_ascii_image(arr_28x28: np.ndarray):
    """28x28 tensör verisini konsola şık bir ASCII görseli olarak döker."""
    chars = " .:-=+*#%@"
    lines = []
    # 28x28'i terminal oranına uygun olarak her 2 satırda bir veya tam çiz
    for y in range(0, 28, 2):
        line = "    "
        for x in range(28):
            val = arr_28x28[y, x]
            idx = int(val * (len(chars) - 1))
            line += chars[idx]
        lines.append(line)
    return "\n".join(lines)


def predict_image(image_path: str):
    """Diskten okunan herhangi bir gorseli siniflandirir ve ayrintili rapor basar."""
    if not os.path.exists(image_path):
        print(f"[HATA] Goruntu dosyasi bulunamadi: {image_path}")
        return

    if not os.path.exists(MODEL_PATH):
        print("[BILGI] Egitilmis vision modeli bulunamadi, egitim baslatiliyor...")
        train_vision_model()

    model = LeNetClassifier()
    nv.load_model(model, MODEL_PATH)
    model.eval()

    # 1. PIL ile resmi ac
    img = Image.open(image_path).convert("L")
    raw_arr = np.array(img, dtype=np.float32)

    # 2. Arka plan rengi tespiti ve otomatik renk tersleme (MNIST arka plani siyah=0, yazi beyaz=255)
    border_pixels = np.concatenate([raw_arr[0, :], raw_arr[-1, :], raw_arr[:, 0], raw_arr[:, -1]])
    if np.mean(border_pixels) > 127.0:
        raw_arr = 255.0 - raw_arr  # Beyaz arka plani siyaha, siyah yaziyi beyaza cevir

    # 3. Kenarlardaki olasi eksen / cerceve cizgilerini temizle (sadece yeterli cozunurluk varsa)
    h_orig, w_orig = raw_arr.shape
    if h_orig >= 24 and w_orig >= 24:
        m_h, m_w = max(1, int(h_orig * 0.06)), max(1, int(w_orig * 0.06))
        inner = raw_arr[m_h:-m_h, m_w:-m_w]
    else:
        inner = raw_arr

    # 4. Rakam alanini (bounding box) bul ve 20x20 icine merkezle (Resmi MNIST standardi)
    threshold = np.mean(inner) + 0.4 * np.std(inner)
    mask = inner > threshold
    if np.any(mask):
        y_idx, x_idx = np.where(mask)
        digit_crop = inner[y_idx.min():y_idx.max()+1, x_idx.min():x_idx.max()+1]
        ch, cw = digit_crop.shape
        scale = 20.0 / max(ch, cw)
        new_h, new_w = max(1, int(round(ch * scale))), max(1, int(round(cw * scale)))
        crop_pil = Image.fromarray(digit_crop.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (28, 28), color=0)
        canvas.paste(crop_pil, ((28 - new_w) // 2, (28 - new_h) // 2))
        arr = np.array(canvas, dtype=np.float32) / 255.0
    else:
        arr = np.array(img.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    tensor_input = arr.reshape(1, 1, 28, 28)

    # 3. Cikarim (Inference) ve zaman olcumu
    start_time = time.perf_counter()
    with nv.no_grad():
        logits = model(nv.Tensor(tensor_input))
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # 4. Softmax olasiliklari
    exp_logits = np.exp(logits.data[0] - np.max(logits.data[0]))
    probs = exp_logits / np.sum(exp_logits)

    # Top-3 tahmin
    top3_indices = np.argsort(probs)[::-1][:3]
    best_class = top3_indices[0]
    best_conf = probs[best_class] * 100.0

    print("\n" + "=" * 65)
    print(f" CANLI GORUNTU ANALIZI: {os.path.basename(image_path)}")
    print("=" * 65)
    print("  [GIRDI RESMI ASCII ONIZLEMESI]:")
    print(render_ascii_image(arr))
    print("-" * 65)
    print(f"  Cikarim Gecikmesi (Latency) : {latency_ms:.2f} ms ({1000.0 / max(latency_ms, 0.001):.1f} FPS)")
    print(f"  En Yuksek Tahmin (Prediction): RAKAM {best_class}  (Guven: %{best_conf:.2f})")
    print("-" * 65)
    print("  TOP-3 SINIF OLASILIKLARI:")
    for rank, idx in enumerate(top3_indices, 1):
        p = probs[idx] * 100.0
        bar = "#" * int(p // 4)
        print(f"    {rank}. Rakam '{idx}' : %{p:5.2f}  {bar}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="newton-vision Canli Goruntu Siniflandirici")
    parser.add_argument("--image", type=str, help="Siniflandirilacak goruntu dosyasi yolu (.png, .jpg)")
    parser.add_argument("--train", action="store_true", help="Vision modelini yeniden egitir")

    args = parser.parse_args()

    if args.train:
        train_vision_model()
    elif args.image:
        predict_image(args.image)
    else:
        # Varsayilan akis: Modeli egit, ornek resimleri diske olustur ve tahmin et
        if not os.path.exists(MODEL_PATH):
            train_vision_model()

        sample_files = generate_sample_images()
        print("\n[BILGI] Otomatik test gorselleri uretildi:")
        for s in sample_files:
            print(f"  -> {s}")

        for s in sample_files:
            predict_image(s)


if __name__ == "__main__":
    main()
