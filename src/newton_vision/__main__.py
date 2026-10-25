"""newton_vision.__main__ - Interaktif Komut Satiri Kullanici Kilavuzu.

Kullanim:
    python -m newton_vision             # Genel kilavuz ve modul haritasi
    python -m newton_vision layers      # Tum katmanlarin listesi ve kod ornekleri
    python -m newton_vision autograd    # Tensor ve backward motoru kullanimi
    python -m newton_vision training    # Model egitim dongusu sablonu
    python -m newton_vision pytorch     # PyTorch state_dict aktarimi
    python -m newton_vision quant       # INT8 kuantizasyon rehberi
    python -m newton_vision apps        # 3 somut uygulama (Malware, Vision, Benchmark)
"""

from __future__ import annotations
import sys
import newton_vision as nv


def print_general_help():
    print("=" * 72)
    print(" NEWTON-VISION DERIN OGRENME FRAMEWORK'U - KULLANIM KILAVUZU")
    print(f" Surum: v{nv.__version__} | C++17 & AVX2 SIMD Hizlandirmali")
    print("=" * 72)
    print("\n[MODULLER VE YETENEKLER]:")
    print("  * newton_vision.optics       : Conv2d, Linear, BatchNorm, LayerNorm, MHSA...")
    print("  * newton_vision.fluxion      : Tensor, backward(), autograd, no_grad()...")
    print("  * newton_vision.gravity      : Adam, SGD, CrossEntropyLoss, MSELoss...")
    print("  * newton_vision.models       : ResNet18, MobileNetV2, VisionTransformer...")
    print("  * newton_vision.quantization : INT8 Kuantizasyon & %75 Bellek Tasarrufu...")
    print("  * newton_vision.weights      : PyTorch state_dict Agirlik Aktarimi...")
    print("  * newton_vision.data         : TensorDataset, DataLoader...")
    print("  * newton_vision.containers   : Sequential, ModuleList...")
    print("\n[DETAYLI KONU KILAVUZLARI]:")
    print("  Terminalden su komutlari calistirarak konu bazli ornek kod alabilirsiniz:")
    print("    python -m newton_vision layers      -> Katmanlar ve parametreleri")
    print("    python -m newton_vision autograd    -> Tensor ve turev motoru")
    print("    python -m newton_vision training    -> Egitim dongusu (Training loop)")
    print("    python -m newton_vision data        -> Datasetler (ImageFolder, CSV, TensorDataset)")
    print("    python -m newton_vision pytorch     -> PyTorch modelini yukleme")
    print("    python -m newton_vision quant       -> INT8 kuantizasyon ve hizlandirma")
    print("    python -m newton_vision apps        -> 3 somut arac rehberi")
    print("\n[TAM BELGE]:")
    print("  Ayrintili dokuman icin projedeki docs/guide.md dosyasini inceleyin.")
    print("=" * 72)


def print_layers_help():
    print("=" * 72)
    print(" KATMANLAR REHBERI (LAYERS API)")
    print("=" * 72)
    print("""
1. Dogrusal Katman:
   fc = nv.Linear(in_features=128, out_features=10, bias=True)

2. Evrisimsel Katman (2D Convolution):
   conv = nv.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)

3. Normalizasyon Katmanlari:
   bn = nv.BatchNorm2d(num_features=32)
   ln = nv.LayerNorm(normalized_shape=768)

4. Aktivasyon Fonksiyonlari:
   relu    = nv.ReLU()
   sigmoid = nv.Sigmoid()
   gelu    = nv.GELU()
   tanh    = nv.Tanh()

5. Havuzlama (Pooling):
   pool = nv.MaxPool2d(kernel_size=2, stride=2)
   avg  = nv.AvgPool2d(kernel_size=7, stride=1)

6. Attention (Transformer MHSA):
   mhsa = nv.MultiHeadAttention(embed_dim=256, num_heads=8)
""")
    print("=" * 72)


def print_training_help():
    print("=" * 72)
    print(" MODEL EGITIM DONGUSU (TRAINING LOOP RECIPE)")
    print("=" * 72)
    print("""
import numpy as np
import newton_vision as nv

# 1. Model
model = nv.Sequential(
    nv.Linear(20, 64),
    nv.ReLU(),
    nv.Linear(64, 2)
)

# 2. Kayip ve Optimizer
criterion = nv.gravity.CrossEntropyLoss()
optimizer = nv.gravity.Adam(model.parameters(), lr=0.005)

# 3. Veri
dataset = nv.TensorDataset(X_data, y_labels)
loader = nv.DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Egitim
model.train()
for epoch in range(10):
    for bx, by in loader:
        optimizer.zero_grad()       # Gradyanlari sifirla
        out = model(bx)             # Forward pass
        loss = criterion(out, by)   # Loss hesapla
        loss.backward()             # Backward pass (Turev al)
        optimizer.step()            # Agirliklari guncelle
""")
    print("=" * 72)


def print_pytorch_help():
    print("=" * 72)
    print(" PYTORCH AGIRLIK AKTARIMI (STATE DICT PARITY)")
    print("=" * 72)
    print("""
import torch
import torchvision.models as models
import newton_vision as nv

# 1. PyTorch modelinden state_dict al
pt_model = models.resnet18(weights=None)
weights = {k: v.numpy() for k, v in pt_model.state_dict().items()}

# 2. newton-vision modeline 1 satirda aktar
nv_model = nv.models.resnet18(num_classes=1000)
nv.load_state_dict(nv_model, weights)

# 3. Hizli Cikarim (Inference)
with nv.no_grad():
    output = nv_model(nv.Tensor(input_image))
""")
    print("=" * 72)


def print_quant_help():
    print("=" * 72)
    print(" INT8 KUANTIZASYON REHBERI (QUANTIZATION)")
    print("=" * 72)
    print("""
import newton_vision as nv

model = nv.models.resnet18(num_classes=1000)

# Modeli INT8 kuantize modullere donustur (%75 RAM tasarrufu)
quantized_model, stats = nv.quantize_model(model)

print(f"FP32 Boyut: {stats['fp32_memory_kb']:.1f} KB")
print(f"INT8 Boyut: {stats['int8_memory_kb']:.1f} KB")
print(f"Tasarruf  : %{stats['memory_reduction_percent']:.1f}")

with nv.no_grad():
    out = quantized_model(nv.Tensor(x))
""")
    print("=" * 72)


def print_apps_help():
    print("=" * 72)
    print(" 3 SOMUT GERCEK DUNYA UYGULAMASI (APPS)")
    print("=" * 72)
    print("""
1. Siber Guvenlik Zararli Yazilim Tespiti:
   python apps/malware_classifier.py --scan C:\\Windows\\explorer.exe
   -> PE/EXE dosyasinin byte akisini goruntuye cevirip [TEMIZ]/[ZARARLI] teshisi koyar.

2. Canli Resim Siniflandirici:
   python apps/image_predictor.py --image assets/real_mnist/number.jpg
   -> Gercek resmi okur, ASCII onizleme basar ve milisaniyede tahmin uretir.

3. PyTorch vs newton-vision Benchmark:
   python apps/pytorch_int8_benchmark.py
   -> PyTorch ile 1-e-1 esitlik ve INT8 %75 bellek kuculmesini dogrular.
""")
    print("=" * 72)


def print_data_help():
    print("=" * 72)
    print(" VERI KUMELERI REHBERI (DATASETS & DATALOADER)")
    print("=" * 72)
    print("""
1. Bellekteki Diziler (TensorDataset):
   dataset = nv.TensorDataset(X_data, y_labels)
   loader  = nv.DataLoader(dataset, batch_size=32, shuffle=True)

2. Resim Klasorleri (ImageFolder - Torchvision Standart):
   # Klasor: root/kedi/*.jpg, root/kopek/*.jpg
   dataset = nv.ImageFolder("root_dir/", target_size=(28, 28), grayscale=True)
   print("Siniflar:", dataset.classes)
   loader  = nv.DataLoader(dataset, batch_size=32, shuffle=True)

3. Tabuler CSV Verileri (CSVDataset):
   dataset = nv.CSVDataset("veriler.csv", target_column="hedef_kolon")
   loader  = nv.DataLoader(dataset, batch_size=64, shuffle=True)
""")
    print("=" * 72)


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ""

    if arg in ("layers", "layer"):
        print_layers_help()
    elif arg in ("training", "train"):
        print_training_help()
    elif arg in ("data", "dataset", "datasets"):
        print_data_help()
    elif arg in ("pytorch", "weights"):
        print_pytorch_help()
    elif arg in ("quant", "quantization", "int8"):
        print_quant_help()
    elif arg in ("apps", "app"):
        print_apps_help()
    else:
        print_general_help()


if __name__ == "__main__":
    main()
