"""newton_vision.data - Veri Kümeleri (Dataset) ve Yükleyicileri (DataLoader)."""

from __future__ import annotations
from typing import Iterator, Sequence
import numpy as np
from newton_vision.tensor import Tensor


class Dataset:
    """Tüm veri kümeleri için temel soyut sınıf."""
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Tensör ve NumPy dizilerini bir araya getiren veri kümesi.

    transform: Verilirse ilk tensörün (girdi) her örneğine ayrı ayrı uygulanır, ör.
    `nv.transforms.RandomHorizontalFlip()`. Orijinal veri değiştirilmez. (PyTorch'un
    TensorDataset'inde bu parametre yoktur; veri artırmayı tek satırda eklemek için vardır.)
    """
    def __init__(self, *tensors: Tensor | np.ndarray, transform=None):
        assert len(tensors) > 0, "TensorDataset en az bir tensör gerektirir."
        self.transform = transform
        first_len = len(tensors[0].data if isinstance(tensors[0], Tensor) else tensors[0])
        for t in tensors:
            length = len(t.data if isinstance(t, Tensor) else t)
            assert length == first_len, f"Tüm tensörler aynı boyutta olmalıdır: {length} != {first_len}"

        self.tensors = [
            t if isinstance(t, Tensor) else Tensor(t, requires_grad=False)
            for t in tensors
        ]

    def __len__(self) -> int:
        return len(self.tensors[0].data)

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        items = [t[idx] for t in self.tensors]
        if self.transform is not None:
            items[0] = Tensor(self.transform(self.tensors[0].data[idx]), requires_grad=False, device=self.tensors[0].device)
        return tuple(items)

    def _batch_first(self, batch_idx: np.ndarray) -> np.ndarray:
        """İlk tensörün batch'i; transform varsa her örneğe ayrı uygulanır."""
        data = self.tensors[0].data[batch_idx]
        if self.transform is None:
            return data
        return np.stack([self.transform(sample) for sample in data])


class DataLoader:
    """Veri kümesinden mini-batch'ler üreten ve karıştırma (shuffle) desteği sunan yükleyici.

    Örnek:
        dataset = TensorDataset(x_train, y_train)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for bx, by in loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
    """
    def __init__(self, dataset: Dataset | Sequence, batch_size: int = 1, shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, ...] | Tensor]:
        total = len(self.dataset)
        indices = np.arange(total)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, total, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break

            # Eğer TensorDataset ise tüm tensörleri batch dilimle
            if isinstance(self.dataset, TensorDataset):
                batch_tensors = []
                for k, t in enumerate(self.dataset.tensors):
                    batch_data = self.dataset._batch_first(batch_idx) if k == 0 else t.data[batch_idx]
                    batch_tensors.append(Tensor(batch_data, requires_grad=False, device=t.device))
                yield tuple(batch_tensors) if len(batch_tensors) > 1 else batch_tensors[0]
            else:
                samples = [self.dataset[idx] for idx in batch_idx]
                if isinstance(samples[0], tuple):
                    num_fields = len(samples[0])
                    collated = []
                    for f in range(num_fields):
                        field_data = np.stack([
                            s[f].data if isinstance(s[f], Tensor) else s[f]
                            for s in samples
                        ])
                        collated.append(Tensor(field_data, requires_grad=False))
                    yield tuple(collated)
                else:
                    batch_data = np.stack([
                        s.data if isinstance(s, Tensor) else s
                        for s in samples
                    ])
                    yield Tensor(batch_data, requires_grad=False)


class ImageFolder(Dataset):
    """Alt klasör yapısına göre resimleri otomatik etiketleyen veri kümesi (torchvision standardı).
    
    Örnek Klasör Yapısı:
        root/
            kedi/
                kedi1.jpg
                kedi2.png
            kopek/
                kopek1.jpg
                kopek2.png
    """
    def __init__(self, root_dir: str, target_size: tuple[int, int] = (28, 28), grayscale: bool = True):
        import os
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Veri kümesi kök dizini bulunamadı: {root_dir}")
        self.root_dir = root_dir
        self.target_size = target_size
        self.grayscale = grayscale
        
        self.classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        if len(self.classes) == 0:
            raise ValueError(f"Dizin içinde sınıf alt klasörü bulunamadı: {root_dir}")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            cls_idx = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in valid_exts:
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        import os
        fpath, label = self.samples[idx]
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("ImageFolder kütüphanesi Pillow gerektirir: pip install pillow")
            
        img = Image.open(fpath)
        if self.grayscale:
            img = img.convert("L")
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor_data = arr.reshape(1, self.target_size[1], self.target_size[0])
        else:
            img = img.convert("RGB")
            img = img.resize(self.target_size, Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            tensor_data = np.transpose(arr, (2, 0, 1))
            
        return Tensor(tensor_data, requires_grad=False), Tensor(label, requires_grad=False)


class CSVDataset(Dataset):
    """Tabüler verileri veya öznitelik CSV dosyalarını okuyan veri kümesi."""
    def __init__(self, csv_file: str, target_column: int | str = -1, has_header: bool = True):
        import os
        import csv
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_file}")
            
        self.samples_x = []
        self.samples_y = []
        
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if has_header:
                header = next(reader)
                if isinstance(target_column, str):
                    target_idx = header.index(target_column)
                else:
                    target_idx = target_column if target_column >= 0 else len(header) + target_column
            else:
                first_row = next(reader)
                target_idx = target_column if target_column >= 0 else len(first_row) + target_column
                f.seek(0)
                reader = csv.reader(f)
                
            for row in reader:
                if not row or not any(row):
                    continue
                y_val = float(row[target_idx])
                x_vals = [float(val) for i, val in enumerate(row) if i != target_idx]
                self.samples_x.append(x_vals)
                self.samples_y.append(y_val)
                
        self.X = np.array(self.samples_x, dtype=np.float32)
        self.y = np.array(self.samples_y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return Tensor(self.X[idx], requires_grad=False), Tensor(self.y[idx], requires_grad=False)
