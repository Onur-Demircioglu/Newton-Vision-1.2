"""newton_vision.text - Basit kelime/karakter tokenizer'ı.

Metni `nv.Embedding`'e verilebilecek tam sayı indekslerine çevirir. Alt-kelime (BPE/WordPiece)
yöntemleri içermez; kelime dağarcığı dışındaki tokenlar `<unk>` olur.

Örnek:
    tok = nv.text.Tokenizer(level="word").fit(["merhaba dünya", "merhaba newton"])
    ids = tok.encode_batch(["merhaba dünya"], max_len=8)   # (1, 8) int64, <pad> ile dolgulu
    emb = nv.Embedding(tok.vocab_size, 32, padding_idx=tok.pad_id)
"""

from __future__ import annotations
import re
from collections import Counter
from typing import Iterable, Sequence
import numpy as np

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer:
    """Kelime ("word") veya karakter ("char") düzeyinde tokenizer.

    * Kelime dağarcığı sıklığa (azalan), eşitlikte tokenin kendisine göre sıralanır: deterministiktir.
    * İlk indeksler özel tokenlara ayrılır: 0 = `<pad>`, 1 = `<unk>`.
    * lower=True: `str.lower()` kullanır (Türkçe 'I' -> 'i' olur, 'ı' değil; Python yerel ayardan bağımsızdır).
    """
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, level: str = "word", lower: bool = True, min_freq: int = 1, max_size: int | None = None):
        if level not in ("word", "char"):
            raise ValueError(f"level 'word' veya 'char' olmalı, alınan: {level!r}")
        if min_freq < 1:
            raise ValueError(f"min_freq >= 1 olmalı, alınan: {min_freq}")
        if max_size is not None and max_size < 3:
            raise ValueError(f"max_size özel tokenlarla birlikte en az 3 olmalı, alınan: {max_size}")
        self.level = level
        self.lower = lower
        self.min_freq = min_freq
        self.max_size = max_size
        self.id_to_token: list[str] = [self.PAD, self.UNK]
        self.token_to_id: dict[str, int] = {self.PAD: 0, self.UNK: 1}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError(f"Metin str olmalı, alınan: {type(text)}")
        if self.lower:
            text = text.lower()
        return list(text) if self.level == "char" else _WORD_RE.findall(text)

    def fit(self, texts: Iterable[str]) -> Tokenizer:
        counts = Counter(tok for text in texts for tok in self.tokenize(text))
        tokens = sorted((t for t, c in counts.items() if c >= self.min_freq and t not in self.token_to_id),
                        key=lambda t: (-counts[t], t))
        if self.max_size is not None:
            tokens = tokens[: self.max_size - 2]
        self.id_to_token = [self.PAD, self.UNK] + tokens
        self.token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        return self

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(t, self.unk_id) for t in self.tokenize(text)]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            i = int(i)
            if not 0 <= i < self.vocab_size:
                raise IndexError(f"Token id {i} kelime dağarcığı dışında (vocab_size={self.vocab_size})")
            if skip_special and i in (self.pad_id, self.unk_id):
                continue
            tokens.append(self.id_to_token[i])
        return "".join(tokens) if self.level == "char" else " ".join(tokens)

    def encode_batch(self, texts: Sequence[str], max_len: int | None = None) -> np.ndarray:
        """(N, L) int64 matris: kısa diziler `<pad>` ile sağdan doldurulur, uzunlar max_len'de kesilir."""
        encoded = [self.encode(t) for t in texts]
        length = max_len if max_len is not None else max((len(e) for e in encoded), default=0)
        out = np.full((len(encoded), length), self.pad_id, dtype=np.int64)
        for row, ids in enumerate(encoded):
            ids = ids[:length]
            out[row, :len(ids)] = ids
        return out

    def state_dict(self) -> dict:
        return {"level": self.level, "lower": self.lower, "min_freq": self.min_freq,
                "max_size": self.max_size, "vocab": list(self.id_to_token)}

    def load_state_dict(self, state: dict) -> Tokenizer:
        vocab = list(state["vocab"])
        if vocab[:2] != [self.PAD, self.UNK]:
            raise ValueError("Geçersiz tokenizer durumu: ilk iki token <pad>, <unk> olmalı")
        if len(set(vocab)) != len(vocab):
            raise ValueError("Geçersiz tokenizer durumu: kelime dağarcığında tekrar eden token var")
        self.level, self.lower = state["level"], state["lower"]
        self.min_freq, self.max_size = state["min_freq"], state["max_size"]
        self.id_to_token = vocab
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        return self


__all__ = ["Tokenizer"]
