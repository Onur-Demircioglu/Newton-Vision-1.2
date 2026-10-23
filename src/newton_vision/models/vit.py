"""newton_vision.models.vit - Vision Transformer (ViT) Mimari Uygulaması."""

from __future__ import annotations
import numpy as np
import newton_vision as nv
from newton_vision.tensor import Tensor
from newton_vision.module import Module, Parameter
from newton_vision.layers import Conv2d, Linear, LayerNorm, GELU, MultiHeadAttention


class PatchEmbedding(Module):
    """2D Resmi (N, C, H, W) Jeton Dizilerine (N, Num_Patches, Embed_Dim) Dönüştüren Katman."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768, device: str = "cpu"):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        self.device = device.lower()

        self.proj = Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, device=self.device)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        feat = self.proj(x_tensor)  # (N, Embed_Dim, Grid_H, Grid_W)
        shape = feat.data.shape
        N, D, GH, GW = shape[0], shape[1], shape[2], shape[3]
        
        # Flatten to (N, Num_Patches, Embed_Dim)
        return feat.reshape(N, D, GH * GW).transpose(0, 2, 1)


class TransformerBlock(Module):
    """Tekli Transformer Encoder Bloğu (LayerNorm -> MHSA -> LayerNorm -> MLP)."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, device: str = "cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.device = device.lower()

        self.norm1 = LayerNorm(embed_dim, device=self.device)
        self.attn = MultiHeadAttention(embed_dim, num_heads, device=self.device)
        self.norm2 = LayerNorm(embed_dim, device=self.device)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = Linear(embed_dim, mlp_hidden_dim, device=self.device)
        self.act = GELU()
        self.fc2 = Linear(mlp_hidden_dim, embed_dim, device=self.device)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        
        # 1. Multi-Head Self-Attention Block with Residual Skip Connection
        h1 = self.attn(self.norm1(x_tensor))
        x1 = x_tensor + h1

        # 2. MLP Block with Residual Skip Connection
        h2 = self.fc2(self.act(self.fc1(self.norm2(x1))))
        return x1 + h2


class VisionTransformer(Module):
    """Vision Transformer (ViT) Sınıflandırıcı Modeli."""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.device = device.lower()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, device=self.device)
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(np.zeros((1, 1, embed_dim), dtype=np.float32), device=self.device)
        self.pos_embed = Parameter(np.random.randn(1, num_patches + 1, embed_dim).astype(np.float32) * 0.02, device=self.device)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_ratio, device=self.device)
            for _ in range(depth)
        ]

        self.norm = LayerNorm(embed_dim, device=self.device)
        self.head = Linear(embed_dim, num_classes, device=self.device)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        
        # 1. Patch Embedding (N, Num_Patches, Embed_Dim)
        x_patches = self.patch_embed(x_tensor)
        N = x_patches.data.shape[0]

        # 2. Prepend CLS token and add positional embedding
        cls_tokens = self.cls_token.tile((N, 1, 1))
        x_seq = Tensor.concatenate([cls_tokens, x_patches], axis=1) + self.pos_embed

        # 3. Pass through Transformer Blocks
        for blk in self.blocks:
            x_seq = blk(x_seq)

        # 4. LayerNorm over sequence
        x_norm = self.norm(x_seq)

        # 5. Extract CLS token output (N, Embed_Dim)
        cls_tensor = x_norm[:, 0, :]

        # 6. Classification Head (N, Num_Classes)
        return self.head(cls_tensor)


def vit_tiny(num_classes: int = 1000, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, device: str = "cpu") -> VisionTransformer:
    """ViT-Tiny Yapılandırması (Embed Dim: 192, Depth: 12, Heads: 3)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        device=device
    )


def vit_base(num_classes: int = 1000, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, device: str = "cpu") -> VisionTransformer:
    """ViT-Base Yapılandırması (Embed Dim: 768, Depth: 12, Heads: 12)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        device=device
    )
