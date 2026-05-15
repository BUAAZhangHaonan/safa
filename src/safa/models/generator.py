from __future__ import annotations


class ZOnlyGenerator:
    def __new__(cls, embedding_dim: int = 512, image_size: int = 224, base_channels: int = 256):
        import torch
        from torch import nn

        if embedding_dim != 512:
            raise ValueError(f"Generator expects 512-d z for the minimal validation, got {embedding_dim}")
        if image_size != 224:
            raise ValueError(f"Generator image_size is fixed at 224 for the minimal validation, got {image_size}")

        class _ZOnlyGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.image_size = image_size
                self.fc = nn.Sequential(
                    nn.Linear(embedding_dim, base_channels * 7 * 7),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                self.net = nn.Sequential(
                    _block(base_channels, 256),
                    _block(256, 128),
                    _block(128, 64),
                    _block(64, 32),
                    _block(32, 16),
                    nn.Conv2d(16, 3, kernel_size=3, padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, z):
                if z.ndim != 2 or z.shape[1] != self.embedding_dim:
                    raise ValueError(f"G expects z with shape [B,{self.embedding_dim}], got {tuple(z.shape)}")
                hidden = self.fc(z).view(z.shape[0], base_channels, 7, 7)
                return self.net(hidden)

        def _block(in_channels: int, out_channels: int):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

        return _ZOnlyGenerator()

