from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class DivideNetV3(nn.Module):
    """Dual-encoder DivideNet V3 selected by the thesis."""

    def __init__(self) -> None:
        super().__init__()
        self.enc_c = nn.ModuleList([ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128)])
        self.enc_b = nn.ModuleList([ConvBlock(1, 32), ConvBlock(32, 64), ConvBlock(64, 128)])
        self.pool = nn.MaxPool2d(2)

        self.up1_c = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_c = ConvBlock(128, 64)
        self.up2_c = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2_c = ConvBlock(64, 32)
        self.out_c = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

        self.up1_b = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_b = ConvBlock(128, 64)
        self.up2_b = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2_b = ConvBlock(64, 32)
        self.out_b = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _encode(self, inputs: torch.Tensor, encoder: nn.ModuleList) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        level1 = encoder[0](inputs)
        level2 = encoder[1](self.pool(level1))
        level3 = encoder[2](self.pool(level2))
        return level1, level2, level3

    def _decode_clear(self, levels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        level1, level2, level3 = levels
        decoded = self.dec1_c(torch.cat((self.up1_c(level3), level2), dim=1))
        decoded = self.dec2_c(torch.cat((self.up2_c(decoded), level1), dim=1))
        return self.out_c(decoded)

    def _decode_blurred(self, levels: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        level1, level2, level3 = levels
        decoded = self.dec1_b(torch.cat((self.up1_b(level3), level2), dim=1))
        decoded = self.dec2_b(torch.cat((self.up2_b(decoded), level1), dim=1))
        return self.out_b(decoded)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clear = self._decode_clear(self._encode(inputs, self.enc_c))
        blurred = self._decode_blurred(self._encode(inputs, self.enc_b))
        return clear, blurred
