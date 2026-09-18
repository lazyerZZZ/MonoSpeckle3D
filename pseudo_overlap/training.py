from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import DivideNetV3


class SpeckleDataset(Dataset):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.blended_paths = sorted(self.directory.glob("*_blended.png"))
        if not self.blended_paths:
            raise FileNotFoundError(f"No training triplets found in {self.directory}")
        for blended_path in self.blended_paths:
            stem = blended_path.name.removesuffix("_blended.png")
            for label in ("clear", "blurred"):
                target = self.directory / f"{stem}_{label}.png"
                if not target.exists():
                    raise FileNotFoundError(target)

    def __len__(self) -> int:
        return len(self.blended_paths)

    @staticmethod
    def _tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        blended_path = self.blended_paths[index]
        stem = blended_path.name.removesuffix("_blended.png")
        clear_path = self.directory / f"{stem}_clear.png"
        blurred_path = self.directory / f"{stem}_blurred.png"
        return self._tensor(blended_path), self._tensor(clear_path), self._tensor(blurred_path)


def _loss(
    predicted_clear: torch.Tensor,
    predicted_blurred: torch.Tensor,
    blended: torch.Tensor,
    clear: torch.Tensor,
    blurred: torch.Tensor,
) -> torch.Tensor:
    mse = nn.functional.mse_loss
    l1 = nn.functional.l1_loss
    pixel_loss = mse(predicted_clear, clear) + l1(predicted_clear, clear)
    pixel_loss += mse(predicted_blurred, blurred) + l1(predicted_blurred, blurred)

    # Kept for compatibility with the thesis-trained checkpoint. This is a
    # simplified reconstruction constraint, not strict energy conservation.
    reconstruction_loss = 0.5 * mse(predicted_clear + predicted_blurred, blended)

    clear_only = (clear < 0.4) & (blurred > 0.8)
    blurred_only = (blurred < 0.4) & (clear > 0.8)
    exclusion_loss = torch.zeros((), device=blended.device)
    if clear_only.any():
        exclusion_loss += torch.mean(torch.abs(1.0 - predicted_blurred[clear_only]))
    if blurred_only.any():
        exclusion_loss += torch.mean(torch.abs(1.0 - predicted_clear[blurred_only]))
    return pixel_loss + reconstruction_loss + 5.0 * exclusion_loss


def train_dividenet(dataset_root: str | Path, checkpoint_dir: str | Path) -> Path:
    """Train the single thesis-selected model with the paper's 7:2:1 prepared data."""
    root = Path(dataset_root)
    output = Path(checkpoint_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_loader = DataLoader(SpeckleDataset(root / "train"), batch_size=128, shuffle=True, num_workers=4)
    validation_loader = DataLoader(SpeckleDataset(root / "validation"), batch_size=128, num_workers=4)
    model = DivideNetV3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    best_validation = float("inf")
    best_path = output / "best_model_v3.pth"

    for epoch in range(200):
        model.train()
        for blended, clear, blurred in tqdm(train_loader, desc=f"epoch {epoch + 1:03d} train"):
            blended, clear, blurred = blended.to(device), clear.to(device), blurred.to(device)
            predicted_clear, predicted_blurred = model(blended)
            loss = _loss(predicted_clear, predicted_blurred, blended, clear, blurred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for blended, clear, blurred in validation_loader:
                blended, clear, blurred = blended.to(device), clear.to(device), blurred.to(device)
                predicted_clear, predicted_blurred = model(blended)
                validation_total += (
                    nn.functional.mse_loss(predicted_clear, clear)
                    + nn.functional.mse_loss(predicted_blurred, blurred)
                ).item()
        validation_loss = validation_total / len(validation_loader)
        scheduler.step(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(model.state_dict(), best_path)
        print(f"epoch={epoch + 1:03d} validation_mse={validation_loss:.6f}")
    return best_path
