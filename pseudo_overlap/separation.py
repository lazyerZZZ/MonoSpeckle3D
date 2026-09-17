from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_dividenet_v3(checkpoint: str | Path, device: str | None = None):
    import torch

    from models.self_model import DivideNet_V3

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DivideNet_V3().to(selected_device)
    state = torch.load(checkpoint, map_location=selected_device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model, selected_device


def separate_image(image: np.ndarray, model, device, tile_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Run V3 on padded tiles and crop the assembled result back to the input size."""
    import torch

    if image.ndim != 2:
        raise ValueError("DivideNet expects one grayscale image")
    height, width = image.shape
    padded_height = ((height + tile_size - 1) // tile_size) * tile_size
    padded_width = ((width + tile_size - 1) // tile_size) * tile_size
    padded = np.pad(image, ((0, padded_height - height), (0, padded_width - width)), mode="reflect")
    clear = np.empty_like(padded, dtype=np.float32)
    blurred = np.empty_like(padded, dtype=np.float32)
    with torch.no_grad():
        for top in range(0, padded_height, tile_size):
            for left in range(0, padded_width, tile_size):
                tile = padded[top : top + tile_size, left : left + tile_size]
                tensor = torch.from_numpy(tile.astype(np.float32) / 255.0)[None, None].to(device)
                predicted_clear, predicted_blurred = model(tensor)
                clear[top : top + tile_size, left : left + tile_size] = predicted_clear[0, 0].cpu().numpy()
                blurred[top : top + tile_size, left : left + tile_size] = predicted_blurred[0, 0].cpu().numpy()
    return clear[:height, :width], blurred[:height, :width]


def save_grayscale(image: np.ndarray, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="L").save(output)
