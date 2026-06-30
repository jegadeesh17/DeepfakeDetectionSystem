"""Quick smoke test: inference preprocessing + existing checkpoint."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image

from src.data_ingestion import resolve_image_path
from src.data_preprocessing import get_dataloaders
from src.explainability import preprocess_for_inference_v2
from src.model_loader import load_model


def main():
    model, arch, device = load_model()
    print(f"Loaded {arch} on {device}")

    _, _, test_loader = get_dataloaders(batch_size=8)
    samples = test_loader.dataset.dataframe.sample(6, random_state=42)

    ok = 0
    for _, row in samples.iterrows():
        img = Image.open(resolve_image_path(row["local_path"])).convert("RGB")
        tensor, _ = preprocess_for_inference_v2(img, device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor).squeeze(1)).item()
        pred = "FAKE" if prob > 0.5 else "REAL"
        actual = str(row["label"]).upper()
        match = pred == actual
        ok += int(match)
        status = "OK" if match else "MISMATCH"
        print(f"{status} actual={actual} pred={pred} prob={prob:.3f}")

    print(f"Smoke test: {ok}/6 correct")


if __name__ == "__main__":
    main()
