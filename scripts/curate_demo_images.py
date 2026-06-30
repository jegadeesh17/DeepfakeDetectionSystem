"""Copy curated demo images for presentation (Unsplash REAL, avatar FAKE)."""
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_ingestion import IMAGE_DIR, ingest_data, resolve_image_path

DEMO_DIR = PROJECT_ROOT / "demo_images"
REAL_DIR = DEMO_DIR / "real"
FAKE_DIR = DEMO_DIR / "fake"
N_EACH = 5


def _image_path(row):
    if "local_path" in row and row.get("local_path"):
        return resolve_image_path(row["local_path"])
    image_id = row.get("image_id", row.name)
    return IMAGE_DIR / f"{image_id}.jpg"


def _pick_rows(df, label: str, n: int, url_filter=None, url_exclude=None):
    subset = df[df["label"].str.upper() == label].copy()
    if "image_url" in subset.columns:
        if url_filter:
            subset = subset[subset["image_url"].str.contains(url_filter, case=False, na=False)]
        if url_exclude:
            subset = subset[~subset["image_url"].str.contains(url_exclude, case=False, na=False)]
    return subset.head(n)


def main():
    df = ingest_data(force_rebuild=False)

    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)

    real_rows = _pick_rows(df, "REAL", N_EACH, url_filter="unsplash")
    fake_rows = _pick_rows(df, "FAKE", N_EACH, url_exclude="randomuser")
    if len(fake_rows) < N_EACH:
        fake_rows = _pick_rows(df, "FAKE", N_EACH)

    for i, (_, row) in enumerate(real_rows.iterrows(), start=1):
        src = _image_path(row)
        dst = REAL_DIR / f"real_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print(f"REAL  -> {dst.name}")

    for i, (_, row) in enumerate(fake_rows.iterrows(), start=1):
        src = _image_path(row)
        dst = FAKE_DIR / f"fake_{i:02d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        print(f"FAKE  -> {dst.name}")

    print(f"Demo images ready in {DEMO_DIR}")


if __name__ == "__main__":
    main()
