import os
import shutil
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "dataset.csv"
IMAGE_DIR = DATA_DIR / "images"
RAW_IMAGE_DIR = DATA_DIR / "raw_images"
REAL_DIR = RAW_IMAGE_DIR / "REAL"
FAKE_DIR = RAW_IMAGE_DIR / "FAKE"

# Target size should match the training transform size (224x224)
TARGET_SIZE = (224, 224)
# Margin around detected face for extra context (useful for deepfake boundary artifacts)
FACE_MARGIN = 0.2


def setup_dirs():
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    FAKE_DIR.mkdir(parents=True, exist_ok=True)


def process_image(row):
    image_id = str(row["image_id"])
    label = str(row["label"]).upper()
    src_path = IMAGE_DIR / f"{image_id}.jpg"

    if not src_path.exists():
        return False

    if label not in ("REAL", "FAKE"):
        return False

    # Use the same destination logic for both classes
    dst_dir = REAL_DIR if label == "REAL" else FAKE_DIR
    dst_path = dst_dir / f"{image_id}.jpg"
    if dst_path.exists():
        return True

    img = cv2.imread(str(src_path))
    if img is None:
        return False

    # Apply identical face detection + cropping to BOTH REAL and FAKE images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        # Add margin around face for context (helps detect boundary artifacts)
        mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img.shape[1], x + w + mx)
        y2 = min(img.shape[0], y + h + my)
        face_img = img[y1:y2, x1:x2]
    else:
        # Fallback: use full image if no face detected
        face_img = img

    try:
        face_img_resized = cv2.resize(face_img, TARGET_SIZE)
        cv2.imwrite(str(dst_path), face_img_resized)
        return True
    except Exception:
        return False


def main():
    if not CSV_FILE.exists():
        print(f"CSV file not found at {CSV_FILE}")
        return

    print("Loading dataset metadata...")
    df = pd.read_csv(CSV_FILE)

    setup_dirs()

    print(f"Processing {len(df)} images...")
    successful = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_image, row) for _, row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            if future.result():
                successful += 1

    print(f"\nProcessing complete! Successfully aligned {successful} out of {len(df)} images.")
    print(f"REAL images: {len(list(REAL_DIR.glob('*.jpg')))}")
    print(f"FAKE images: {len(list(FAKE_DIR.glob('*.jpg')))}")

if __name__ == "__main__":
    main()
