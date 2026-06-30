import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import psycopg2
import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import uuid

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "dataset.csv"
IMAGE_DIR = DATA_DIR / "images"
RAW_IMAGE_DIR = DATA_DIR / "raw_images"


def resolve_image_path(local_path: str) -> Path:
    """Resolve image paths stored as relative or absolute paths."""
    path = Path(local_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "data" / "images" / path.name,
        Path(local_path.replace("\\", "/").lstrip("./")),
        PROJECT_ROOT / local_path.replace("\\", "/").lstrip("./"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Image not found: {local_path}")


def get_db_connection():
    """Establish a PostgreSQL connection using environment variables."""
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        return None

    try:
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None


def setup_database():
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS images_metadata (
                id SERIAL PRIMARY KEY,
                image_id VARCHAR(50) UNIQUE,
                label VARCHAR(20),
                gender VARCHAR(20),
                age_group VARCHAR(20),
                local_path VARCHAR(255)
            )
        """)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error setting up the database table: {e}")
        return False
    finally:
        conn.close()


def drop_database():
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS images_metadata;")
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error dropping the database table: {e}")
        return False
    finally:
        conn.close()


def try_resolve_image_path(local_path: str) -> str:
    try:
        return str(resolve_image_path(local_path))
    except FileNotFoundError:
        return local_path


def _normalize_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["local_path"] = df["local_path"].apply(
        lambda p: try_resolve_image_path(p) if pd.notna(p) else p
    )
    return df


def check_data_exists():
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        df = pd.read_sql(
            "SELECT id, image_id, label, gender, age_group, local_path FROM images_metadata",
            conn,
        )
        if not df.empty:
            return _normalize_paths(df)
    except Exception as e:
        print(f"Database read failed: {e}")
    finally:
        conn.close()

    return None


def _download_images_from_csv(raw_df: pd.DataFrame) -> pd.DataFrame:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    def download_and_process_image(row):
        image_id = str(row["image_id"])
        image_url = row["image_url"]
        local_path = IMAGE_DIR / f"{image_id}.jpg"

        if not local_path.exists():
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img.verify()
                local_path.write_bytes(response.content)
            except Exception:
                return None

        return {
            "image_id": image_id,
            "label": row["label"],
            "gender": row.get("gender"),
            "age_group": row.get("age_group"),
            "local_path": str(local_path.resolve()),
        }

    records = []
    print(f"Downloading {len(raw_df)} images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_and_process_image, row) for _, row in raw_df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                records.append(result)

    return pd.DataFrame(records)


def _insert_records_to_db(records_df: pd.DataFrame) -> bool:
    conn = get_db_connection()
    if conn is None:
        return False

    try:
        cur = conn.cursor()
        insert_query = """
            INSERT INTO images_metadata (image_id, label, gender, age_group, local_path)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (image_id) DO NOTHING;
        """
        rows = [
            (r.image_id, r.label, r.gender, r.age_group, r.local_path)
            for r in records_df.itertuples(index=False)
        ]
        cur.executemany(insert_query, rows)
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Database insert failed: {e}")
        return False
    finally:
        conn.close()


def ingest_from_csv():
    if not CSV_FILE.exists():
        print(f"CSV file not found at {CSV_FILE}.")
        return None

    raw_df = pd.read_csv(CSV_FILE)
    df = _download_images_from_csv(raw_df)
    if df.empty:
        print("No valid images were downloaded from CSV.")
        return None

    if setup_database():
        _insert_records_to_db(df)

    print(f"Ingestion complete. Loaded {len(df)} records from CSV.")
    return df


def ingest_from_local_folders():
    if not RAW_IMAGE_DIR.exists():
        print(f"Raw image directory not found at {RAW_IMAGE_DIR}.")
        return None

    records = []
    print(f"Scanning local folders in {RAW_IMAGE_DIR}...")
    
    for label in ["REAL", "FAKE"]:
        label_dir = RAW_IMAGE_DIR / label
        if not label_dir.exists():
            print(f"Directory {label_dir} not found. Skipping...")
            continue
            
        for img_file in label_dir.glob("*.*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                image_id = str(uuid.uuid4())[:8] + "_" + img_file.stem
                records.append({
                    "image_id": image_id,
                    "label": label,
                    "gender": None,
                    "age_group": None,
                    "local_path": str(img_file.resolve())
                })
                
    if not records:
        print("No valid images were found in the raw image folders.")
        return None
        
    df = pd.DataFrame(records)
    
    if setup_database():
        _insert_records_to_db(df)
        
    print(f"Ingestion complete. Loaded {len(df)} records from local folders.")
    return df


def ingest_data(force_rebuild=False):
    if force_rebuild:
        print("Forcing database rebuild. Dropping existing metadata table...")
        drop_database()

    print("Checking if data is already ingested...")
    df = check_data_exists()

    if df is not None and not df.empty:
        print(f"Data already exists! Found {len(df)} records in the database.")
        return df

    if RAW_IMAGE_DIR.exists() and any(RAW_IMAGE_DIR.iterdir()):
        print("Database unavailable or empty. Falling back to local folder ingestion...")
        return ingest_from_local_folders()

    print("Database unavailable or empty and no raw images found. Falling back to CSV ingestion...")
    return ingest_from_csv()


if __name__ == "__main__":
    ingest_data()
