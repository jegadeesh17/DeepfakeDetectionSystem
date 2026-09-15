"""
Unit and Integration Tests for Deepfake Detection API.
"""

import io
import os
import sys
from PIL import Image
import pytest
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "architecture" in data


def test_predict_valid_image(client):
    img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["REAL", "FAKE"]
    assert 0.0 <= data["fake_probability"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert "inference_time_ms" in data


def test_predict_invalid_file_type(client):
    buf = io.BytesIO(b"not an image text file")
    response = client.post(
        "/predict",
        files={"file": ("test.txt", buf, "text/plain")}
    )
    assert response.status_code == 400
