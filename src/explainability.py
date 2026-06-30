import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data_preprocessing import get_inference_transform
from src.model_builder import get_gradcam_target_layer

# Margin around detected face — must match align_dataset.py
FACE_MARGIN = 0.2


def detect_and_crop_face(image: Image.Image) -> Image.Image:
    """Detect face in PIL image and return cropped region, or original if no face found.

    Uses the same Haar cascade parameters and margin as align_dataset.py
    so that inference preprocessing matches the training data distribution.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_array.shape[1], x + w + mx)
        y2 = min(img_array.shape[0], y + h + my)
        cropped = img_array[y1:y2, x1:x2]
        return Image.fromarray(cropped)

    return image  # fallback: no face found


def preprocess_for_inference_v2(image: Image.Image, device: torch.device):
    """Preprocess image for inference — matches training (full image, Resize 224)."""
    rgb_image = image.convert("RGB")
    transform = get_inference_transform()
    tensor = transform(rgb_image).unsqueeze(0)
    display_image = rgb_image.resize((224, 224), Image.Resampling.BILINEAR)
    return tensor.to(device), display_image


def generate_gradcam_heatmap(model, image_tensor, original_image, architecture: str):
    """Generate a Grad-CAM heatmap overlaid on the original image."""
    target_layers = get_gradcam_target_layer(model, architecture)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)[0]

    if not isinstance(original_image, np.ndarray):
        original_image = np.array(original_image.convert("RGB"))

    if original_image.shape[:2] != (224, 224):
        original_image = cv2.resize(original_image, (224, 224))

    rgb_float = original_image.astype(np.float32) / 255.0
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
