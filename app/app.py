import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import generate_gradcam_heatmap, preprocess_for_inference_v2
from src.model_loader import find_checkpoint_path, load_model

st.set_page_config(page_title="Deepfake Detection", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}

    .title-text {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        text-align: center;
    }

    .subtitle-text {
        font-size: 1.1rem;
        font-weight: 300;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 40px;
    }

    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-top: 20px;
    }

    .fake-text {
        color: #ef4444;
        text-shadow: 0px 0px 10px rgba(239,68,68,0.5);
    }

    .real-text {
        color: #10b981;
        text-shadow: 0px 0px 10px rgba(16,185,129,0.5);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    .metric-label {
        font-size: 1rem;
        font-weight: 600;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None, None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None


st.markdown('<div class="title-text">Deepfake Shield</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Advanced Deep Learning Image Forgery Detection</div>',
    unsafe_allow_html=True,
)

model, architecture, device = load_trained_model()

if model is None:
    checkpoint_hint = find_checkpoint_path()
    if checkpoint_hint is None:
        st.error(
            "Trained model not found. Place `final_deepfake_detector.pth` or `best_model.pth` "
            "in the `models/` folder."
        )
    else:
        st.error("Model checkpoint found but could not be loaded. Check architecture metadata or `.env`.")
else:
    st.caption(f"Loaded {architecture} on {device}")

    st.markdown("### Upload Suspect Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("<br><hr style='border: 1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Source Image")
                st.image(image)
                st.markdown("#### Model Input (224×224)")

            with st.status("Running Neural Network Inference...", expanded=True) as status:
                st.write("Analyzing facial features and artifacts...")
                image_tensor, model_input = preprocess_for_inference_v2(image, device)

            with col1:
                st.image(model_input)
                with torch.no_grad():
                    output = model(image_tensor).squeeze(1)
                    prob = torch.sigmoid(output).item()
                status.update(label="Inference Complete!", state="complete", expanded=False)

            prediction = "FAKE" if prob > 0.5 else "REAL"
            confidence = prob if prob > 0.5 else (1 - prob)
            pred_class = "fake-text" if prediction == "FAKE" else "real-text"

            with col2:
                st.markdown("#### Analysis Results")
                st.markdown(f"""
                <div class="result-card">
                    <p class="metric-label">Classification</p>
                    <p class="metric-value {pred_class}">{prediction}</p>
                    <br>
                    <p class="metric-label">Confidence</p>
                    <p class="metric-value" style="color: #38bdf8;">{confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Explainable AI Analysis")
            st.markdown(
                "<p style='color: #94a3b8;'>The heatmap highlights the specific regions "
                "the neural network focused on to make its decision.</p>",
                unsafe_allow_html=True,
            )

            with st.status("Generating Grad-CAM Heatmap...", expanded=True) as status2:
                st.write("Computing gradient activations...")
                heatmap_image = generate_gradcam_heatmap(model, image_tensor, model_input, architecture)
                status2.update(label="Heatmap Generated!", state="complete", expanded=False)

            st.image(heatmap_image)

        except Exception as e:
            import traceback

            st.error(f"Error processing image: {e}")
            st.code(traceback.format_exc())
