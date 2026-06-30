# Deepfake Detection System
---
### **Project Overview**
Deepfake Detection System is a binary image classifier that distinguishes REAL faces from AI-generated FAKE faces. Training runs in Google Colab (GPU); inference and explainability run locally via a Streamlit dashboard with Grad-CAM heatmaps.

---
### **Key Features**
- Transfer learning with EfficientNet-B0 and ResNet-50
- Automatic best-model selection by ROC-AUC
- Grad-CAM explainability in the Streamlit UI
- Architecture auto-detection for legacy Colab checkpoints (`best_model.pth`)
- Optional PostgreSQL metadata store with CSV fallback

---
### **Dataset**
- Metadata CSV: `data/dataset.csv` (columns: `image_id`, `image_url`, `label`, `gender`, `age_group`)
- Images downloaded to `data/images/`
- Labels: `REAL` (0) and `FAKE` (1)

---
### **Project Structure**
```text
DeepfakeDetectionSystem/
├── app/app.py              # Streamlit dashboard
├── data/                   # Dataset CSV and images (gitignored)
├── models/                 # Trained checkpoints (gitignored)
├── notebooks/              # 10-step training notebook
├── src/                    # Core Python modules
├── requirements.txt
├── .env.example
└── README.md
```

---
### **How It Works**
1. **Data ingestion** loads metadata from PostgreSQL or downloads images from CSV URLs.
2. **Preprocessing** applies ImageNet normalization and augmentation.
3. **Training** compares EfficientNet and ResNet, fine-tunes the winner, and saves a checkpoint with architecture metadata.
4. **Inference** loads the checkpoint, predicts FAKE/REAL probability, and generates a Grad-CAM heatmap.

---
### **Model Performance**
Performance depends on your Colab training run. The notebook reports validation accuracy, confusion matrix, classification report, and ROC-AUC for both architectures before selecting the best model.

---
### **Interactive Application Deployment**
```bash
cd DeepfakeDetectionSystem
pip install -r requirements.txt
streamlit run app/app.py
```
Place your trained checkpoint in `models/` as either:
- `final_deepfake_detector.pth` (recommended, includes architecture metadata), or
- `best_model.pth` (legacy Colab export; architecture is auto-detected)

---
### **Technology Stack**
- PyTorch, torchvision
- scikit-learn, pandas
- Streamlit
- pytorch-grad-cam (`grad-cam` package), OpenCV
- PostgreSQL (optional)

---
### **Getting Started**
### **1. Clone Repository**
```bash
git clone <your-repo-url>
cd DeepfakeDetectionSystem
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
cp .env.example .env   # edit DB credentials if using PostgreSQL
```

### **3. Launch Notebook**
Open `notebooks/Deepfake Detection.ipynb` in Google Colab. After training, download:
- `models/final_deepfake_detector.pth`
- `models/model_metadata.json`

Copy them into your local `models/` folder.

### **4. Launch Dashboard**
```bash
streamlit run app/app.py
```

If architecture auto-detection fails for a legacy checkpoint, set `MODEL_ARCH=EfficientNet` or `MODEL_ARCH=ResNet` in `.env`.

---
### **Example Use Case**
Upload a suspect profile photo to the Streamlit app. The model returns a FAKE/REAL classification with confidence score and a Grad-CAM heatmap showing which facial regions influenced the decision.

---
### **Future Improvements**
- Add video/deepfake frame analysis
- Deploy model via FastAPI for API access
- Add test-time augmentation for more robust inference

---
### **Contributors**
Your Name

---
### **License**
MIT
