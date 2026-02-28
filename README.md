# AI-CROP-DISEASE-PREDICTION
A deep learning project that classifies crop diseases from leaf images using CNN and Python.
# AI Crop Disease Prediction 🌱

## 📌 Project Overview
AI Crop Disease Prediction is a deep learning–based application that detects and classifies crop diseases from leaf images.  
The system helps farmers and agricultural experts identify diseases at an early stage using image classification techniques.

---

## 🎯 Objectives
- To identify crop diseases from leaf images
- To reduce manual inspection and human error
- To assist farmers with early disease detection
- To apply deep learning concepts in agriculture

---

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- NumPy
- OpenCV
- Streamlit (for UI)

---

## 📂 Project Structure

AI-CROP-DISEASE-PREDICTION/
│
├── app.py # Streamlit application
├── train.py # Model training script
├── predict.py # Prediction logic
├── class_indices.json # Class index mapping
├── class_labels.json # Disease labels
├── requirements.txt # Required Python packages
├── README.md # Project documentation
└── .gitignore # Ignored files


---

## ⚠️ Note About Model File
The trained model file (`.h5`) is **not included** in this repository due to GitHub file size limitations.

You can:
- Train the model using `train.py`, OR
- Download the trained model from an external source (Google Drive / Kaggle)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/s4njana28/AI-CROP-DISEASE-PREDICTION.git
cd AI-CROP-DISEASE-PREDICTION
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
streamlit run app.py
