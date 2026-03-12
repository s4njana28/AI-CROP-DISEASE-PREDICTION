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

## Dataset
This project uses the PlantVillage Dataset.

Download from Kaggle:
(https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

After downloading:
1. Extract the zip file
2. Place all class folders inside dataset/train/
---
## Dataset
This project uses a subset of the PlantVillage Dataset.

Download from Kaggle:
https://www.kaggle.com/datasets/emmarex/plantdisease

## Crops and Diseases Covered (20 Classes)

🍎 Apple (4 classes)
   - Apple Scab
   - Black Rot
   - Cedar Apple Rust
   - Healthy

🌽 Corn / Maize (4 classes)
   - Cercospora Leaf Spot
   - Common Rust
   - Northern Leaf Blight
   - Healthy

🍇 Grape (4 classes)
   - Black Rot
   - Esca (Black Measles)
   - Leaf Blight
   - Healthy

🥔 Potato (2 classes)
   - Late Blight
   - Healthy

🍓 Strawberry (2 classes)
   - Leaf Scorch
   - Healthy

🍅 Tomato (4 classes)
   - Bacterial Spot
   - Late Blight
   - Septoria Leaf Spot
   - Healthy

Total: 6 Crops | 20 Disease Classes

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/s4njana28/AI-CROP-DISEASE-PREDICTION.git
cd AI-CROP-DISEASE-PREDICTION
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the application
streamlit run app.py
📊 Dataset

The dataset consists of labeled leaf images of different crop diseases.
Due to size limitations, the dataset is not uploaded here.

Dataset source:

Kaggle (Plant Disease Dataset)

🎓 Academic Use

This project was developed as part of the BCA (Generative AI) curriculum and is intended for academic and learning purposes.

👩‍💻 Author

Sanjana N
BCA – Generative AI
SRM University

⭐ Future Enhancements

Support for more crop varieties

Mobile application integration

Real-time disease detection

Severity prediction
