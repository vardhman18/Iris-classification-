# 🌸 Iris Species Classifier

A simple machine learning web app to classify Iris flower species using **Random Forest** and **SVM** models. Built with **Streamlit** for a clean, interactive UI.

---

## 🔍 Features

- Input sepal and petal measurements via sliders
- Get predictions from both Random Forest and SVM
- Clean and responsive interface
- No database needed — models are pre-trained and saved as `.pkl` files

---

## 🗂️ Project Structure

iris_classifier_project/
│
├── iris_random_forest.pkl # Trained Random Forest model
├── iris_svm.pkl # Trained SVM model
├── label_encoder.pkl # Label encoder used during training
├── iris_classification
├── iris.csv
├── app.py # Streamlit web app
├── requirements.txt # Python dependencies
└── README.md # Project documentation



---

## 🚀 Run Locally

### 1. Clone the repo or download the folder

git clone https://github.com/yourusername/iris-classifier.git
cd iris-classifier

### 2. Install dependencies

pip install -r requirements.txt

### 3. Launch the app

streamlit run app.py

### 🧠 Model Info
Dataset: Iris Dataset

Models: Trained using RandomForestClassifier and SVC from scikit-learn

Training and saving done in Jupyter Notebook

### 📦 Deploy on Streamlit Cloud
Push this folder to a GitHub repo

Go to https://streamlit.io/cloud

Connect your repo and select app.py as the entry point

Add requirements.txt if not auto-detected

### 🛠️ Tech Stack
Frontend: Streamlit

Backend: Pre-trained ML models (.pkl files)

Language: Python 3.9+

### 🧾 License
This project is for educational purposes as part of a Data Science Internship.

