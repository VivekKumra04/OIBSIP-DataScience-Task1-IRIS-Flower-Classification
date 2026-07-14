# 🌸 Iris Classifier AI

Iris Classifier AI is a premium, full-stack machine learning application designed to classify iris flowers into their respective species (*Setosa*, *Versicolor*, and *Virginica*) in real-time. Built with a modern, glassmorphic dark-themed user interface, it combines a robust Python-based ML training pipeline with a Flask web API and an interactive web demo.

---

## ✨ Features

- **📊 Comparative ML Pipeline**: Automatically trains, evaluates, and compares **Logistic Regression**, **Support Vector Classifier (RBF Kernel)**, and **Random Forest Classifier** models.
- **✨ Real-Time Classification**: Interact with sliders or sample presets in a stunning dark-mode web interface to classify flower types on the fly.
- **🌺 SVG Flower Visualization**: Interactive flower SVG graphics that dynamically reshape and animate based on the predicted flower class and confidence level.
- **📈 Dynamic Model Analytics**: Full analytics dashboard embedded in the UI showing PCA clusters (2D projections), Confusion Matrix heatmaps, and Feature Correlation matrices.
- **💻 CLI Predictor**: Standalone CLI tool to easily perform predictions from your terminal.
- **⚡ Flask REST API**: Standardized JSON endpoints to fetch model metadata, make POST predictions, and retrieve training evaluation plots.

---

## 📂 Project Structure

```directory
iris_flower_classification/
├── app.py                     # Main Flask web application & REST API
├── requirements.txt           # Python library dependencies
├── dataset/
│   └── Iris.csv               # Raw Iris flower dataset
├── models/
│   ├── iris_model.pkl         # Trained winning model file
│   ├── scaler.pkl             # Fitted StandardScaler object
│   ├── metadata.json          # Exported training metrics and class mapping
│   └── evaluation_plots/      # Matplotlib/Seaborn-generated charts
│       ├── confusion_matrix.png
│       ├── feature_correlation.png
│       └── pca_projection.png
├── src/
│   ├── train.py               # Model training and comparison pipeline
│   └── predict.py             # CLI application for single predictions
├── static/
│   ├── app.js                 # Frontend interactive logic & SVG morphing
│   └── style.css              # Custom CSS rules with glassmorphic layout
└── templates/
    └── index.html             # Main dashboard template
```

---

## 🛠️ Tech Stack

- **ML & Data Processing**: Python, NumPy, Pandas, Scikit-learn, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Backend / Web Server**: Flask (Python)
- **Frontend / UI**: HTML5, Vanilla CSS3 (Custom Glassmorphism, HSL tailormade colors), Vanilla JavaScript

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Installation
Clone this repository, navigate to the root directory, and install the required dependencies:

```bash
# Create a virtual environment
python -m venv venv
# Activate virtual environment (Windows Powershell)
.\venv\Scripts\Activate.ps1
# Active virtual environment (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Training the Model Pipeline
Run the training script to evaluate the models, choose the best performer, output metrics, and save evaluation plots:

```bash
python src/train.py
```
*Note: This script will run 5-fold cross-validation on Logistic Regression, SVC, and Random Forest, export the winner to the `models/` directory, and generate the visualization plots in `models/evaluation_plots/`.*

### 4. Running the Web Application
To start the Flask development server:

```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the visual dashboard.

### 5. Running CLI Predictions
You can run predictions directly from the command line using `src/predict.py`:

```bash
python src/predict.py --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2
```

---

## 🔌 API Documentation

### 1. Get Model Metadata
Returns information about the currently loaded model, training accuracy, features, and classes.
- **Endpoint**: `/api/metadata`
- **Method**: `GET`
- **Response**:
```json
{
    "model_name": "Support Vector Classifier (RBF Kernel)",
    "features": ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    "classes": ["Setosa", "Versicolor", "Virginica"],
    "test_accuracy": 0.9666666666666667,
    "cv_accuracy": 0.9666666666666668
}
```

### 2. Make Classifications
Predicts the Iris species based on flower measurements.
- **Endpoint**: `/api/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```
- **Response**:
```json
{
    "status": "success",
    "prediction": "Setosa",
    "confidence": 0.9845,
    "probabilities": {
        "Setosa": 0.9845,
        "Versicolor": 0.0125,
        "Virginica": 0.003
    }
}
```

### 3. Retrieve Evaluation Plots
Serves the generated training plots to the frontend dashboard.
- **Endpoint**: `/api/plots/<filename>`
- **Method**: `GET`
- **Available Filenames**: 
  - `confusion_matrix.png`
  - `pca_projection.png`
  - `feature_correlation.png`
  - `feature_importance.png` (if Random Forest is the selected model)

---

## 📈 Model Performance Summary

- **Winning Model**: Support Vector Classifier (RBF Kernel)
- **5-Fold Cross Validation Accuracy**: ~96.67%
- **Holdout Test Set Accuracy**: ~96.67%
- **Key Insignts**:
  - **Setosa** is completely linearly separable from the other classes due to its wide sepals and small petals.
  - **Petal Length** and **Petal Width** are highly correlated (+0.96) and represent the most critical features for distinguishing between **Versicolor** and **Virginica**.
