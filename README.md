# 💧 AI-Powered Water Quality Predictor

An intelligent machine learning application that predicts whether water is safe or unsafe for consumption based on its chemical properties.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [API Reference](#api-reference)
- [Results](#results)

## 🎯 Overview

This project implements an AI-powered water quality prediction system using machine learning. It analyzes 9 key water quality parameters and provides a binary classification (Safe/Unsafe) for water potability, along with confidence scores and actionable insights.

## ✨ Features

- **Real-time Predictions**: Instantly predict water quality based on chemical parameters
- **Interactive Web Interface**: User-friendly Streamlit dashboard with two main pages
- **Data Visualization**: Explore dataset distributions, correlations, and patterns
- **Model Transparency**: View confusion matrices and classification reports
- **Missing Data Handling**: Robust imputation strategies for incomplete data
- **Feature Scaling**: Normalized input values for consistent predictions
- **Explanations**: Simple interpretability of why water might be unsafe

## 📁 Project Structure

```
AI-powered-water-quality-predictor/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── generate_data.py            # Synthetic data generation
├── requirements.txt            # Python dependencies
├── water_quality.csv           # Dataset (generated or real)
├── water_quality_model.pkl     # Trained Random Forest model
├── scaler.pkl                  # StandardScaler artifact
├── imputer.pkl                 # SimpleImputer artifact
└── README.md                   # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/David-Ochieng/AI-powered-water-quality-predictor.git
cd AI-powered-water-quality-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

## 📖 Usage

### 1. Generate Synthetic Data
First, generate the synthetic water quality dataset:
```bash
python generate_data.py
```
This creates `water_quality.csv` with 2,000 samples and simulates real-world missing values.

### 2. Train the Model
Train the Random Forest classifier on the generated data:
```bash
python train_model.py
```
This will:
- Load and preprocess the data
- Handle missing values using mean imputation
- Normalize features using StandardScaler
- Train a Random Forest model with 100 estimators
- Evaluate performance on test data
- Save model artifacts (`.pkl` files)

### 3. Run the Web Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
The app will open in your default browser at `http://localhost:8501`

## 📊 Dataset

### Data Generation
- **Samples**: 2,000 water quality measurements
- **Features**: 9 chemical parameters
- **Target**: Binary classification (0=Unsafe, 1=Safe)
- **Missing Values**: ~5% per feature (simulated real-world conditions)

### Features

| Parameter | Unit | Description |
|-----------|------|-------------|
| **pH** | 0-14 | Acidity/alkalinity level |
| **Hardness** | mg/L | Mineral content |
| **Solids** | ppm | Total dissolved solids |
| **Chloramines** | ppm | Disinfectant level |
| **Sulfate** | mg/L | Sulfate ion concentration |
| **Conductivity** | μS/cm | Electrical conductivity |
| **Organic_carbon** | ppm | Organic compound concentration |
| **Trihalomethanes** | μg/L | Disinfection byproducts |
| **Turbidity** | NTU | Water clarity measure |

### Data Rules
Water is marked as **unsafe** when:
- pH < 6.5 or pH > 8.5
- Turbidity > 5 NTU
- Sulfate > 400 mg/L

## 🤖 Model Details

### Algorithm
- **Model Type**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Random State**: 42 (reproducibility)

### Preprocessing Pipeline
1. **Imputation**: Mean strategy for missing values
2. **Scaling**: StandardScaler normalization (zero mean, unit variance)
3. **Train/Test Split**: 80/20 ratio with random state 42

### Performance Metrics
The model generates:
- **Accuracy Score**: Overall classification accuracy
- **Confusion Matrix**: True positives, false positives, etc.
- **Classification Report**: Precision, recall, F1-score per class
- **Probability Scores**: Confidence level for predictions

## 💻 API Reference

### app.py - Streamlit Application

#### Pages
1. **Prediction Page**
   - Input 9 water quality parameters
   - Get Safe/Unsafe classification
   - View probability score
   - See potential issues identified

2. **Data Visualization Page**
   - Dataset overview and statistics
   - Potability distribution
   - Feature distribution analysis
   - Correlation heatmap

### train_model.py

```python
def train_model():
    """
    Trains a Random Forest classifier on water quality data.
    Saves model, scaler, and imputer as pickle files.
    """
```

### generate_data.py

```python
def generate_water_quality_data(n_samples=1000):
    """
    Generates synthetic water quality dataset.
    
    Parameters:
        n_samples (int): Number of samples to generate
    
    Returns:
        pd.DataFrame: Water quality dataset with 10 columns
    """
```

## 📈 Results

### Model Output Example
```
Accuracy: 0.8450
Precision (Safe): 0.82
Recall (Safe): 0.87
F1-Score (Safe): 0.84
```

### Prediction Example
- **Input**: pH=7.2, Hardness=200, Solids=20000, Chloramines=7.0, Sulfate=300, Conductivity=400, Organic_carbon=10, Trihalomethanes=60, Turbidity=3.5
- **Output**: ✅ SAFE - Probability: 92.3%

## 🛠️ Technologies

| Package | Version | Purpose |
|---------|---------|---------|
| **streamlit** | Latest | Web framework |
| **pandas** | Latest | Data manipulation |
| **numpy** | Latest | Numerical computing |
| **scikit-learn** | Latest | ML algorithms & preprocessing |
| **matplotlib** | Latest | Static visualizations |
| **seaborn** | Latest | Statistical graphics |
| **joblib** | Latest | Model serialization |

## 📝 Workflow

```
generate_data.py → water_quality.csv → train_model.py → Model Artifacts
                                                              ↓
                                                          app.py
                                                          (Web UI)
```

## 🔍 Key Components

### Data Pipeline
- **Input**: Raw water quality measurements
- **Processing**: Imputation → Scaling → Model Inference
- **Output**: Classification + Probability + Explanation

### User Interface
- **Interactive Inputs**: Number sliders for each parameter
- **Real-time Prediction**: Instant results with confidence scores
- **Visual Feedback**: Color-coded results (green=safe, red=unsafe)
- **Data Explorer**: Interactive charts and heatmaps

## 🚦 Quick Start Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data
python generate_data.py

# 3. Train model
python train_model.py

# 4. Run app
streamlit run app.py

# 5. Open browser to http://localhost:8501
```

## 📧 Contact

For questions or feedback about this project, please reach out to the repository owner.

## 📄 License

This project is open source and available under the MIT License.

---

**Note**: This is a demonstration project using synthetic data. For production use with real water quality data, ensure compliance with relevant environmental regulations and data privacy laws.
