# Credit Card Fraud Detection Project

## Overview
This is my machine learning project for detecting fraudulent credit card transactions. I used the famous Kaggle credit card dataset and implemented different ML algorithms to identify fraud patterns.

## Dataset
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Link**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size**: 284,807 transactions
- **Features**: 30 features (V1-V28 are PCA transformed, Time, Amount)
- **Target**: Class (0=Normal, 1=Fraud)
- **Imbalance**: Only 0.172% transactions are fraudulent

## Project Structure
```
fraud-detection-project/
│
├── data/                    # Dataset folder (download from Kaggle)
│   └── creditcard.csv
│
├── models/                  # Saved trained models
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── figures/                 # Generated plots
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
├── fraud_detection.ipynb    # Main Jupyter notebook
├── fraud_detection.py       # Python script version
├── requirements.txt         # Required packages
└── README.md               # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place `creditcard.csv` in the `data/` folder

## How to Run

### Using Jupyter Notebook:
1. Open terminal and navigate to project folder
2. Run: `jupyter notebook`
3. Open `fraud_detection.ipynb`
4. Run all cells

### Using Python Script:
```bash
python fraud_detection.py
```

## Methodology

### 1. Data Preprocessing
- Scaled Amount and Time features using StandardScaler
- Split data into 70% training and 30% testing

### 2. Handling Imbalanced Data
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the training set
- This helps the model learn fraud patterns better

### 3. Models Implemented
1. **Logistic Regression**: Simple baseline model
2. **Decision Tree**: Tree-based model with max_depth=10
3. **Random Forest**: Ensemble model with 100 trees

### 4. Evaluation Metrics
Since the dataset is imbalanced, I focused on:
- **Precision**: How many predicted frauds are actual frauds
- **Recall**: How many actual frauds were caught
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall performance measure

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.974 | 0.062 | 0.923 | 0.116 |
| Decision Tree | 0.999 | 0.739 | 0.782 | 0.760 |
| **Random Forest** | **0.999** | **0.933** | **0.810** | **0.867** |

**Best Model**: Random Forest with F1-Score of 0.867

## Key Findings
- Random Forest performed best overall
- Most important features for fraud detection are V14, V4, V11, V12
- Model achieves high recall (81%) meaning it catches most frauds
- Very high precision (93.3%) means few false alarms

## Technologies Used
- Python 3.8
- Pandas & NumPy for data manipulation
- Scikit-learn for ML models
- Imbalanced-learn for SMOTE
- Matplotlib & Seaborn for visualization

## Future Improvements
- Try XGBoost or LightGBM models
- Implement neural networks
- Use GridSearchCV for hyperparameter tuning
- Try different resampling techniques
- Add real-time prediction capability

## Author
**Name**: John Doe  
**Email**: john.doe@student.edu  
**Date**: December 2024

## License
This project is for educational purposes

## Acknowledgments
- Thanks to Kaggle for providing the dataset
- Thanks to my ML course instructor for guidance