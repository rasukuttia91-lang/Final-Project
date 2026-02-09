# Final-Project :# Advanced Time Series Forecasting using LSTM with Attention

This project implements an end-to-end deep learning framework for multivariate time series forecasting using an LSTM network enhanced with a self-attention mechanism. The objective is to model complex temporal dependencies and compare the performance of deep learning with a classical statistical baseline.
## Project Objectives

- Generate a synthetic multivariate time series dataset.
- Capture trend, seasonality, noise, and feature correlations.
- Build an LSTM-based forecasting model with self-attention.
- Apply rolling time series cross-validation.
- Compare results with a SARIMA baseline model.
- Evaluate using RMSE, MAE, and MAPE metrics.
- Visualize and interpret attention weights.
## Dataset

A synthetic dataset with **1500 time steps** is generated programmatically using NumPy.  
It consists of:

- **5 input features**: f1, f2, f3, f4, f5  
- **1 target variable**: y  

The dataset includes:
- Linear trend component  
- Sinusoidal seasonal patterns  
- Gaussian noise  
- Correlated features  

The target variable is constructed as a weighted combination of multiple features and seasonal components to ensure meaningful temporal dependencies.
## Data Preprocessing

- The dataset is standardized using `StandardScaler`.
- A sliding window approach with **sequence length = 30** is used.
- Each input sample consists of 30 historical time steps and 5 features.
- The dataset is split into **80% training** and **20% testing** while preserving temporal order.

---
## Model Architecture

The deep learning model consists of:

1. Input Layer: Receives sequences of shape (30, 5)
2. LSTM Layer: 64 hidden units with return sequences enabled
3. Self-Attention Layer: Learns temporal importance weights
4. Context Vector: Temporal aggregation of attention outputs
5. Dense Output Layer: Produces final prediction

The model is trained using:
- Optimizer: Adam  
- Loss function: Mean Squared Error (MSE)

---
## Training Strategy

- Rolling time series cross-validation is performed using `TimeSeriesSplit`.
- Each fold trains on past data and validates on future data.
- Final training is performed on the full training dataset.

This strategy ensures realistic forecasting and avoids information leakage.

---

## Baseline Model

A classical **SARIMA (2,1,2)** model is used as a baseline.  
This allows comparison between:
- Statistical forecasting approach  
- Deep learning-based approach  

---
## Evaluation Metrics

Both models are evaluated using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

These metrics quantify prediction accuracy and robustness.

---
## Results

The LSTM with Attention model outperforms the SARIMA baseline across all evaluation metrics, demonstrating the effectiveness of attention mechanisms in capturing long-term temporal dependencies.

---
## Attention Visualization

Attention weights are extracted from the trained model and visualized using a heatmap.  
This provides interpretability by showing which past time steps the model focused on during prediction.

---
## Technologies Used

- Python  
- NumPy, Pandas  
- Scikit-learn  
- TensorFlow / Keras  
- Statsmodels  
- Matplotlib, Seaborn  

---
Project Structure
.
├── finalproject.py
├── README.md
├── requirements.txt
└── results/
Conclusion
This project demonstrates how attention-enhanced LSTM networks can significantly improve multivariate time series forecasting performance. The synthetic dataset enables controlled experimentation, while attention visualizations provide interpretability, making the model both accurate and explainable.


