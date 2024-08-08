# Time Series Stock Prediction using RNN and LSTM

## Overview

This project leverages Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models to predict stock prices based on historical market data. By analyzing features such as opening and closing prices, volume, and market trends, the model identifies optimal buying and selling opportunities to maximize investment returns.

## Features

- **Stock Price Prediction**: Uses RNN and LSTM to forecast future stock prices.
- **Data Normalization**: Ensures accurate predictions by normalizing input data.
- **Advanced Neural Networks**: Utilizes deep learning techniques for improved accuracy.
- **Optimized Performance**: Fine-tuned hyperparameters for better model performance.

## Project Structure

- **`data/`**: Contains the dataset used for training and testing the model.
- **`models/`**: Includes saved model files.
- **`scripts/`**: Python scripts for data preprocessing, model training, and evaluation.
- **`notebooks/`**: Jupyter notebooks used for experimentation and visualization.
- **`results/`**: Output graphs, predictions, and performance metrics.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (optional, for running notebooks)
- Libraries: `numpy`, `pandas`, `tensorflow`, `keras`, `matplotlib`, `scikit-learn`, `pyspark`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/StockPrediction.git
   cd StockPrediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Data Preprocessing**: 
   Run the `data_preprocessing.py` script to clean and normalize the data.
   ```bash
   python scripts/data_preprocessing.py
   ```

2. **Model Training**: 
   Train the LSTM model using the preprocessed data.
   ```bash
   python scripts/train_model.py
   ```

3. **Evaluation**: 
   Evaluate the model's performance using test data.
   ```bash
   python scripts/evaluate_model.py
   ```

4. **Prediction**: 
   Use the trained model to predict future stock prices.
   ```bash
   python scripts/predict_stock.py
   ```

### Results

- The model's predictions closely align with actual stock prices, as shown in the `results/` directory.
- The optimal batch size and epoch values were determined to minimize RMSE and prevent overfitting.


## Acknowledgments

- Thanks to [Kaggle](https://www.kaggle.com) for providing the dataset used in this project.
- Inspired by research papers on stock market prediction using deep learning models.

