import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from predictor import Predictor

tickers = ["MSFT", "AAPL", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "NVO", "TSLA"]

closing_prices = pd.DataFrame()

for ticker in tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y", interval="1d").resample("QE").mean()
    closing_price = hist["Close"]

    closing_prices[ticker] = closing_price
print(closing_prices)

# Load the new data for prediction
new_close = pd.DataFrame()

new_tickers = ["ASML",]
for ticker in new_tickers:
    stock = yf.Ticker(ticker)
    hist = stock.history(period="10y", interval="1d").resample("QE").mean()
    closing_price = hist["Close"]

    new_close[ticker] = closing_price
print(f"New data for prediction: {new_close}")
print(f"New data shape: {new_close.shape}")

# Overview of the data
# print(closing_prices.info())  # check for missing values and data types
# print(closing_prices.describe())  # check for outliers

# for stock in closing_prices.columns:
#     sns.boxplot(closing_prices[stock])
#     plt.title(f'Boxplot of closing price for {stock}')
#     plt.xlabel('Closing price')
#     plt.show()
#
#     sns.histplot(closing_prices[stock], kde=True)
#     plt.title(f'Histogram of closing price for {stock}')
#     plt.xlabel('Closing price')
#     plt.show()

# Preprocessing
def preprocess(data):
    # Define the input X and target Y using time lagged data
    X = data.iloc[:-1]
    Y = data.iloc[1:]

    # Data splitting
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # 각 세트에 할당할 데이터 포인트의 수 계산
    total_points = data.shape[0]
    train_points = int(total_points * train_ratio)
    validation_points = int(total_points * validation_ratio)
    test_points = total_points - train_points - validation_points

    # 데이터 순서대로 분할
    train_X = X[:train_points]
    train_Y = Y[:train_points]

    validation_X = X[train_points:train_points + validation_points]
    validation_Y = Y[train_points:train_points + validation_points]

    test_X = X[train_points + validation_points:]
    test_Y = Y[train_points + validation_points:]

    # 데이터를 PyTorch 텐서로 변환
    train_X = torch.tensor(train_X.values, dtype=torch.float32)
    train_Y = torch.tensor(train_Y.values, dtype=torch.float32)

    validation_X = torch.tensor(validation_X.values, dtype=torch.float32)
    validation_Y = torch.tensor(validation_Y.values, dtype=torch.float32)

    test_X = torch.tensor(test_X.values, dtype=torch.float32)
    test_Y = torch.tensor(test_Y.values, dtype=torch.float32)

    return train_X, train_Y, validation_X, validation_Y, test_X, test_Y

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = preprocess(closing_prices)
new_train_X, new_train_Y, new_validation_X, new_validation_Y, new_test_X, new_test_Y = preprocess(new_close)

# new_train_X의 형태를 조정
predict_X = torch.zeros(new_train_X.size(0), 10)  # 10개의 특성을 가진 새로운 텐서 생성
predict_X[:, 0] = new_train_X.squeeze()  # 첫 번째 특성에만 원래 데이터를 넣고 나머지는 0으로 남김

# 학습 파라미터 설정
epochs = 1000
learning_rate = 0.001
input_dim = train_X.shape[1]
hidden_dim1 = 100
hidden_dim2 = 50
output_dim = len(tickers)

# 모델 생성
predictor = Predictor(input_dim, hidden_dim1, hidden_dim2, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()
    output = predictor(train_X)
    loss = criterion(output, train_Y)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

    # 모델 평가
    with torch.no_grad():
        validation_output = predictor(validation_X)
        validation_loss = criterion(validation_output, validation_Y)
        if epoch % 100 == 0:
            print(f"Validation Loss: {validation_loss.item()}")

with torch.no_grad():
    test_output = predictor(test_X)
    test_loss = criterion(test_output, test_Y)
    print(f"Test Loss: {test_loss.item()}")

# 새로운 데이터에 대한 예측
predictor.eval()
with torch.no_grad():
    prediction = predictor(predict_X)

    print(f"New data prediction: {prediction[:, 0]}")