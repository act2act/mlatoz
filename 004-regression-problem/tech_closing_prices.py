import yfinance as yf
import pandas as pd
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
    hist = stock.history(period="5y", interval="1d").resample("QE").mean()
    closing_price = hist["Close"]

    closing_prices[ticker] = closing_price
print(closing_prices)

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

# Define the input X and target Y using time lagged data
X = closing_prices.iloc[:-1]
Y = closing_prices.iloc[1:]

# Data splitting
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# 각 세트에 할당할 데이터 포인트의 수 계산
total_points = closing_prices.shape[0]
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

# 학습 파라미터 설정
epochs = 1000
learning_rate = 0.001
input_dim = len(tickers)
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
    print(f"Validation Loss: {validation_loss.item()}")
