import yfinance as yf
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from predictor import Predictor

def fetch_stock_data(tickers):
    # Initialize an empty DataFrame for historical data
    historical_data = pd.DataFrame()

    for ticker in tickers:
        stock = yf.Ticker(ticker)

        # Fetch 10 years of historical data
        hist = stock.history(period="10y")

        # Get current info
        info = stock.info
        selected_info = {
            'CurrentPrice': info.get('currentPrice'),
            'MarketCap': info.get('marketCap'),
            'BookValue': info.get('bookValue'),
            'ProfitMargins': info.get('profitMargins'),
            'EarningsGrowth': info.get('earningsGrowth'),
            'RevenueGrowth': info.get('revenueGrowth'),
            'ReturnOnEquity': info.get('returnOnEquity'),
            'ForwardEPS': info.get('forwardEps'),
            'TrailingEPS': info.get('trailingEps'),
            'ForwardPE': info.get('forwardPE'),
            'TrailingPE': info.get('trailingPE'),
            'FreeCashflow': info.get('freeCashflow')
        }

        # Repeat the info data for each date in the historical data
        for key, value in selected_info.items():
            hist[key] = value

        # Add a column for the ticker symbol
        hist['Ticker'] = ticker

        # Use pd.concat to append this data to the main DataFrame
        historical_data = pd.concat([historical_data, hist], ignore_index=True)

    # Reset the index of the DataFrame
    historical_data.reset_index(inplace=True, drop=True)

    return historical_data

tickers = ["MSFT", "AAPL", "NVDA", "GOOG", "AMZN", "META", "TSM", "AVGO", "NVO", "TSLA"]

# Fetch historical data
historical_data = fetch_stock_data(tickers)

# Display the DataFrame
print(historical_data)
print(historical_data.columns)

# Check for missing values and data types
print(historical_data.info())

# Check for outliers
import seaborn as sns
import matplotlib.pyplot as plt

grouped = historical_data.groupby('Ticker')
continuous_columns = [
    'Close', 'Volume',
    'BookValue', 'ProfitMargins',
    'EarningsGrowth', 'RevenueGrowth', 'ReturnOnEquity',
    'ForwardEPS', 'TrailingEPS', 'ForwardPE', 'TrailingPE', 'FreeCashflow'
]

for name, group in grouped:
    print(f"The {name}'s summary statistics: {group.describe()}") # Display summary statistics for each group

    # Plot boxplots for each column in the group
    # for column in group.columns:
    #     if column in continuous_columns:
    #         sns.boxplot(data=group[column])
    #         plt.title(f"{name}: {column}")
    #         plt.show()

# Initialize the StandardScaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

def preprocess(data, scaler_X, scaler_Y):
    # Split the data into numerical and ticker features
    numerical_data = data.iloc[:, :-1]

    # Define the input X and target Y using time lagged data
    X = numerical_data.iloc[:-1]
    Y = numerical_data['Close'].iloc[1:].values.reshape(-1, 1)

    print(f"X: {X}, and shape: {X.shape}")
    print(f"Y: {Y}, and shape: {Y.shape}")

    # Set ratio for train, validation, and test sets
    train_ratio = 0.7
    validation_ratio = 0.2
    test_ratio = 0.1

    # Calculate the number of points for each set
    total_points = X.shape[0]
    train_points = int(total_points * train_ratio)
    validation_points = int(total_points * validation_ratio)
    test_points = total_points - train_points - validation_points

    # Split the data sequentially
    train_X = X[:train_points]
    validation_X = X[train_points:train_points + validation_points]
    test_X = X[train_points + validation_points:]

    train_Y = Y[:train_points]
    validation_Y = Y[train_points:train_points + validation_points]
    test_Y = Y[train_points + validation_points:]

    # Standardize the data
    train_X = scaler_X.fit_transform(train_X)
    validation_X = scaler_X.transform(validation_X)
    test_X = scaler_X.transform(test_X)

    train_Y = scaler_Y.fit_transform(train_Y)
    validation_Y = scaler_Y.transform(validation_Y)
    test_Y = scaler_Y.transform(test_Y)

    # Convert the DataFrames to PyTorch tensors
    train_X = torch.tensor(train_X, dtype=torch.float32)
    validation_X = torch.tensor(validation_X, dtype=torch.float32)
    test_X = torch.tensor(test_X, dtype=torch.float32)

    train_Y = torch.tensor(train_Y, dtype=torch.float32)
    validation_Y = torch.tensor(validation_Y, dtype=torch.float32)
    test_Y = torch.tensor(test_Y, dtype=torch.float32)

    return train_X, validation_X, test_X, train_Y, validation_Y, test_Y

train_X, validation_X, test_X, train_Y, validation_Y, test_Y = preprocess(historical_data, scaler_X, scaler_Y)
print(f"Train X: {train_X}, and shape: {train_X.shape}")
print(f"Train Y: {train_Y}, and shape: {train_Y.shape}")

# Define hyperparameters
input_dim = train_X.shape[1]
hidden_dim1 = 128
hidden_dim2 = 64
output_dim = 1
learning_rate = 0.001
epochs = 5000
patience = 10
best_loss = float('inf')
counter = 0

# Initialize the model
predi = Predictor(input_dim, hidden_dim1, hidden_dim2, output_dim)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(predi.parameters(), lr=learning_rate)

# Train the model
for epoch in range(epochs):
    # Set the model to training mode
    predi.train()

    # Forward pass
    outputs = predi(train_X)
    loss = criterion(outputs, train_Y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Set the model to evaluation mode
    predi.eval()

    # Validation loss
    with torch.no_grad():
        validation_output = predi(validation_X)
        validation_loss = criterion(validation_output, validation_Y)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Validation Loss: {validation_loss.item():.4f}")

    # Early stopping
    if validation_loss.item() < best_loss:
        best_loss = validation_loss.item()
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Stopped at epoch {epoch} with validation loss {validation_loss.item()}")
        break

    predi.train() # Set the model back to training mode

# Test the model
predi.eval()
with torch.no_grad():
    test_output = predi(test_X)
    test_loss = criterion(test_output, test_Y)
    print(f"Test Loss: {test_loss.item():.4f}")

# Make predictions
new_data = fetch_stock_data(["NFLX"]) # Fetch new data
new_train_X, _, _, _, _, _ = preprocess(new_data, scaler_X, scaler_Y) # Preprocess the new data

predi.eval()
with torch.no_grad():
    new_prediction = predi(new_train_X) # Make predictions
    prediction_numpy = new_prediction.numpy() # Convert the tensor to a NumPy array
    prediction_actual = scaler_Y.inverse_transform(prediction_numpy) # Inverse transform the predictions

    # np.set_printoptions(threshold=np.inf) # Display the entire array
    print(f"New data prediction: {prediction_actual}")