import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from classificator import Classificator

class FinancialDataset(Dataset):
    def __init__(self, data):
        # Data Scaling
        scaler = StandardScaler()
        features = data.drop(['Fiscal Quarters', 'Ticker', 'Target'], axis=1)
        features = features.fillna(features.mean())
        scaled_features = scaler.fit_transform(features)

        # Tensor Transformation
        self.features = torch.tensor(scaled_features, dtype=torch.float)
        self.targets = torch.tensor(data['Target'].values, dtype=torch.float)
        self.tickers = data['Ticker'].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.tickers[idx]

stocks = ['AAPL', 'AMZN', 'ASML', 'AVGO', 'BRKA', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA', 'TSM']

# [x] - for loop to read all the files and transform them from xlsx to csv
def convert_and_read_xlsx(stocks):
    for stock in stocks:
        base_path = f'data/{stock}_fa'
        if not os.path.exists(base_path):
            base_path = f'new_data/{stock}_fa'
        csv_path = base_path + '.csv'
        xlsx_path = base_path + '.xlsx'

        if os.path.exists(csv_path):
            df_csv = pd.read_csv(csv_path)
            print(f"Reading from CSV for {stock}:")
            print(df_csv.head())
        elif os.path.exists(xlsx_path):
            df_xlsx = pd.read_excel(xlsx_path)

            df_xlsx.to_csv(csv_path, index=False)

            df_csv = pd.read_csv(csv_path)
            print(f"Converted XLSX to CSV for {stock}, now reading CSV:")
            print(df_csv.head())

# [x] - Exploratory Data Analysis
def eda(stocks):
    for stock in stocks:
        csv_path = f'data/{stock}_fa.csv'
        df_csv = pd.read_csv(csv_path, na_values='-')

        df_csv = df_csv.drop([0, 1], axis=0) # drop the first two rows
        df_csv = df_csv.reset_index(drop=True) # reset the index

        print(f"Exploratory Data Analysis for {stock}:")
        print(df_csv.head())
        print(df_csv.info())
        print(df_csv.describe())

        df_csv_transposed = df_csv.T # transpose the dataframe to visualize the data conveniently
        df_csv_transposed.columns = df_csv_transposed.iloc[0]
        df_csv_transposed = df_csv_transposed[1:]
        df_csv_transposed = df_csv_transposed.reset_index().rename(columns={'index': 'Fiscal Quarters'})
        df_long = df_csv_transposed.melt(id_vars='Fiscal Quarters', var_name='Metrics', value_name='Values')
        plt.figure(figsize=(20, 8))
        sns.lineplot(data=df_long, x='Fiscal Quarters', y='Values', hue='Metrics')
        plt.title(f'{stock}\'s Trend of Financial Metrics Over Fiscal Quarters')
        plt.xlabel('Fiscal Quarters')
        plt.ylabel('Metric Values')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()

# [] - preprocessing the data
def preprocess(stocks):
    # [x] - data merging with data loader and feature engineering
    combined_data = pd.DataFrame()
    train_list = []
    validation_list = []
    test_list = []

    for stock in stocks:
        csv_path = f'data/{stock}_fa.csv'
        if not os.path.exists(csv_path):
            csv_path = f'new_data/{stock}_fa.csv'

        df_csv = pd.read_csv(csv_path, na_values='-')
        df_csv = df_csv.drop([0, 1], axis=0)
        df_csv = df_csv.reset_index(drop=True)
        df_csv_transposed = df_csv.T # transpose the dataframe to visualize the data conveniently
        df_csv_transposed.columns = df_csv_transposed.iloc[0] # set the first row as the metrics name
        df_csv_transposed = df_csv_transposed[1:] # drop the first row
        df_csv_transposed = df_csv_transposed.reset_index().rename(columns={'index': 'Fiscal Quarters'}) # rename the columns
        df_csv_transposed['Ticker'] = stock # add the ticker column
        df_csv_transposed['Target'] = label_stalwartz_growth(df_csv_transposed) # add the target column (whether the stock will be stalwartz or growth)

        # Data Splitting
        train_ratio = 0.7
        validation_ratio = 0.2
        test_ratio = 0.1
        train_size = int(train_ratio * len(df_csv_transposed))
        validation_size = int(validation_ratio * len(df_csv_transposed))
        test_size = len(df_csv_transposed) - train_size - validation_size

        train_list.append(df_csv_transposed[:train_size])
        validation_list.append(df_csv_transposed[train_size:train_size+validation_size])
        test_list.append(df_csv_transposed[train_size+validation_size:])

        combined_data = pd.concat([combined_data, df_csv_transposed], axis=0)

    train_data = pd.concat(train_list)
    validation_data = pd.concat(validation_list)
    test_data = pd.concat(test_list)

    X_train = FinancialDataset(train_data)
    X_validation = FinancialDataset(validation_data)
    X_test = FinancialDataset(test_data)

    train_loader = DataLoader(X_train, batch_size=32)
    validation_loader = DataLoader(X_validation, batch_size=32)
    test_loader = DataLoader(X_test, batch_size=32)

    return train_loader, validation_loader, test_loader

def label_stalwartz_growth(df):
    # Decide whether the stock is stalwartz or growth
    target_list = []
    criteria = {
        'PE': 25,
        'ROE': 15,
        'RevCAGR': 10,
        'NICAGR': 10,
        'NNICAGR': 10
    }

    for _, _ in df.iterrows():
        growth_point = 0
        for metric in df.columns.drop(['Fiscal Quarters', 'Ticker']):
            recent_metric = df[metric].tail(3)
            if metric == 'Price / Earnings - P/E (LTM)':
                if recent_metric.iloc[0] > criteria['PE'] and recent_metric.iloc[0] < recent_metric.iloc[1] < recent_metric.iloc[2]:
                    growth_point += 1
            elif metric == 'Return On Equity %':
                if recent_metric.iloc[0] > criteria['ROE'] and recent_metric.iloc[0] < recent_metric.iloc[1] < recent_metric.iloc[2]:
                    growth_point += 1
            elif metric == 'Total Revenues / CAGR 5Y':
                if recent_metric.iloc[0] > criteria['RevCAGR'] and recent_metric.iloc[0] < recent_metric.iloc[1] < recent_metric.iloc[2]:
                    growth_point += 1
            elif metric == 'Net Income / CAGR 5Y':
                if recent_metric.iloc[0] > criteria['NICAGR'] and recent_metric.iloc[0] < recent_metric.iloc[1] < recent_metric.iloc[2]:
                    growth_point += 1
            elif metric == 'Normalized Net Income / CAGR 5Y':
                if recent_metric.iloc[0] > criteria['NNICAGR'] and recent_metric.iloc[0] < recent_metric.iloc[1] < recent_metric.iloc[2]:
                    growth_point += 1
        target_list.append(1) if growth_point >= 1 else target_list.append(0)

    return target_list


# convert_and_read_xlsx(stocks)
# eda(stocks)
train_loader, validation_loader, test_loader = preprocess(stocks)

# [x] - import the model
first_batch = next(iter(train_loader))
features, targets, tickers = first_batch

input_dim = features.shape[1]
hidden_dim1 = 64
hidden_dim2 = 32
output_dim = 2

classi = Classificator(input_dim, hidden_dim1, hidden_dim2, output_dim)

# [x] - train the model
epochs = 5000
learning_rate = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classi.parameters(), lr=learning_rate)
patience = 10
best_loss = float('inf')
counter = 0

for epoch in range(epochs):
    classi.train() # set the model to training mode
    train_loss = 0
    for features, targets, tickers in train_loader:
        optimizer.zero_grad()
        outputs = classi(features)
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    classi.eval() # set the model to evaluation mode
    validation_loss = 0
    with torch.no_grad():
        for features, targets, tickers in validation_loader:
            outputs = classi(features)
            loss = criterion(outputs, targets.long())
            validation_loss += loss.item()

    validation_loss /= len(validation_loader)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Train Loss: {loss:.4f}, Validation Loss: {validation_loss:.4f}")

    # Early stopping
    if validation_loss < best_loss:
        best_loss = validation_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Stopped at epoch {epoch} with validation loss {validation_loss:.4f}")
        break

# [] - test the model
classi.eval() # set the model to evaluation mode
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for features, targets, tickers in test_loader:
        outputs = classi(features)
        loss = criterion(outputs, targets.long())
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    test_loss /= len(test_loader)
    accuracy = correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

# [] - make a prediction
new_stocks = ['LLY', 'NVO', 'V']
convert_and_read_xlsx(new_stocks)
new_train_loader, _, _ = preprocess(new_stocks)

class_labels = {0: 'stalwartz', 1: 'growth'}
classi.eval() # set the model to evaluation mode
with torch.no_grad():
    for features, targets, tickers in new_train_loader:
        outputs = classi(features)
        _, predicted = torch.max(outputs, 1)
        predicted_labels = [class_labels[p.item()] for p in predicted]

        for ticker, label in zip(tickers, predicted_labels):
            print(f"{ticker} is predicted as {label}.")