import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class FinancialDataset(Dataset):
    def __init__(self, data):
        # Data Scaling
        scaler = StandardScaler()
        features = data.drop(['Fiscal Quarters', 'Ticker', 'Target'], axis=1)
        scaled_features = scaler.fit_transform(features)

        # Tensor Transformation
        self.features = torch.tensor(scaled_features, dtype=torch.float)
        self.targets = torch.tensor(data['Target'].values, dtype=torch.float)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

stocks = ['AAPL', 'AMZN', 'ASML', 'AVGO', 'BRKA', 'GOOG', 'META', 'MSFT', 'NVDA', 'TSLA', 'TSM']

# [x] - for loop to read all the files and transform them from xlsx to csv
def convert_and_read_xlsx(stocks):
    for stock in stocks:
        base_path = f'data/{stock}_fa'
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
        df_csv = pd.read_csv(csv_path, na_values='-')
        df_csv = df_csv.drop([0, 1], axis=0)
        df_csv = df_csv.reset_index(drop=True)
        df_csv_transposed = df_csv.T # transpose the dataframe to visualize the data conveniently
        df_csv_transposed.columns = df_csv_transposed.iloc[0] # set the first row as the metrics name
        df_csv_transposed = df_csv_transposed[1:] # drop the first row
        df_csv_transposed = df_csv_transposed.reset_index().rename(columns={'index': 'Fiscal Quarters'}) # rename the columns
        df_csv_transposed['Ticker'] = stock # add the ticker column
        df_csv_transposed['Target'] = 0 # add the target column (whether the stock will be stalwartz or growth)

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

# convert_and_read_xlsx(stocks)
# eda(stocks)
train_loader, validation_loader, test_loader = preprocess(stocks)

# [] - import the model

# [] - train the model

# [] - test the model

# [] - make a prediction