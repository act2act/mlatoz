import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Define the stock symbol
stock = 'AAPL'

# Get data on this ticker
data = yf.Ticker(stock)

# Get the historical prices for this ticker
hist = data.history(period="5y", interval="1d").resample('QE').mean()

# Overview of the data
print(hist.head()) # First 5 rows
print(hist.columns) # Column names
print(hist.info()) # Data types and missing values
print(hist.describe()) # Summary statistics

# Extract the closing price
closing_price = hist['Close']
print(closing_price)


# Visualize the closing price
sns.lineplot(x=closing_price.index, y=closing_price, marker='o')
plt.title('Time series plot of closing price')
plt.xlabel('Date')
plt.ylabel('Closing price')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix
# pd.set_option('display.max_columns', None)
# print(hist.corr())
sns.heatmap(hist.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()