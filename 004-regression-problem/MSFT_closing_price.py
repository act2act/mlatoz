import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

msft = yf.Ticker("MSFT")

# get all stock info
for key, value in msft.info.items():
    print(key, ":", value)

# get historical price
hist = msft.history(period="5y", interval="1d").resample("QE").mean()

# get closing price
closing_price = hist["Close"]

# visualize closing price
sns.set(style="darkgrid")
sns.lineplot(x=closing_price.index, y=closing_price, marker='o')
plt.title('Time series plot of closing price')
plt.xlabel('Date')
plt.ylabel('Closing price')
plt.xticks(rotation=45)
plt.show()

# overview of the data
print(closing_price.info()) # check for missing values and data types

print(closing_price.describe()) # check for outliers
sns.boxplot(closing_price)
plt.title('Boxplot of closing price')
plt.xlabel('Closing price')
plt.show()

sns.histplot(closing_price, kde=True)
plt.title('Histogram of closing price')
plt.xlabel('Closing price')
plt.show()