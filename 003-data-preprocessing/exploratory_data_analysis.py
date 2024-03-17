import pandas as pd

# Load the data
df = pd.read_csv('data/exercise_data.csv')

# Data Overview
## Check the first 5 rows of the dataset
# print(df.head())

## Check the fundamental information of the dataset
print(df.info)
# print(df.describe())

"""
Columns of the dataset:
- distance: the distance (in km) that I walked/jogged/ran basically total distance I traversed that day.
- rhr: resting heart rate (in bpm) for that day.
- zone mins: minutes of vigorous exercise for that day.
"""

# Check missing values
# print(df.isnull().sum())

## If there are missing values,
## Drop the missing values
# df.dropna(inplace=True)

## Or, fill the missing values with the mean value of the column
# df.fillna(df.mean(), inplace=True)

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# sns.histplot(df['distance'], kde=True)

## Or, Check the skewness and kurtosis of the dataset using numerical methods
from scipy.stats import skew, kurtosis
# skewness = df.skew()
# kurtosis = df.kurtosis()
# print("Skewness: ", skewness)
# print("Kurtosis: ", kurtosis)

# sns.scatterplot(x='distance', y='rhr', data=df)

## Check the correlation between the columns
correlation_matrix = df.corr()
# print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True)

plt.show()