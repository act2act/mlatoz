import seaborn as sns
import matplotlib as plt
import pandas as pd
import ssl
import certifi

ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(certifi.where())

iris = sns.load_dataset('iris')

print(iris.head())