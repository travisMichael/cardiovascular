import pandas as pd


df = pd.read_csv("cardio_train.csv", delimiter=";")

max_stats = df.max()

min_stats = df.min()

y = df.iloc[:,12].values
# Index(['a1_count', 'a1_mean', 'a1_std'], dtype='object')
x = df.iloc[:,1:11].values

print("done")