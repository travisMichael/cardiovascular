import pandas as pd
import pickle
import numpy as np
import os


df = pd.read_csv("cardio_train.csv", delimiter=";")

max_stats = df.max()
min_stats = df.min()

# df_positive = df[df['cardio'] == 1]
# df_negative = df[df['cardio'] == 0]
# df = df_negative

df['age'] = df['age'].apply(lambda x: x / max_stats['age'])
df['height'] = df['height'].apply(lambda x: x / max_stats['height'])
df['weight'] = df['weight'].apply(lambda x: x / max_stats['weight'])
df['ap_hi'] = df['ap_hi'].apply(lambda x: x / max_stats['ap_hi'])
df['ap_lo'] = df['ap_lo'].apply(lambda x: x / max_stats['ap_lo'])


y = np.array(df.iloc[:,12])
x = df.iloc[:,1:11]

x_train = x.iloc[0:56000, :].values
y_train = y[0:56000]

x_test = x.iloc[56000:, :].values
y_test = y[56000:]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

if not os.path.exists('data/train'):
    os.makedirs('data/train')
if not os.path.exists('data/test'):
    os.makedirs('data/test')

x_train_file = open('data/train/x', 'wb')
y_train_file = open('data/train/y', 'wb')
x_test_file = open('data/test/x', 'wb')
y_test_file = open('data/test/y', 'wb')

pickle.dump(x_train, x_train_file)
pickle.dump(y_train, y_train_file)
pickle.dump(x_test, x_test_file)
pickle.dump(y_test, y_test_file)

x_train_file.close()
y_train_file.close()
x_test_file.close()
y_test_file.close()

print("done")