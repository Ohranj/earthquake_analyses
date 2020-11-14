import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


workingData = pd.read_csv('./clean_earthquake_data.csv')

workingMagData = workingData.dropna(subset=['Magnitude'])

maxMagPerYear = {}
for i, row in workingMagData.iterrows():
    if row.Year in maxMagPerYear:
        if row.Magnitude > maxMagPerYear[row.Year]:
            maxMagPerYear[row.Year] = row.Magnitude
    else:
        maxMagPerYear[row.Year] = row.Magnitude

df = pd.DataFrame(list(maxMagPerYear.items()),columns = ['Year','Max magnitude']) 

data = df.filter(['Max magnitude'])
dataset = data.values
training_data_len = math.ceil(len(dataset) *.8)

scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, :]

x_train=[]
y_train = []
for i in range(30,len(train_data)):
    x_train.append(train_data[i-30:i,0])
    y_train.append(train_data[i,0])


x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

LSTM_model = Sequential()
LSTM_model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
LSTM_model.add(LSTM(units=50, return_sequences=False))
LSTM_model.add(Dense(units=25))
LSTM_model.add(Dense(units=1))

LSTM_model.compile(optimizer='adam', loss='mean_squared_error')

LSTM_model.fit(x_train, y_train, batch_size=32, epochs=5)

test_data = scaled_data[training_data_len - 30:,: ]

x_test = []
y_test =  dataset[training_data_len:, :] 
for i in range(30,len(test_data)):
    x_test.append(test_data[i-30:i,0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

predictions = LSTM_model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)

# rmse=np.sqrt(np.mean(((predictions- y_test)**2)))

# print(rmse)

# rollingMean = df[['Max magnitude']].rolling(6).mean()


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('LSTM prediction for proceeding years highest magnitude')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Max magnitude', fontsize=18)
plt.plot(train['Max magnitude'])
plt.plot(valid[['Max magnitude', 'Predictions']])
plt.grid(True, axis='y')
plt.legend(['Training data', 'Testing data', 'Predictions'], loc='lower right')
plt.show()