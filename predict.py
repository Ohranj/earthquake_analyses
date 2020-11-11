import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
params = {'format': 'geojson', 'starttime': '2019-09-01', 'endtime': '2019-12-31', 'minlatitude': '18', 'maxlatitude': '54', 'minlongitude': '73', 'maxlongitude': '135'}

fetchData = np.array(requests.get(url, params=params).json()['features'])

data = []
for i in fetchData:
    data.append({'time': i['properties']['time'], 'mag': i['properties']['mag'], 'significance': i['properties']['sig'], 'longitude': i['geometry']['coordinates'][0], 'latitude': i['geometry']['coordinates'][1]})

data.reverse()
data = pd.DataFrame(data)

df = data[['mag']]





#Create variable to predict into future
future_days = 25

#Create a new column with target data shifted x units
df['Prediction'] = df[['mag']].shift(-future_days)

#Create feature dataset (X) and convert to np array and remove last x rows
X = np.array(df.drop(['Prediction'], 1))[:-future_days]

#Create target dataset (y) and convert to np array get target values except last x rows
y = np.array(df['Prediction'])[:-future_days]

#Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Create decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)

#Create linear regression model
lr = LinearRegression().fit(x_train, y_train)

#Get the last x rows of the feature dataset
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

#Show the model tree prediction
tree_prediction = tree.predict(x_future)

print(tree_prediction)
print()
#Show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)

#Visualise the data
predictions = tree_prediction
valid = df[X.shape[0]:]

valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(df['mag'])
plt.plot(valid[['mag', 'Predictions']])
plt.legend(['Orig', 'val', 'pred'])
plt.show()


predictions = lr_prediction
valid = df[X.shape[0]:]

valid['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.plot(df['mag'])
plt.plot(valid[['mag', 'Predictions']])
plt.legend(['Orig', 'val', 'pred'])
plt.show()



#https://www.youtube.com/watch?v=hOLSGMEEwlI&t=317s