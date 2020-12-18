import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
params = {'format': 'geojson', 'starttime': '2019-06-01', 'endtime': '2019-12-31', 'minlatitude': '18', 'maxlatitude': '54', 'minlongitude': '73', 'maxlongitude': '135'}

fetchData = np.array(requests.get(url, params).json()['features'])[::-1] 

data = []
for i in fetchData:
    data.append({'mag': i['properties']['mag'], 'time': i['properties']['time'], 'depth': i['properties']['tz']})

df1 = pd.DataFrame(data)
df2 = df1[['mag']]

days_to_predict = len(df1[df1['time'] > 1575158400000])
df2['Prediction'] = df2[['mag']].shift(-days_to_predict)
X = np.array(df2.drop(['Prediction'], 1))[:-days_to_predict]
y = np.array(df2['Prediction'])[:-days_to_predict]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

Decision_tree = DecisionTreeRegressor().fit(x_train, y_train)

x_future = df2.drop(['Prediction'], 1)[:-days_to_predict]
x_future = x_future.tail(days_to_predict)
x_future = np.array(x_future)

valid = df2[X.shape[0]:]
valid['Predictions'] = Decision_tree.predict(x_future)

plt.figure(figsize=(12, 6))
plt.title('Decision tree prediction of earthquake magnitudes occuring throughout December 2019 - China')
plt.xlabel('Earthquake mangnitudes through June 1st 2019 - December 31st 2019')
plt.ylabel('Magnitude')
plt.plot(df2['mag'])
plt.plot(valid[['mag', 'Predictions']])
plt.legend(['Training data', 'December Testing data', 'December prediction'])
plt.xlim((0,436))
plt.show()
