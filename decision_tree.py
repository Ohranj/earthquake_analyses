import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
params = {'format': 'geojson', 'starttime': '2019-10-01', 'endtime': '2019-12-31', 'minlatitude': '18', 'maxlatitude': '54', 'minlongitude': '73', 'maxlongitude': '135'}

fetchData = np.array(requests.get(url, params=params).json()['features'])

data = []
for i in fetchData:
    data.append({'mag': i['properties']['mag'], 'depth': i['geometry']['coordinates'][2], 'longitude': i['geometry']['coordinates'][0], 'latitude': i['geometry']['coordinates'][1]})

data.reverse()
df = pd.DataFrame(data)

df.insert(4, 'highMag', np.where(df['mag'] >= 5.0, 1, 0), True)

x = df.drop(['highMag', 'mag'], axis=1)
y = df['highMag']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

DecisionTree_Class_Model = DecisionTreeClassifier()
DecisionTree_Class_Model.fit(x_train, y_train)

y_pred = DecisionTree_Class_Model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)