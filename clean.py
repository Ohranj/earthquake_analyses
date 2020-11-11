import pandas as pd

#Remove unnecessary columns
earthquakeData = pd.read_csv('./earthquake-dataset.csv')
earthquakeData['Country'] = earthquakeData['Name'].str.split(':').str[0]
colsToKeep = ['Year', 'Month', 'Date', 'Tsunami', 'Latitude', 'Longitude', 'Focal Depth (km)', 'Magnitude','Deaths','Injuries', 'Country']
cleanEarthquakeData = earthquakeData[colsToKeep]
cleanEarthquakeData.to_csv('clean_earthquake_data.csv', index=False)
