import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import os

#https://randerson112358.medium.com/stock-price-prediction-using-python-machine-learning-e82a039ac2bb
##

os.system('python clean.py')
os.system('python decision_tree.py')
os.system('python lr_regress.py')


#Read cleaned dataset
workingData = pd.read_csv('./clean_earthquake_data.csv')


#Graph the average number of earthquakes per year and showing the trend line per decade
eachYear = workingData.Year.unique()
countFreq = workingData.Year.value_counts().sort_index()

yearlyFreq = list(zip(eachYear, countFreq))

decadeTally = {}
for y, v in yearlyFreq:
    getDecade = y // 10 * 10
    decadeTally[getDecade] = round(decadeTally.get(getDecade, 0) + (v / 10))

decade, decadeAverage = zip(*decadeTally.items())

plt.bar(eachYear, countFreq, label='Frequency per year', color='orange')
plt.plot(decade, decadeAverage, color='green', label='Decade average frequency', linestyle='--')

plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Global frequency of major earthquakes per year')
plt.legend()
plt.grid(True, axis='y')
plt.margins(x=0)
plt.tight_layout()

plt.show()





# Graph max magnitude per year
workingMagData = workingData.dropna(subset=['Magnitude'])

maxMagPerYear = {}
for i, row in workingMagData.iterrows():
    if row.Year in maxMagPerYear:
        if row.Magnitude > maxMagPerYear[row.Year]:
            maxMagPerYear[row.Year] = row.Magnitude
    else:
        maxMagPerYear[row.Year] = row.Magnitude


maxMagYear, maxMagnitude = zip(*maxMagPerYear.items())
plt.scatter(maxMagYear, maxMagnitude, color='green', label='Earthquake Magnitude', s=10)

plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Maximum earthquake magnitude per year')
plt.legend()
plt.grid(True, axis='y')
plt.margins(x=0)
plt.tight_layout()

plt.show()





#Graph to show occurances per a month - For those with more than 10 eartquakes
workingData = workingData.dropna(subset=['Month', 'Country'])[['Month', 'Country']]

cntryPerMonth = workingData.groupby(workingData.columns.tolist(), as_index=False).size()
cntryPerMonth = cntryPerMonth[cntryPerMonth['size'].map(int) >= 5].sort_values(['Country', 'Month']).rename(columns={'size': 'Occurances'})

monthOrder = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

cntryCellVal = 1.0
while cntryCellVal <= 12.0:
    cntryPerMonth.loc[cntryPerMonth['Month'] == cntryCellVal, 'Month'] = monthOrder[round(cntryCellVal - 1)]
    cntryCellVal += 1


cntryPerMonth.Month = pd.CategoricalIndex(cntryPerMonth.Month, categories=monthOrder, ordered=True)
cntryPerMonth = cntryPerMonth.sort_index()

plt.figure(figsize=(9, 8))

heatmapData = cntryPerMonth.pivot('Country', 'Month', 'Occurances')
sb.heatmap(heatmapData, cmap="YlOrRd", annot=True)

plt.title('Number of major earthquakes (>= 5) per month since 1600')
plt.ylabel('Country / Territory')
plt.tight_layout()
plt.show()




