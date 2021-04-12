import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *
from calc import *

path = 'Data/Dexcom/'

df = pd.read_csv(path+'dexcom_clean.csv', index_col=0)
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

# select glucose measurements
df = df[((df['Event Type'] == 'EGV') | (df['Event Type'] == 'Calibration'))]

# create calendar with glucose availability
dates_2019 = pd.date_range(start='12/1/2018', end='11/30/2019').date
glucose_avail = pd.DataFrame(index=df.RIDER.unique(), columns=dates_2019)
for i in df.RIDER.unique():
	glucose_avail.loc[i, df[df.RIDER == i].local_timestamp.dt.date.unique()] = 1
glucose_avail.fillna(0, inplace=True)

# plot glucose availability per day
plt.figure(figsize=(25,10))
ax = sns.heatmap(glucose_avail, cmap='Blues', cbar=False, linewidths=.01) 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig('Descriptives/glucose_availability_day_all.pdf', bbox_inches='tight')
plt.savefig('Descriptives/glucose_availability_day_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# aggregate to month level
glucose_avail.columns = pd.MultiIndex.from_arrays([pd.to_datetime(glucose_avail.columns).month, glucose_avail.columns])
glucose_avail = glucose_avail.T.groupby(level=0).sum().T
glucose_avail = glucose_avail[[glucose_avail.columns[-1]] + list(glucose_avail.columns[:-1])]
glucose_avail.columns = glucose_avail.columns.map(month_mapping)

# plot glucose availability per month
sns.heatmap(glucose_avail, annot=True, linewidth=.5, cmap='Greens', cbar_kws={'label': 'days'})
plt.xlabel('month')
plt.ylabel('rider')
plt.savefig('Descriptives/glucose_availability_month_all.pdf', bbox_inches='tight')
plt.savefig('Descriptives/glucose_availability_month_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# TODO all per athlete
# TODO split up athlete, and time-ranges (i.e. during training, nocturnal, etc.)