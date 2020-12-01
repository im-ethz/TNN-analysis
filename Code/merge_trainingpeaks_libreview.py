import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import datetime
import os

lv_path = 'Data/LibreView/clean/'
tp_path = 'Data/TrainingPeaks/csv/'

lv_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(lv_path)])

lv_files = {}
for i in lv_athletes:
	df = pd.read_csv(lv_path+str(i)+'.csv')
	lv_files.update({i:df})
df_lv = pd.concat(lv_files.values(), keys=lv_files.keys())

# drop empty columns
df_lv.dropna(axis=1, how='all', inplace=True)

# insert pandas datetime column
df_lv['Device Timestamp (datetime64)'] = pd.to_datetime(df_lv['Device Timestamp'], dayfirst=True)
df_lv['date'] = df_lv['Device Timestamp (datetime64)'].dt.date
df_lv['time'] = df_lv['Device Timestamp (datetime64)'].dt.time

# check for items with the same timestamp and record id that are duplicate
df_lv['ones'] = 1
df_lv_groupby = df_lv.groupby([df_lv.index.get_level_values(0), df_lv['date'], df_lv['time'], df_lv['Record Type']]).sum()
for i, date, time, r in df_lv_groupby[df_lv_groupby['ones'] != 1].index:
	print(df_lv[(df_lv.date == date) & (df_lv.time == time)].loc[i])
	print(df_lv_groupby.loc[i, date, time, r])
df_lv.drop('ones', axis=1, inplace=True)

# process glucose availability
datelist = pd.date_range(datetime.date(day=1, month=df_lv['date'].min().month, year=df_lv['date'].min().year),
						 datetime.date(day=30,month=df_lv['date'].max().month, year=df_lv['date'].max().year)).to_series()
df_lv_dates = pd.DataFrame.from_dict({i : datelist.isin(df_lv.loc[i, 'date']) for i in lv_athletes})

plt.figure(figsize=(10,4))
cmap_base = plt.cm.get_cmap('Blues')
cmap_custom = cmap_base.from_list('Blues_binary', cmap_base(np.linspace(0.1, 0.7, 2)), 2)
ax = sns.heatmap(df_lv_dates.T, cmap=cmap_custom, cbar=False)
ax.set_xticks(np.arange(0, len(df_lv_dates.index), 7))
ax.set_xticklabels([i.strftime('%d-%m-%Y') for i in df_lv_dates.index[::7]], rotation=45)
plt.yticks(rotation=0)
plt.xlabel('Date')
plt.ylabel('Athlete')
plt.title("Glucose availability")
plt.show()