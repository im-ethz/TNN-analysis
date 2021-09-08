import os
import gc
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd

from plot import *
from helper import *
from calc import *
from config import rider_mapping

path = './export_eva/'

fnames = os.listdir(path)
fnames.sort()

# --------------------- read the individual 90 days files and merge
df_raw = [pd.read_csv(path+f) for f in fnames]
df = [] # CGM data
df_alert = [] # alerts data

for df_i in df_raw:
	# extract patient info of rider
	df_patient = df_i[(df_i['Event Type'] == 'FirstName') | (df_i['Event Type'] == 'LastName') | (df_i['Event Type'] == 'DateOfBirth')]

	# extract name and correct for name mistakes in dexcom CLARITY
	name = df_patient.loc[df_i['Event Type'] == 'LastName', 'Patient Info'].str.lower().replace({'declan':'irvine', 'clancey':'clancy'}).values[0]
	
	df_i.drop('Patient Info', axis=1, inplace=True)
	df_i.drop(df_patient.index, inplace=True)

	# convert timestamp
	df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'] = pd.to_datetime(df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'])

	df_i['RIDER'] = name

	# extract alert info
	df_a = df_i.loc[(df_i['Event Type'] == 'Device') | (df_i['Event Type'] == 'Alert'), ['Event Type', 'Event Subtype', 'Device Info', 'Source Device ID', 'Glucose Value (mg/dL)', 'Duration (hh:mm:ss)', 'Glucose Rate of Change (mg/dL/min)']]
	df_i.drop(df_a.index, inplace=True) # remove alert info from first lines of cgm data
	df_a['RIDER'] = name
	df_a['min_date'] = df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'].min()
	df_a['max_date'] = df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'].max()

	if not df_a.empty:
		df_alert.append(df_a)

	if not df_i.empty:
		df.append(df_i)

# --------------------- CGM data
df = pd.concat(df)

# remove useless and empty columns
df.drop('Index', axis=1, inplace=True)
df.drop(df.columns[df.isna().sum() / len(df) == 1], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)

# sort riders
df = df.reset_index().sort_values(by=['RIDER', 'index'], key=lambda x: x.map(rider_mapping)).drop('index', axis=1)

df.to_csv('TNN_CGM_2019_US_exported_20210819.csv')

# --------------------- alert data
df_alert_clean = pd.DataFrame(columns=['RIDER', 'min_date', 'max_date', 'Device Info', 'Source Device ID', 
										'High', 'Low', 'Urgent Low', 'Urgent Low Soon', 'Signal Loss'])
df_alert = pd.concat(df_alert)
df_alert.reset_index(drop=True, inplace=True)
df_alert['start_device'] = df_alert['Event Type'] == 'Device'
device_idx = df_alert['start_device'][df_alert['start_device']].index
for n in range(len(device_idx)-1):
	df_alert_n = df_alert[device_idx[n]:device_idx[n+1]]
	device_info = df_alert_n[['RIDER', 'min_date', 'max_date', 'Device Info', 'Source Device ID']].iloc[0].values
	glucose_info = df_alert_n[['Event Subtype', 'Glucose Value (mg/dL)']].set_index('Event Subtype').T
	try:
		glucose_info = glucose_info[['High', 'Low', 'Urgent Low', 'Urgent Low Soon']].values[0]
	except KeyError:
		try:
			high = glucose_info['High'].values[0]
		except KeyError:
			high = np.nan
		try:
			low = glucose_info['Low'].values[0]
		except KeyError:
			low = np.nan
		try:
			urgent_low = glucose_info['Urgent Low'].values[0]
		except KeyError:
			urgent_low = np.nan
		try:
			urgent_low_soon = glucose_info['Urgent Low Soon'].values[0]
		except KeyError:
			urgent_low_soon = np.nan
		glucose_info = [high, low, urgent_low, urgent_low_soon]
	try:
		duration_info = df_alert_n.loc[df_alert_n['Event Subtype'] == 'Signal Loss', 'Duration (hh:mm:ss)'].values
	except KeyError:
		duration_info = [np.nan]
	df_alert_clean.loc[n] = np.concatenate([device_info, glucose_info, duration_info])

dupl_last = df_alert_clean.drop_duplicates(subset=set(df_alert_clean.columns)-set(['min_date', 'max_date']), keep='last').reset_index(drop=True)
		
df_alert_clean.drop_duplicates(subset=set(df_alert_clean.columns)-set(['min_date', 'max_date']), inplace=True)
df_alert_clean.reset_index(drop=True, inplace=True)
df_alert_clean['max_date'] = dupl_last['max_date']

df_alert_clean.to_csv('TNN_CGM_2019_US_exported_20210819_alerts.csv')

# --------------------- alert data
# plot to check if this is really the missing data
df.rename(columns={'Timestamp (YYYY-MM-DDThh:mm:ss)':'timestamp'}, inplace=True)
dates_2019 = pd.date_range(start='12/1/2018', end='11/30/2019').date
glucose_avail = pd.DataFrame(index=df.RIDER.unique(), columns=dates_2019)
for i in df.RIDER.unique():
	glucose_avail.loc[i, df[df.RIDER == i].timestamp.dt.date.unique()] = 1
glucose_avail.fillna(0, inplace=True)

# plot glucose availability per day
plt.figure(figsize=(25,10))
ax = sns.heatmap(glucose_avail, cmap='Blues', cbar=False, linewidths=.01) 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.show()
plt.close()
