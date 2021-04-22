# TODO: check if we indeed can use event subtype to fill extremes with
# TODO: check why there can be duplicate timestamps again
# TODO: why duplicates (doesn't matter though, just remove)
# TODO: why duplicate timestamps in there?
import os
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from pandas_profiling import ProfileReport

from plot import *
from helper import *
from calc import *
from config import rider_mapping

path = './'
if not os.path.exists(path+'dupl/'):
	os.mkdir(path+'dupl/')
if not os.path.exists(path+'exceed/'):
	os.mkdir(path+'exceed/')
if not os.path.exists(path+'exceed/without_dupl/'):
	os.mkdir(path+'exceed/without_dupl/')

df_excel = pd.read_excel(path+'TNN_CGM_2019.xlsx', sheet_name=None)

# quick clean
df_excel['kamstra'] = df_excel.pop('Sheet4')
df_excel['peron'].drop(0, inplace=True)

# --------------------- descriptives
for i, df_i in df_excel.items():
	print("\n------------------------------- Athlete ", i)

	print("Riders in the file for athlete %s: %s"%(i, df_i.RIDER.unique()))

	# timestamps
	ts = pd.to_datetime(df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'])
	print("First glucose measurement: ", ts.min())
	print("Last glucose measurement: ", ts.max())
	# event types
	print("Event types: ", df_i['Event Type'].unique())

	# devices
	print("Devices: ", df_i['Source Device ID'].unique())

	# calibration
	print("Number of times Event Type == Calibration",
		(df_i['Event Type'] == 'Calibration').sum())
	print("Number of times Event Type == Calibration and Source Device ID != Android G6 or iPhone G6",
		((df_i['Event Type'] == 'Calibration') & ((df_i['Source Device ID'] != 'Android G6') | df_i['Source Device ID'] != 'iPhone G6')).sum())
	
	print("Fraction of nans")
	print(df_i.isna().sum() / len(df_i))

# --------------------- combine, anonymize and first clean
df = pd.DataFrame(columns=list(df_excel.values())[0].columns)
for i, df_i in df_excel.items():
	df = df.append(df_i)

df.reset_index(drop=True, inplace=True)

# anonymize file
df.RIDER = df.RIDER.apply(lambda x: rider_mapping[x.lower()])

# timestamp
df['Timestamp (YYYY-MM-DDThh:mm:ss)'] = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)'])
df.rename({'Timestamp (YYYY-MM-DDThh:mm:ss)':'local_timestamp'}, axis=1, inplace=True)

# remove useless and empty columns
df.drop('Index', axis=1, inplace=True)
df.drop(df.columns[df.isna().sum() / len(df) == 1], axis=1, inplace=True)

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(path+'dexcom_raw.csv')

# create pandas profiling report
profile = ProfileReport(df, title='pandas profiling report', minimal=True)
profile.to_file(path+'report_raw.html')

# --------------------- clean
# remove high and low from glucose values
df['Glucose Value (mmol/L) EXTREME'] = df['Glucose Value (mmol/L)'].apply(lambda x: x if isinstance(x, str) else np.nan)
df['Glucose Value (mg/dL) EXTREME'] = df['Glucose Value (mg/dL)'].apply(lambda x: x if isinstance(x, str) else np.nan)

df['Glucose Value (mmol/L)'] = pd.to_numeric(df['Glucose Value (mmol/L)'], errors='coerce')
df['Glucose Value (mg/dL)'] = pd.to_numeric(df['Glucose Value (mg/dL)'], errors='coerce')

# fill up glucose values mg/dL with mmol/L
df['Glucose Value (mg/dL)'] = df['Glucose Value (mg/dL)'].fillna(df['Glucose Value (mmol/L)'] * mmoll_mgdl)
df['Glucose Value (mg/dL) EXTREME'] = df['Glucose Value (mg/dL) EXTREME'].fillna(df['Glucose Value (mmol/L) EXTREME']) 
print("FILLNA Glucose Value mg/dL with mmol/L to mg/dL")
print("FILLNA Glucose Value mg/dL EXTREME with mmol/L")

df.drop(['Glucose Value (mmol/L)', 'Glucose Value (mmol/L) EXTREME'], axis=1, inplace=True)

# clean zeros glucose
df.loc[(df['Event Type'] != 'EGV') & (df['Event Type'] != 'Calibration'), 'Glucose Value (mg/dL)'] = np.nan
print("CHECK Are there remaining zero glucose values: ", not df[df['Glucose Value (mg/dL)'] == 0].empty)

# fix event subtype instead of glucose value extreme
mask_subtype = (df['Event Type'] == 'EGV') & df['Event Subtype'].notna()
print("CHECK Event Subtypes for Event Type == EGV: ", 
	df[mask_subtype]['Event Subtype'].unique())
print("CHECK Number of subtype entries for Event Type == EGV: ", 
	mask_subtype.sum())
print("CHECK Number of subtype entries for Event Type == EGV where Glucose Value (mg/dL) EXTRME is nan: ",
	(mask_subtype & df['Glucose Value (mg/dL) EXTREME'].isna()).sum())
print("CHECK Event subtype anywhere unequal to EXTREME: ", 
	not df[mask_subtype & df['Glucose Value (mg/dL) EXTREME'].notna() 
	& (df['Event Subtype'] != df['Glucose Value (mg/dL) EXTREME'])].empty)
df.loc[mask_subtype, 'Glucose Value (mg/dL) EXTREME'] = df.loc[mask_subtype, 'Glucose Value (mg/dL) EXTREME']\
															.fillna(df.loc[mask_subtype, 'Event Subtype']) 
df.loc[mask_subtype, 'Event Subtype'] = np.nan
print("FILLNA Glucose Value mg/dL EXTREME with Event Subtype")

# replace low and high with 40 and 400
df['Glucose Value (mg/dL)'].fillna(df['Glucose Value (mg/dL) EXTREME'].replace({'Low':40., 'High':400.}), inplace=True)
df.drop('Glucose Value (mg/dL) EXTREME', axis=1, inplace=True)
print("REPLACE Low with 40 and High with 400")
print("FILLNA Glucose Value mg/dL with EXTREME 40 (Low) and 400 (High)")

df.sort_values(['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], inplace=True)
df.reset_index(drop=True, inplace=True)

# check for nan rows
rows_nan = (df[['Insulin Value (u)', 'Carb Value (grams)', 'Duration (hh:mm:ss)', 'Glucose Value (mg/dL)']].isna().all(axis=1)
			& (df['Event Type'] != 'Insulin') & (df['Event Type'] != 'Health'))
df.drop(df[rows_nan].index, inplace=True)
print("DROPPED %s nan rows (not event type insulin or health)"%rows_nan.sum())

df['date'] = df['local_timestamp'].dt.date

# drop duplicates rows
df_dupl = df[df.duplicated(keep=False)]
df_dupl.to_csv(path+'dupl/dexcom_clean_dupl.csv')

count_dupl = df_dupl.groupby(['RIDER', 'date'])['local_timestamp', 'Event Type', 'Event Subtype', 'Source Device ID', 'Transmitter ID'].nunique()
count_dupl.to_csv(path+'dupl/glucose_dupl.csv')
print("Duplicate rows count:\n", count_dupl)

print("DROPPED %s duplicate rows"%df.duplicated().sum())
df.drop_duplicates(inplace=True)

# duplicate timestamps (per rider and event type)
cols_dupl_ts = ['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype']
df_dupl_ts = df[df.duplicated(subset=cols_dupl_ts, keep=False)]
df_dupl_ts.to_csv(path+'dupl/dexcom_clean_dupl_ts.csv')

print("CHECK Number of duplicate entries for the same RIDER, Event type, Event Subtype and local timestamp: ", len(df_dupl_ts))
print("Duplicate timestamps for each group of activity type:\n",
	df_dupl_ts.groupby(['Event Type'])['local_timestamp'].nunique())

count_dupl_ts = df_dupl_ts.groupby(['RIDER', 'date'])['local_timestamp', 'Event Type', 'Event Subtype', 'Source Device ID', 'Transmitter ID'].nunique()
count_dupl_ts.to_csv(path+'dupl/glucose_dupl_ts.csv')
print("Duplicate timestamps count:\n", count_dupl_ts)

""" TODO: don't drop duplicate timestamps here until issue with max ts is solved
df.drop_duplicates(subset=['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype'], keep='last', inplace=True)
print("DROPPED %s duplicate timestamps (per rider, event type and event subtype)"%df_dupl_ts.duplicated(subset=cols_dupl_ts).sum())
"""
matplotlib.use('TkAgg')
# plot duplicates
for i, (r, d) in enumerate(count_dupl_ts.index):
	df_dupl_ts_i = df[(df.RIDER == r) & (df.date == d) & (df['Event Type'] == 'EGV')]
	df_dupl_ts_i['min'] = df_dupl_ts_i['local_timestamp'].dt.round('min')
	df_dupl_ts_i['dupl'] = df_dupl_ts_i['min'].duplicated(keep=False).astype(int)

	ax = sns.scatterplot(df_dupl_ts_i['local_timestamp'], df_dupl_ts_i['Glucose Value (mg/dL)'], hue=df_dupl_ts_i['dupl'])
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(r,d))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'dupl/glucose_%s_%s.pdf'%(r,d), bbox_inches='tight')
	plt.savefig(path+'dupl/glucose_%s_%s.png'%(r,d), dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

df.reset_index(drop=True, inplace=True)

# measurements per day per athlete
count_readings = df[df['Event Type'] == 'EGV'].groupby(['RIDER', 'date'])['local_timestamp', 'Event Subtype', 'Source Device ID', 'Transmitter ID'].nunique()

# glucose file with readings more than the max per day
max_readings = 24*60/5 # max number of readings on one day on 5 min interval
count_exceed = count_readings[count_readings['local_timestamp'] > max_readings]
count_exceed.to_csv('exceed/glucose_exceed.csv')
print("Days with more than the max number of readings:\n",
	count_exceed)

# save clean glucose file with max readings exceeded
df_exceed = pd.DataFrame(columns=df.columns)
for r, d in count_exceed.index:
	df_exceed = df_exceed.append(df[(df.RIDER == r) & (df.date == d)])
df_exceed.to_csv(path+'exceed/dexcom_clean_exceed.csv')

for i, (r, d) in enumerate(count_exceed.index):
	df_exceed_i = df_exceed[(df_exceed.RIDER == r) & (df_exceed.date == d) & (df_exceed['Event Type'] == 'EGV')]
	df_exceed_i['min'] = df_exceed_i['local_timestamp'].dt.round('min')
	df_exceed_i['dupl'] = df_exceed_i['min'].duplicated(keep=False).astype(int)

	ax = sns.scatterplot(df_exceed_i['local_timestamp'], df_exceed_i['Glucose Value (mg/dL)'], hue=df_exceed_i['dupl'])
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(r,d))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'exceed/glucose_%s_%s.pdf'%(r,d), bbox_inches='tight')
	plt.savefig(path+'exceed/glucose_%s_%s.png'%(r,d), dpi=300, bbox_inches='tight')
	plt.close()

# filter date range
df = df[(df.local_timestamp.dt.date < datetime.date(2019,12,1)) & (df.local_timestamp.dt.date >= datetime.date(2018,12,1))]
print("DROPPED entries after 30-11-2019 or before 01-12-2018")

df.reset_index(drop=True, inplace=True)
df.drop('date', axis=1, inplace=True)

df.to_csv(path+'dexcom_clean.csv')

# create pandas profiling report
matplotlib.use('Agg')
profile = ProfileReport(df, title='pandas profiling report', minimal=True)
profile.to_file(path+'report_clean.html')