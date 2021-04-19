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

# drop duplicates rows
print("DROPPED %s duplicate rows"%df.duplicated().sum())
df.drop_duplicates(inplace=True)

# duplicate timestamps (per rider and event type)
cols_dupl = ['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype']
df_dupl = df[df.duplicated(subset=cols_dupl, keep=False)]

print("CHECK Number of duplicate entries for the same RIDER, Event type, Event Subtype and local timestamp: ", len(df_dupl))
print("Duplicate timestamps for each group of activity type:\n",
	df_dupl.groupby(['RIDER'])['local_timestamp'].nunique())
df_dupl['date'] = df['local_timestamp'].dt.date
print("Dates for which this happens:\n", df_dupl[['RIDER', 'date']].drop_duplicates().set_index('RIDER'))

df.drop_duplicates(subset=['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype'], keep='last', inplace=True)
df.reset_index(drop=True, inplace=True)
print("DROPPED %s duplicate timestamps (per rider, event type and event subtype)"%df_dupl.duplicated(subset=cols_dupl).sum())

# check for nan rows
rows_nan = (df[['Insulin Value (u)', 'Carb Value (grams)', 'Duration (hh:mm:ss)', 'Glucose Value (mg/dL)']].isna().all(axis=1)
			& (df['Event Type'] != 'Insulin') & (df['Event Type'] != 'Health'))
df.drop(df[rows_nan].index, inplace=True)
print("DROPPED %s nan rows (not event type insulin or health)"%rows_nan.sum())

# filter date range
df = df[(df.local_timestamp.dt.date < datetime.date(2019,12,1)) & (df.local_timestamp.dt.date >= datetime.date(2018,12,1))]
print("DROPPED entries after 30-11-2019 or before 01-12-2018")

df.reset_index(drop=True, inplace=True)

df.to_csv(path+'dexcom_clean.csv')

# create pandas profiling report
profile = ProfileReport(df, title='pandas profiling report', minimal=True)
profile.to_file(path+'report_clean.html')