import numpy as np
import pandas as pd
import datetime
import os
import gc

import sys
sys.path.append(os.path.abspath('../'))

from config import DATA_PATH

SAVE_PATH = './'

# -------------------------- Read data
df = pd.read_csv(DATA_PATH+'dexcom.csv', index_col=0)
df.drop('local_timestamp_raw', axis=1, inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

# select glucose measurements
df = df[df['Event Type'] == 'EGV']
df = df[['RIDER', 'timestamp', 'local_timestamp', 'Glucose Value (mg/dL)']]

# -------------------------- read trainingpeaks info
"""
df_training = {}
for i in df.RIDER.unique():
	print(i)
	df_i = pd.read_csv(DATA_PATH+'trainingpeaks/%s_data.csv'%i)

	df_i['timestamp'] = pd.to_datetime(df_i['timestamp'])
	df_i['local_timestamp'] = pd.to_datetime(df_i['local_timestamp'])

	df_i['date'] = df_i['local_timestamp'].dt.date

	df_agg = df_i.groupby('file_id').agg({'timestamp' : ['min', 'max'], 'local_timestamp': ['min', 'max']})

	df_training[i] = df_agg

	del df_i, df_agg ; gc.collect()

df_training = pd.concat(df_training)
df_training.columns = [c1+'_'+c2 for c1, c2 in df_training.columns]
df_training = df_training.reset_index().rename(columns={'level_0':'RIDER'})
df_training.to_csv(SAVE_PATH+'training.csv', index_label=False)
"""
df_training = pd.read_csv(SAVE_PATH+'training.csv', index_col=0)
df_training['timestamp_min'] = pd.to_datetime(df_training['timestamp_min'])
df_training['timestamp_max'] = pd.to_datetime(df_training['timestamp_max'])
df_training['local_timestamp_min'] = pd.to_datetime(df_training['local_timestamp_min'])
df_training['local_timestamp_max'] = pd.to_datetime(df_training['local_timestamp_max'])

# -------------------------- resample to remove duplicates from TrainingPeaks
# resample every 5 min
df['timezone'] = df['local_timestamp'] - df['timestamp']
df = df.set_index('timestamp').groupby('RIDER').resample('5min').apply({'timezone'				:'first',
																		'Glucose Value (mg/dL)'	:'mean'})

# note: also could have used asfreq probably
# ensure there is for every 5 min a timestamp
ts_range = pd.date_range(start='2018-11-30 07:00:00', end='2019-11-30 23:55:00', freq='5min').to_series().rename('timestamp')
ts_index = pd.MultiIndex.from_product([df.index.get_level_values(0).unique(), ts_range], names=['RIDER', 'timestamp'])
df = df.reindex(ts_index)
df.reset_index(inplace=True)

# get local timestamp from resampling
df['timezone'] = df['timezone'].fillna(method='ffill').fillna(method='bfill')
df['local_timestamp'] = df['timestamp'] + df['timezone']
df.drop('timezone', axis=1, inplace=True)

df.to_csv(SAVE_PATH+'dexcom_resampled.csv', index_label=False)

# -------------------------- identify sections
# TODO: should we exclude post if it's part of a new training session?
df['train'] = False
df['post'] = False
for _, (i, ts_min, ts_max) in df_training[['RIDER', 'timestamp_min', 'timestamp_max']].iterrows():
	# identify during training
	df.loc[(df.RIDER == i) & (df.timestamp >= ts_min) & (df.timestamp <= ts_max), 'train'] = True

	# identify post training
	df.loc[(df.RIDER == i) & (df.timestamp > ts_max) & (df.timestamp <= ts_max + pd.to_timedelta('4h')), 'post'] = True

df['wake'] = (df.local_timestamp.dt.time >= datetime.time(6)) & (df.local_timestamp.dt.time <= datetime.time(23,59,59))
df['sleep'] = (df.local_timestamp.dt.time < datetime.time(6)) & (df.local_timestamp.dt.time >= datetime.time(0))

# -------------------------- identify race and travel days
calendar = pd.read_csv(SAVE_PATH+'trainingpeaks_day.csv')
calendar = calendar[['RIDER', 'date', 'race', 'travel']]
calendar['date'] = pd.to_datetime(calendar['date']).dt.date

df['date'] = df.local_timestamp.dt.date

df = pd.merge(df, calendar, how='left', on=['RIDER', 'date'])
df.drop('date', axis=1, inplace=True)

df.to_csv(SAVE_PATH+'dexcom_sections.csv', index_label=False)

# -------------------------- identify days for which completeness is higher than 70%
# USE TIMESTAMP HERE
df['date'] = pd.to_datetime(df.timestamp.dt.date)

# calculate completeness
df_comp = df.groupby(['RIDER', 'date']).apply(lambda x: x['Glucose Value (mg/dL)'].count() / x['timestamp'].count()).rename('completeness')
df = pd.merge(df, df_comp, how='left', on=['RIDER', 'date'])
df.drop('date', axis=1, inplace=True)
df.to_csv(SAVE_PATH+'dexcom_clean.csv', index_label=False)

# -------------------------- remove compression errors
df = pd.read_csv(SAVE_PATH+'dexcom_clean.csv', index_col=0)
df.timestamp = pd.to_datetime(df.timestamp)
df.local_timestamp = pd.to_datetime(df.local_timestamp)

# remove compression lows (i.e. if dropping rate is higher than 1.5 mmol/L/5min = 27 mg/dL/5min)
df['glucose_diff'] = df.groupby('RIDER')['Glucose Value (mg/dL)'].transform(lambda x: x.diff())
df['timestamp_diff'] = df.groupby('RIDER')['timestamp'].transform(lambda x: x.diff())
df['glucose_rate'] = df['glucose_diff'] / (df['timestamp_diff'] / pd.to_timedelta('5min'))
df.loc[df['glucose_rate'] < -27, 'Glucose Value (mg/dL)'] = np.nan
df.loc[df['glucose_rate'] < -27, 'glucose_rate'] = np.nan
df.drop(['glucose_diff', 'timestamp_diff'], axis=1, inplace=True)

df.to_csv(SAVE_PATH+'dexcom_clean_nocomp.csv', index_label=False)