import numpy as np
import pandas as pd
import datetime
import os
import gc

path = 'data/'

# -------------------------- Read data
df = pd.read_csv(path+'dexcom.csv', index_col=0)
df.drop('local_timestamp_raw', axis=1, inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

# select glucose measurements
df = df[df['Event Type'] == 'EGV']
df = df[['RIDER', 'timestamp', 'local_timestamp', 'Source Device ID', 'Glucose Value (mg/dL)',
		 'Transmitter ID', 'Transmitter Time (Long Integer)', 'source']]

# -------------------------- read trainingpeaks info
"""
df_training = {}
for i in df.RIDER.unique():
	print(i)
	df_i = pd.read_csv(path+'trainingpeaks/%s_data.csv'%i)

	df_i['timestamp'] = pd.to_datetime(df_i['timestamp'])
	df_i['local_timestamp'] = pd.to_datetime(df_i['local_timestamp'])

	df_i['date'] = df_i['local_timestamp'].dt.date

	df_agg = df_i.groupby('file_id').agg({'timestamp' : ['min', 'max'], 'local_timestamp': ['min', 'max']})

	df_training[i] = df_agg

	del df_i, df_agg ; gc.collect()

df_training = pd.concat(df_training)
df_training.columns = [c1+'_'+c2 for c1, c2 in df_training.columns]
df_training = df_training.reset_index().rename(columns={'level_0':'RIDER'})
df_training.to_csv('training.csv', index_label=False)
"""
df_training = pd.read_csv(path+'training.csv', index_col=0)
df_training['timestamp_min'] = pd.to_datetime(df_training['timestamp_min'])
df_training['timestamp_max'] = pd.to_datetime(df_training['timestamp_max'])

# -------------------------- identify different parts of the day
df['train'] = False
df['after'] = False
for _, (i, ts_min, ts_max) in df_training[['RIDER', 'timestamp_min', 'timestamp_max']].iterrows():
	# identify during training
	df.loc[(df.RIDER == i) & (df.timestamp >= ts_min) & (df.timestamp <= ts_max), 'train'] = True

	# identify after training
	df.loc[(df.RIDER == i) & (df.timestamp > ts_max) & (df.timestamp <= ts_max + pd.to_timedelta('4h')), 'after'] = True

# TODO: should we exclude after if it's part of a new training session?

df['wake'] = (df.local_timestamp.dt.time >= datetime.time(6)) & (df.local_timestamp.dt.time <= datetime.time(23,59,59))
df['sleep'] = (df.local_timestamp.dt.time < datetime.time(6)) & (df.local_timestamp.dt.time >= datetime.time(0))

del df_training

# -------------------------- identify race and travel days
calendar = pd.read_csv(path+'trainingpeaks_day.csv')
calendar = calendar[['RIDER', 'date', 'race', 'travel']]
calendar['date'] = pd.to_datetime(calendar['date']).dt.date

df['date'] = df.local_timestamp.dt.date

df = pd.merge(df, calendar, how='left', on=['RIDER', 'date'])
df.drop('date', axis=1, inplace=True)

df.to_csv(path+'dexcom_sections.csv', index_label=False)

# -------------------------- resample to remove duplicates from TrainingPeaks
# resample every 5 min
df['timezone'] = df['local_timestamp'] - df['timestamp']
df = df.set_index('timestamp').groupby('RIDER').resample('5min').apply({'timezone'							:'first',
																		'Source Device ID'					:'first',
																		'Glucose Value (mg/dL)'				:'mean',
																		'Transmitter ID'					:'first',
																		'Transmitter Time (Long Integer)'	:'mean',
																		'source'							:'first',
																		'train'								:'first',
																		'after'								:'first',
																		'wake'								:'first',
																		'sleep'								:'first',
																		'race'								:'first',
																		'travel'							:'first'})
df.dropna(subset=['Glucose Value (mg/dL)'], inplace=True)
df.reset_index(inplace=True)
df['local_timestamp'] = df['timestamp'] + df['timezone']
df.drop('timezone', axis=1, inplace=True)
df.to_csv(path+'dexcom_resampled.csv', index_label=False)

# -------------------------- identify days for which completeness is higher than 70%
# USE TIMESTAMP HERE
df['date'] = df.timestamp.dt.date

# TODO: change according to hypex-noc-hypo
# calculate completeness
max_readings = 24*60/5
df_comp = df.groupby(['RIDER', 'date'])['Glucose Value (mg/dL)'].count().to_frame()
df_comp /= max_readings
df_comp['keep'] = df_comp >= 0.7
df_comp.reset_index(inplace=True)
df_comp.rename(columns={'Glucose Value (mg/dL)':'completeness'}, inplace=True)

# select only data with day completeness above 70%
df = pd.merge(df, df_comp, how='left', on=['RIDER', 'date'])
df = df[df['keep']]
df.drop(['date', 'completeness', 'keep'], axis=1, inplace=True)

df.to_csv(path+'dexcom_resampled_selectcompleteness.csv', index_label=False)

# check if the dropping rate is anywhere higher than 1.5 mmol/L/5min = 27 mg/dL/5min
df['glucose_diff'] = df.groupby('RIDER')['Glucose Value (mg/dL)'].transform(lambda x: x.diff())
df['timestamp_diff'] = df.groupby('RIDER')['timestamp'].transform(lambda x: x.diff())
df['glucose_rate'] = df['glucose_diff'] / (df['timestamp_diff'] / pd.to_timedelta('5min'))
(df['glucose_rate'].abs() > 27).sum()