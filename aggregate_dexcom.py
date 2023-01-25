# TODO: skewness, kurtosis, risk index, FFT with 4 highest amplitudes and their frequencies
import numpy as np
import pandas as pd
import datetime
import os
import gc

from config import DATA_PATH
from calc import hypo, hyper, stats_cgm

SAVE_PATH = DATA_PATH+'agg/'

col = 'Glucose Value (mg/dL)'

# ------------- Read data
df = pd.read_csv(DATA_PATH+'source/Dexcom/clean/dexcom_clean5.csv', index_col=0, parse_dates=['timestamp', 'local_timestamp'])
df[col] = df[col].astype(float)

# ------------- Define day to be between 6am and 6am
df['date'] = pd.to_datetime(df.local_timestamp.dt.date)
df['date_6h'] = pd.to_datetime((df.local_timestamp - pd.to_timedelta('6h')).dt.date)

# shift race and travel days
days = df[['RIDER', 'date', 'race', 'travel']].drop_duplicates()
df = pd.merge(df, days, left_on=['RIDER', 'date_6h'], right_on=['RIDER', 'date'], how='left', suffixes=('_raw', ''))
df = df.dropna(subset=['date'])
df['race_day'] = df['race'].fillna(False).astype(bool)
df['travel_day'] = df['travel'].fillna(False).astype(bool)
df = df.drop(['date_raw', 'date_6h', 'race_raw', 'travel_raw', 'race', 'travel'], axis=1)

# include exercise day
df['exercise_day'] = df.groupby(['RIDER', 'date'])['exercise'].transform('any').fillna(False).astype(bool)

# calculate dysglycemia events
df = df.set_index(['RIDER', 'timestamp'])
df['hypo'] = df.groupby(level=0, sort=False).apply(lambda x: hypo(x.reset_index().set_index('timestamp')[col]))
df['hyper'] = df.groupby(level=0, sort=False).apply(lambda x: hyper(x.reset_index().set_index('timestamp')[col]))
df = df.reset_index()

# glucose rate
df['glucose_rate'] = df[col].diff() / (df['timestamp'].diff()/pd.to_timedelta('5min'))

# completeness
df['completeness'] = df.groupby(['RIDER', 'date'])[col].transform('count') / df.groupby(['RIDER', 'date'])['timestamp'].transform('count')

# to csv
df.to_csv(SAVE_PATH+'dexcom.csv', index_label=False)

# ------------- aggregate to calculate CGM statistics
# stats for individual sections
SECTIONS = ('day', 'wake', 'exercise', 'recovery', 'sleep')

# section day
df['day'] = True

# aggregate per day
df_sec = pd.concat([df[df[sec]].groupby(['RIDER', 'date']).apply(stats_cgm, sec=sec).apply(pd.Series) for sec in SECTIONS], axis=1)

df_day = df.groupby(['RIDER', 'date'])[['race_day', 'travel_day', 'exercise_day', 'completeness']].first()
df_sec = pd.merge(df_sec, df_day, left_index=True, right_index=True, how='left')

df_sec.reset_index().to_csv(SAVE_PATH+'dexcom_day.csv', index_label=False)

"""
# NOTE: this part is currently not used for the analysis
def select_times(df, w, x):
	""
	Select data over which we want to apply the stats_cgm, using a time window before midnight.

	Note that this window can extend the group x (limited to one rider and one date), 
	that is why we need to pass the entire dataframe as well to select data.

	Note that we make the selection using timestamp (UTC) instead of local_timestamp. 
	In that way we do not select more data than specified by w, e.g. due to travelling.
	
	Arguments:
		x 		(pd.DataFrame) group of data
		df 		(pd.DataFrame) entire data
		w 		(str) window length
	""
	raise NotImplementedError, "This code contains an error at the moment. Please do not use it."
	# see example rider 4, 2018-12-09 (mostly when travelling happens)
	tz = (x.local_timestamp - x.timestamp).iloc[-1]
	ts_max = x.date.max() + pd.to_timedelta('23:55:00') - tz # get end of day in UTC time
	return df.loc[(df.RIDER == x.RIDER.unique()[0]) & (df.timestamp > ts_max-pd.to_timedelta(w)) & (df.timestamp <= ts_max)]

# stats for individual time windows
windows = ('1h', '3h', '6h', '12h', '18h', '1d', '3d', '7d', '14d')

df_win = pd.concat([df.groupby(['RIDER', 'date'], as_index=False).apply(lambda x: pd.Series(stats_cgm(select_times(df, w, x), sec=w)))\
	.set_index(['RIDER', 'date']) for w in windows], axis=1)
df_win.to_csv(SAVE_PATH+'dexcom_win.csv', index_label=False)
"""