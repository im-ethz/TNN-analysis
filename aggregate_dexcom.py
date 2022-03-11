# TODO: skewness, kurtosis, risk index, FFT with 4 highest amplitudes and their frequencies
# TODO: change time_in_level to perc in level??
import numpy as np
import pandas as pd
import datetime
import os
import gc

from config import DATA_PATH
from calc import glucose_levels, stats_cgm

SAVE_PATH = DATA_PATH+'agg/'

col = 'Glucose Value (mg/dL)'

# ------------- Read data
df = pd.read_csv(DATA_PATH+'Dexcom/clean/dexcom_clean5.csv', index_col=0)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

df['date'] = pd.to_datetime(df.local_timestamp.dt.date)

# calculate dysglycemia events
df['hypo'] = df.groupby('RIDER')[col].transform(lambda x: (x <= glucose_levels['hypo L1'][1]) \
														& (x.shift(1) <= glucose_levels['hypo L1'][1]) \
														& (x.shift(2) <= glucose_levels['hypo L1'][1]))
df['hyper'] = df.groupby('RIDER')[col].transform(lambda x: (x >= glucose_levels['hyper L1'][0]) \
														 & (x.shift(1) >= glucose_levels['hyper L1'][0]) \
														 & (x.shift(2) >= glucose_levels['hyper L1'][0]))

# glucose rate
df['glucose_rate'] = df[col] / (df['timestamp'].diff()/pd.to_timedelta('5min'))

# section day
df['day'] = True

# ------------- aggregate to calculate CGM statistics
# perform some adjustments to the glucose levels, so that floats don't fall in between glucose levels
# e.g. 69.5 falls in between 69 and 70
glucose_levels = {level: (lmin-(1-1e-8), lmax) if level.startswith('hyper') else (
    (lmin, lmax+(1-1e-8)) if level.startswith('hypo') else (
    (lmin, lmax))) for level, (lmin, lmax) in glucose_levels.items()}

# stats for individual sections
sections = ('exercise', 'recovery', 'wake', 'sleep', 'day')

df_sec = pd.concat([df[df[sec]].groupby(['RIDER', 'date']).apply(stats_cgm, sec=sec).apply(pd.Series) for sec in sections], axis=1)
df_sec.to_csv(SAVE_PATH+'dexcom_sec.csv', index_label=False)

df_comp = pd.concat([df[df['race'] & df[sec]].groupby(['RIDER', 'date']).apply(stats_cgm, sec=sec).apply(pd.Series) for sec in sections], axis=1)
df_comp.to_csv(SAVE_PATH+'dexcom_sec_comp.csv', index_label=False)

df_nocomp = pd.concat([df[~df['race'] & df[sec]].groupby(['RIDER', 'date']).apply(stats_cgm, sec=sec).apply(pd.Series) for sec in sections], axis=1)
df_nocomp.to_csv(SAVE_PATH+'dexcom_sec_nocomp.csv', index_label=False)

def select_times(df, w, x):
	"""
	Select data over which we want to apply the stats_cgm, using a time window before midnight.

	Note that this window can extend the group x (limited to one rider and one date), 
	that is why we need to pass the entire dataframe as well to select data.

	Note that we make the selection using timestamp (UTC) instead of local_timestamp. 
	In that way we do not select more data than specified by w, e.g. due to travelling.
	
	Arguments:
		x 		(pd.DataFrame) group of data
		df 		(pd.DataFrame) entire data
		w 		(str) window length
	"""
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