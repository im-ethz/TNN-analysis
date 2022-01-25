import numpy as np
import pandas as pd
import datetime
import os
import gc

from config import DATA_PATH
from calc import glucose_levels, time_in_level, LBGI, HBGI

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

# ------------- aggregate to calculate CGM statistics
# TODO: skewness, kurtosis, risk index, FFT with 4 highest amplitudes and their frequencies

sections = ('exercise', 'recovery', 'wake', 'sleep')
windows = ('1h', '3h', '6h', '12h', '18', '1d', '3d', '7d')

# TODO: change time_in_level to perc in level??
def stat_func(x, sec=''):
	return {'time_in_hypo_'+sec 	: time_in_level(x[col], 'hypo'),
			'time_in_hypoL2_'+sec 	: time_in_level(x[col], 'hypo L2'),
			'time_in_hypoL1_'+sec 	: time_in_level(x[col], 'hypo L1'),
			'time_in_target_'+sec 	: time_in_level(x[col], 'target'),
			'time_in_hyper_'+sec 	: time_in_level(x[col], 'hyper'),
			'time_in_hyperL1_'+sec 	: time_in_level(x[col], 'hyper L1'),
			'time_in_hyperL2_'+sec 	: time_in_level(x[col], 'hyper L2'),
			'glucose_mean_'+sec 	: x[col].mean(),
			'glucose_std_'+sec 		: x[col].std(),
			'glucose_cv_'+sec 		: x[col].std() / x[col].mean(),
			'glucose_rate_'+sec		: x['glucose_rate'].mean(),
			'completeness_'+sec 	: x[col].count() / x['timestamp'].count(),
			'LBGI_'+sec 			: LBGI(x[col]),
			'HBGI_'+sec 			: HBGI(x[col]),
			'AUC_'+sec 				: np.trapz(y=x[col], x=x['timestamp']) / np.timedelta64(5, 'm'),
			'hypo_'+sec 			: x['hypo'].any(),
			'hyper_'+sec 			: x['hyper'].any()}

def select_times(df, w, x):
	"""
	Select data over which we want to apply the stat_func, using a time window before midnight.

	Note that this window can extend the group x (limited to one rider and one date), 
	that is why we need to pass the entire dataframe as well to select data.

	Note that we make the selection using timestamp (UTC) instead of local_timestamp. 
	In that way we do not select more data than specified by w, e.g. due to travelling.
	
	Arguments:
		x 		(pd.DataFrame) group of data
		df 		(pd.DataFrame) entire data
		w 		(str) window length
	"""
	return df.loc[(df.RIDER == x.RIDER.unique()[0]) & (df.timestamp > x.timestamp.max()-pd.to_timedelta(w)) & (df.timestamp <= x.timestamp.max())]

# stats for individual sections
df_sec = pd.concat([df[df[sec]].groupby(['RIDER', 'date']).apply(stat_func, sec=sec).apply(pd.Series) for sec in sections], axis=1)

# stats for individual time windows
df_win = pd.concat([df.groupby(['RIDER', 'date'], as_index=False).apply(lambda x: pd.Series(stat_func(select_times(df, w, x), sec=w)))\
	.set_index(['RIDER', 'date']) for w in windows], axis=1)

df_agg = pd.concat([df_sec, df_win], axis=1).reset_index()
df_agg.to_csv(SAVE_PATH+'dexcom_day.csv', index_label=False)