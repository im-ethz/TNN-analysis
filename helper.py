import numpy as np
import pandas as pd
import datetime

month_mapping = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july',
				 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
month_firstday = {'january':0, 'february':31, 'march':59, 'april':90, 'may':120, 'june':151, 'july':181,
				'august':212, 'september':243, 'october':273, 'november':304, 'december':334}

def create_calendar_array(df, datacol):
	"""
	Create array in the form of calendar (month x days)
	df 		- pandas DataFrame with dates as index (for only one year!)
	"""
	df['month'] = pd.to_datetime(df.index).month
	df['day'] = pd.to_datetime(df.index).day
	df_calendar = df.pivot('month','day', datacol)
	df_calendar.index = pd.CategoricalIndex(df_calendar.index.map(month_mapping), 
		ordered=True, categories=month_mapping.values())
	return df_calendar

def print_times_dates(text, df:pd.DataFrame, df_mask, ts='timestamp'):
	print("\n", text)
	print("times: ", df_mask.sum())
	print("days: ", len(df[df_mask][ts].dt.date.unique()))
	if verbose == 2:
		print("file ids: ", df[df_mask].file_id.unique())

def shift_glucose(df:pd.DataFrame, df_lv:pd.DataFrame, glucose:str, shift:int) -> pd.DataFrame:
	# shift glucose values backwards in time by shift
	df_lv_shift = df_lv.set_index('Device Timestamp').shift(periods=-shift, freq='min').reset_index()\
		[['Device Timestamp', glucose]].rename(columns={glucose: glucose + ' (shift-%s)'%str(shift)})

	df = pd.merge(df, df_lv_shift, how='left', left_on='local_timestamp', right_on='Device Timestamp')
	df.drop('Device Timestamp', axis=1, inplace=True)
	return df

def fill_glucose(df:pd.DataFrame, shift=0) -> pd.DataFrame:
	# backwards-fill historic glucose values 15 min
	# as historic glucose is the average glucose value of the last 15 minutes
	name = 'Historic Glucose mg/dL'
	if shift != 0:
		name += ' (shift-%s)'%shift

	glucose_hist = pd.Series(name=name+' (filled)', dtype=float)
	for t, gl in df[name].dropna().sort_index().iteritems():
		glucose_hist = pd.concat([glucose_hist, pd.Series(data = gl, 
			index = pd.date_range(end=t+datetime.timedelta(seconds=59), periods=15*60, freq='s'), 
			name = name+' (filled)')])
	glucose_hist = glucose_hist[~glucose_hist.index.duplicated()]
	
	return pd.merge(df, glucose_hist, how='left', left_index=True, right_index=True, validate='one_to_one')

def id_glucose(df:pd.DataFrame, shift=0) -> pd.DataFrame:
	# create glucose_id for each 15-minute glucose measurement range of one athlete
	# easy for modeling purposes
	name_gl = 'Historic Glucose mg/dL'
	name_id = 'glucose_id'
	if shift != 0:
		name_gl += ' (shift-%s)'%shift
		name_id += ' (shift-%s)'%shift

	glucose_id = pd.Series(dtype='float64')
	for idx, t in enumerate(df[name_gl].dropna().index):
		glucose_id = pd.concat([glucose_id, pd.Series(data=idx, 
			index=pd.date_range(end=t, periods=15, freq='min'))])
	glucose_id = pd.DataFrame(glucose_id.values, index=glucose_id.index, 
		columns=pd.MultiIndex.from_tuples([(name_id, 'first')]))
	glucose_id = glucose_id[~glucose_id.index.duplicated()]

	return pd.merge(df, glucose_id, how='left', left_index=True, right_index=True, validate='one_to_one')