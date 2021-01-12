# List of TODOS - overview
# - Can we include other athletes that are not in LibreView? -> No because they don't have the Libre?
# - Can we use the data from the BUBBLE? How do we know that it is data from bubble? I.e. after 7 may and in TP?
# - Does BUBBLE include a 15 min lag from blood glucose to interstitial glucose
# - Why is ascent not the same as altitude (or altitude.diff)?
# - Find out what happened when there are large gaps in the data
# - Find out what happened when there are small gaps in the data
# - Why does the distance decrease sometimes? (Even though it is within the same training)
# - Check GPS accuracy and battery level
# - Possibly remove first entries if the speed = 0
# - Later: check if we can combine data from other devices
# - Feature engineering: include min, max, mean, std, iqr, per minute and per time interval
# - Imputation
# - Model with and without imputation
# - Put data from different athletes in one file
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import os
import gc
import matplotlib

from plot import *
from helper import *

verbose = 1

lv_path = 'Data/LibreView/clean/'
tp_path = 'Data/TrainingPeaks/clean2/'
merge_path = 'Data/TrainingPeaks+LibreView/'

if not os.path.exists(merge_path):
	os.mkdir(merge_path)
if not os.path.exists(merge_path+'raw/'):
	os.mkdir(merge_path+'raw/')
if not os.path.exists(merge_path+'raw_resampled/'):
	os.mkdir(merge_path+'raw_resampled/')
if not os.path.exists(merge_path+'clean/'):
	os.mkdir(merge_path+'clean/')

def print_times_dates(text, df, df_mask, ts='timestamp'):
	print("\n", text)
	print("times: ", df_mask.sum())
	print("days: ", len(df[df_mask][ts].dt.date.unique()))
	if verbose == 2:
		print("file ids: ", df[df_mask].file_id.unique())

# FOR NOW: ONLY SELECT ATHLETES THAT ARE IN LIBREVIEW

lv_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(lv_path) if i.endswith('.csv')])

for i in lv_athletes:
	print("\n------------------------------- Athlete ", i)
	# -------------------- Libreview
	df_lv = pd.read_csv(lv_path+str(i)+'.csv')

	df_lv['Device Timestamp'] = pd.to_datetime(df_lv['Device Timestamp'], dayfirst=True)
	df_lv.sort_values('Device Timestamp', inplace=True)

	df_lv.rename(columns={'Device':'device_glucose', 'Serial Number':'device_glucose_serial_number'}, inplace=True)

	first_date_libre = df_lv['Device Timestamp'].min()

	print("Number of duplicated timestamps LibreView: ", df_lv['Device Timestamp'].duplicated().sum())


	# -------------------- TrainingPeaks
	df_tp = pd.read_csv(tp_path+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df_tp.rename(columns={'glucose':'glucose_BUBBLE', 'Zwift':'zwift'}, inplace=True)

	df_tp['timestamp'] = pd.to_datetime(df_tp['timestamp'])
	df_tp['local_timestamp'] = pd.to_datetime(df_tp['local_timestamp'])

	# Remove data before Libre was used
	df_tp = df_tp[df_tp.local_timestamp >= first_date_libre]


	# -------------------- Merge
	df = pd.merge(df_tp, df_lv, how='left', left_on='local_timestamp', right_on='Device Timestamp')

	df.set_index('local_timestamp', drop=True, inplace=True)
	df.drop('Device Timestamp', axis=1, inplace=True)

	df.to_csv(merge_path+'raw/'+str(i)+'.csv')

	# -------------------- Clean LibreView (glucose)

	# Fill 15 min before Historic Glucose with same glucose value
	# create list with values that we want to fill df with
	glucose_hist_range = pd.Series(name='Historic Glucose mg/dL (filled)')
	for t, gl in df['Historic Glucose mg/dL'].dropna().sort_index().iteritems():
		glucose_hist_range = pd.concat([glucose_hist_range, pd.Series(data=gl, index=pd.date_range(end=t, periods=15*60, freq='s'), name='Historic Glucose mg/dL (filled)')])
	glucose_hist_range = glucose_hist_range[~glucose_hist_range.index.duplicated()]

	df = pd.merge(df, glucose_hist_range, how='left', left_index=True, right_index=True, validate='one_to_one')


	# -------------------- Clean TrainingPeaks (training data)

	# TODO: clean data first for weird zeros at places
	# possibly make extra features without zeros

	# TODO: apply some smoothing to temperature? (because of binned values)
	# smooth temperature (with centering implemented manually)
	temp_window = 200 #in seconds
	df_temperature_smooth = df['temperature']\
		.rolling('%ss'%temp_window, min_periods=1).mean()\
		.shift(-temp_window/2, freq='s').rename('temperature_smooth')
	df = pd.merge(df, df_temperature_smooth, on='timestamp', how='left')
	del df_temperature_smooth ; gc.collect()

	print("Fraction of rows for which difference between original temperature and smoothed temperature is larger than 0.5: ",
		((df['temperature_smooth'] - df['temperature']).abs() > .5).sum() / df.shape[0])

	pfirst = 5#len(df.index_training.unique())#200
	cmap = matplotlib.cm.get_cmap('viridis', len(df.index_training.unique()[:pfirst]))
	ax = plt.subplot()
	for c, idx in enumerate(df.index_training.unique()[:pfirst]): #for date in df.date.unique():
		df[df.index_training == idx].plot(ax=ax, x='time_training', y='temperature_smooth',
			color=cmap(c), legend=False)
		df[df.index_training == idx].plot(ax=ax, x='time_training', y='temperature',
			color=cmap(c), legend=False, alpha=.5, linewidth=2)
	plt.ylabel('temperature')
	plt.show()
	plt.close()

	# find out if/when battery level is not monotonically decreasing
	# TODO: why does this happen?
	battery_monotonic = pd.Series()
	for idx in df.index_training.unique():
		battery_monotonic.loc[idx] = (df.loc[df.index_training == idx, 'battery_soc'].dropna().diff().fillna(0) <= 0).all()

	print("Number of trainings for which battery level is not monotonically decreasing: ",
		(~battery_monotonic).sum())
	battery_notmonotonic_index = df.index_training.unique()[~battery_monotonic]
	
	# plot battery levels
	cmap = matplotlib.cm.get_cmap('viridis', len(battery_notmonotonic_index))
	ax = plt.subplot()
	for c, idx in enumerate(battery_notmonotonic_index): #for date in df.date.unique():
		df[df.index_training == idx].plot(ax=ax, x='time_training', y='battery_soc_ilin',
			color=cmap(c), legend=False, alpha=.5)
		df[df.index_training == idx].plot(ax=ax, x='time_training', y='battery_soc',
			color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
	plt.ylabel('battery_soc')
	plt.show()
	plt.close()

	# linearly interpolate battery level
	for idx in df.index_training.unique():
		df.loc[df.index_training == idx, 'battery_soc_ilin'] = \
		df.loc[df.index_training == idx, 'battery_soc']\
		.interpolate(method='time', limit_direction='forward')

	pfirst = 5#len(df.index_training.unique())#200
	cmap = matplotlib.cm.get_cmap('viridis', len(df.index_training.unique()[:pfirst]))
	ax = plt.subplot()
	for c, idx in enumerate(df.index_training.unique()[:pfirst]): #for date in df.date.unique():
		df[df.index_training == idx].plot(ax=ax, x='time_training', y='battery_soc_ilin',
			color=cmap(c), legend=False, alpha=.5)
		df[df.index_training == idx].plot(ax=ax, x='time_training', y='battery_soc',
			color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
	plt.ylabel('battery_soc')
	plt.show()
	plt.close()

	# TODO: clean GPS
	# only for athlete 10 two negative distance diffs

	# TODO: distance is negative 235 times even though timediff is only 1 min, why??
	print("Negative distance diffs: ", 
		((df.time_training.diff() == 1) & (df['distance'].diff() < 0)).sum())
	# could it be that I make a mistake in identifying new trainings?
	# why is time training so often < 1 and also why is it larger than 1

	# TODO: check if there are timestamps for which there is 0 difference

	# TODO: find out what happened when there are large gaps in the data
	# i.e. is the battery level low? 
	# is the GPS accuracy low?

	# check if position long and lat are missing at the same time or not 
	print("Number of missing values position_lat: ", df['position_lat'].isna().sum())
	print("Number of missing values position_long: ", df['position_long'].isna().sum())
	print("Number of times both position_lat and position_long missing: ", 
		((df['position_lat'].isna()) & (df['position_long'].isna())).sum())

	# speed missing and position missing
	print("Number of missing values speed: ", df['speed'].isna().sum())
	print("Number of missing values speed and position_lat: ",
		(((df['speed'].isna()) & df['position_lat'].isna())).sum())
	# conclusion: if both are equal then when speed is missing, position_lat is missing as well
	
	# gps_accuracy missing and ( position missing or speed missing or speed zero or distance decreasing)
	print("Number of missing values gps_accuracy: ", df['gps_accuracy'].isna().sum())
	print("Number of missing values gps_accuracy and position_lat: ",
		((df['gps_accuracy'].isna()) & (df['position_lat'].isna())).sum())
	print("Number of missing values gps_accuracy and speed: ",
		((df['gps_accuracy'].isna()) & (df['speed'].isna())).sum())
	print("Number of missing values gps_accuracy and speed zero: ",
		((df['gps_accuracy'].isna()) & (df['speed'] == 0)).sum())
	print("Number of missing values gps_accuracy and decreasing distance: ",
		((df['gps_accuracy'].isna()) & (df['distance'].diff() < 0)).sum())


	# conclusion: all moments when position
	# cannot plot difference in GPS levels because gps accuracy is also nan when the rest is nan

	# plot difference in GPS levels when position is missing


	# plot difference in GPS levels when speed is missing



	# speed zero
	print("Is zero speed associated with missing values position?")
	print("Number of zero speed and missing values position_lat: ", 
		((df['position_lat'].isna()) & (df['speed'] == 0)).sum())
	# position missing and speed zero



	# TODO check where speed was zero (remove from file?), remove first entries before having measurements
	# check where speed is zero
	print("Speed zero at %s timestamps"%(df['speed'] == 0).sum())
	print("Speed zero and first timestamp of training: ",
		((df.index.isin(start_timestamps.index)) & (df['speed'] == 0)).sum())
	print("Speed zero and second timestamp of training: ",
		((np.concatenate(([False], df.index.isin(start_timestamps.index)[:-1]))) & (df['speed'] == 0)).sum())
	
	# plot difference in GPS levels when speed is zero, and when 

	# check if speed is zero and position missing


	# TODO: distance is negative 235 times even though timediff is only 1 min, why??
	print("Negative distance diffs: ", 
		((df['diff_timestamp (min)'] <= 1) & (df['distance'].diff() < 0)).sum())
	# could it be that I make a mistake in identifying new trainings?


	print("First entries of file: speed zero and (not) start_timestamps")
	# TODO: check where battery level was 0 or gps accuracy was 0

	# TODO: filter out other devices


	# TODO: move
	# gaps in timestamps in one training (incl. first training)
	print("Number of gaps in timestamps in one training: ", 
	((df.index.to_series().diff().astype('timedelta64[s]').shift(-1) != 1) & ~df.file_id.duplicated()).sum() - 1)



	# -------------------- Resample
	# already get nancols before resampling
	cols_nan = list(set(df.columns) - set(df.dropna(axis=1, how='all').columns))	

	# TODO: check if there are more measurements from libreview in one minute. 
	# in that case it's best not to take the mean twice, but mean over all types of measurements

	def minmax(x):
		return np.max(x) - np.min(x)

	feature_stats = {'mean'		: 'mean',
					 'median'	: np.median,
					 'std'		: 'std',
					 'range'	: minmax,
					 'iqr'		: sp.stats.iqr,
					 'entropy'	: sp.stats.entropy}

	feature_eng_cols = ['distance', 'heart_rate', 'cadence', 'speed', 'power', 
						'altitude', 'grade', 'temperature', 'left_right_balance',
						'left_pedal_smoothness', 'right_pedal_smoothness', 
						'left_torque_effectiveness', 'right_torque_effectiveness']

	agg_dict = {}
	for col in feature_eng_cols:
		agg_dict.update({col+'_'+name : (col, func) for name, func in feature_stats.items()})

	# resample to 1 min
	agg_dict = agg_dict.update({'position_lat'						: 'first', 
								'position_long'						: 'first',
								'gps_accuracy'						: 'mean',
								'battery_soc'						: 'mean',
								'glucose_BUBBLE'						: 'mean',
								'time_from_course'					: 'mean', #??? first?
								'compressed_speed_distance'			: 'mean',
								'cycle_length'						: 'mean',
								'resistance'						: 'mean',
								'calories'							: 'mean', # sum?? TODO: feature eng?
								'ascent'							: 'mean', # sum?? TODO: feature eng?
								'zwift'								: 'first', # check
								'device_glucose'					: 'first',
								'device_glucose_serial_number'		: 'first',
								'Historic Glucose mg/dL'			: 'mean',
								'Scan Glucose mg/dL'				: 'mean',
								'Non-numeric Rapid-Acting Insulin'	: 'sum',
								'Rapid-Acting Insulin (units)'		: 'sum',
								'Non-numeric Food'					: 'sum',
								'Carbohydrates (grams)'				: 'sum',
								'Carbohydrates (servings)'			: 'sum',
								'Non-numeric Long-Acting Insulin'	: 'sum',
								'Long-Acting Insulin Value (units)'	: 'sum',
								'Notes'								: 'first',
								'Strip Glucose mg/dL'				: 'mean',
								'Ketone mmol/L'						: 'sum',
								'Meal Insulin (units)'				: 'sum',
								'Correction Insulin (units)'		: 'sum',
								'User Change Insulin (units)'		: 'sum'})
	
	print("Columns that are not included in the aggregation (and that are thus dropped): ",
		*tuple(set(df.columns) - set(agg_dict.keys())))

	# timestamps that should be in df after resampling to minute
	df_ts = df.index.floor('min').unique()

	# resampling to 1 min
	df = df.resample('1min').agg(agg_dict)

	# keep only timestamps that were in original TP file
	# (due to some weird issue with resampling)
	df = df[df.index.isin(df_ts)]

	# drop columns that are nan
	print("Columns that are nan and will be dropped: ", cols_nan)
	df.drop(cols_nan, axis=1, inplace=True)

	df.to_csv(merge_path+'raw_resampled/'+str(i)+'.csv')

	# -------------------- Clean
	# print duplicate columns
	print("List of duplicated columns: ",
		df.T.duplicated(keep=False))

	"""
	# UPDATE: they're not the same measurements, so don't combine them in this way
	# merge scan glucose and historic glucose
	try:
		df['glucose (mg/dL)'] = df[['Scan Glucose mg/dL', 'Historic Glucose mg/dL', 'Strip Glucose mg/dL']].mean(axis=1)
		df.drop(['Scan Glucose mg/dL', 'Historic Glucose mg/dL', 'Strip Glucose mg/dL'], axis=1, inplace=True)
	except KeyError:
		df['glucose (mg/dL)'] = df[['Scan Glucose mg/dL', 'Historic Glucose mg/dL']].mean(axis=1)
		df.drop(['Scan Glucose mg/dL', 'Historic Glucose mg/dL'], axis=1, inplace=True)
	"""
	df['glucose (mg/dL)'] = df[['Historic Glucose mg/dL']]
	#df.drop(['Scan Glucose mg/dL', 'Historic Glucose mg/dL', 'Strip Glucose mg/dL'], axis=1, inplace=True)

	# compare glucose TP (BUBBLE) and glucose LV
	print("Number of timestamps for which there is glucose data in TP and in LV: ",
		((df['glucose (mg/dL)'].notna()) & (df['glucose_BUBBLE'].notna())).sum())
	print("Overlap TP and LV:\n",
		df[(df['glucose (mg/dL)'].notna()) & (df['glucose_BUBBLE'].notna())][['glucose (mg/dL)', 'glucose_BUBBLE']])
	# TODO: does BUBBLE include a 15 min lag from blood glucose to interstitial glucose
	# for now just take the mean
	# TODO: Figure out what the difference is between LibreView glucose and glucose from BUBBLE

	"""	
	# UPDATE: they're not the same measurements, so don't combine them in this way
	# merge LV glucose and TP glucose (BUBBLE)
	df['glucose (mg/dL)'] = df[['glucose (mg/dL)', 'glucose_BUBBLE']].mean(axis=1)
	df.drop('glucose_BUBBLE', axis=1, inplace=True)
	"""

	"""
	UPDATE: moved upwards
	# create column with training number
	# identify different trainings based on 
	# - difference between timestamps (larger than 1 minute (=timediff)), and
	# - negative distances
	timediff = 1 # TODO: check if the timediff should be longer
	df['diff_timestamp (min)'] = pd.Series(df.index, index=df.index).diff().astype('timedelta64[m]')
	df['index_training'] = np.nan
	index_training_bool = (df['diff_timestamp (min)'] > timediff) & (df['distance'].diff() < 0)
	df.loc[index_training_bool, 'index_training'] = np.arange(index_training_bool.sum())+1
	df.loc[df.index.min(), 'index_training'] = 0
	df.index_training.ffill(inplace=True)
	df.drop('diff_timestamp (min)', axis=1, inplace=True)

	# create column date and time in training
	df['date'] = df.index.date
	start_timestamps = df[~df.index_training.duplicated()]['index_training']
	df['time_training (min)'] = np.nan
	for ts, idx in start_timestamps.iteritems():
		ts_mask = df.index_training == idx
		df.loc[ts_mask, 'time_training (min)'] = (df.index - ts)[ts_mask]
	df['time_training (min)'] = df['time_training (min)'].astype('timedelta64[s]').astype('timedelta64[m]')
	"""

	# find out where missing data is located: per training
	print("Data per training:\n",
		df.groupby('index_training').count())
	data_completion = df.groupby('index_training').count().div(df.groupby('index_training').count().max(axis=1), axis=0)
	print("Data per training as a fraction of training length:\n",
		data_completion)
	ax = sns.heatmap(data_completion, cmap="YlGnBu_r", cbar_kws={'label': 'Fraction complete'})
	ax.set_xticks(np.arange(df.shape[1]))
	ax.set_xticklabels(df.columns, rotation='vertical')
	ax.xaxis.set_ticks_position('top')
	plt.title('Data completion')
	plt.show()

	# find out where missing data is located: per minute in training
	print("Data per minute in training:\n",
		df.groupby('time_training (min)').count())
	# TODO: check if fraction is calculated correctly
	data_completion_min = df.groupby('time_training (min)').count().div(df.groupby('time_training (min)').count().max(axis=1), axis=0)
	print("Data per training as a fraction of training length:\n",
		data_completion_min)
	ax1 = plt.subplot(111)
	sns.heatmap(data_completion_min, ax=ax1, cmap="YlGnBu_r", cbar_kws={'label': 'Fraction complete'})
	ax1.set_xticks(np.arange(df.shape[1]))
	ax1.set_xticklabels(df.columns, rotation='vertical')
	ax1.xaxis.set_ticks_position('top')
	#ax2 = plt.subplot(155, sharey=ax1)
	#df.groupby('time_training (min)').count().max(axis=1).hist(ax=ax2, orientation='horizontal')
	plt.title('Data completion')
	plt.show()
	# Conclusion: delete the following features:
	# - time_from_course
	# - ascent?

	"""
	print("Data completion in the first minutes,")
	first_x = 5
	start_timestamps_first = np.array([df.index[df.index.get_loc(ts):df.index.get_loc(ts)+first_x].values for ts in start_timestamps.index]).flatten()
	data_completion_first = df.loc[start_timestamps_first].groupby('index_training').count()\
		.div(df.loc[start_timestamps_first].groupby('index_training').count().max(axis=1), axis=0)
	ax = sns.heatmap(data_completion_first, cmap="YlGnBu_r", cbar_kws={'label': 'Fraction complete'})
	ax.set_xticks(np.arange(df.shape[1]))
	ax.set_xticklabels(df.columns, rotation='vertical')
	ax.xaxis.set_ticks_position('top')
	plt.title('Data completion in first %s timestamps of each training'%first_x)
	plt.show()
	data_completion_nfirst = df.loc[~df.index.isin(start_timestamps_first)].groupby('index_training').count()\
		.div(df.loc[~df.index.isin(start_timestamps_first)].groupby('index_training').count().max(axis=1), axis=0)
	ax = sns.heatmap(data_completion_nfirst, cmap="YlGnBu_r", cbar_kws={'label': 'Fraction complete'})
	ax.set_xticks(np.arange(df.shape[1]))
	ax.set_xticklabels(df.columns, rotation='vertical')
	ax.xaxis.set_ticks_position('top')
	plt.title('Data completion removing first %s timestamps of each training'%first_x)
	plt.show()
	"""

	# TODO: find out what happened when there are large gaps in the data
	# i.e. is the battery level low? 
	# is the GPS accuracy low?


	# check if position long and lat are missing at the same time or not 
	print("Number of missing values position_lat: ", df['position_lat'].isna().sum())
	print("Number of missing values position_long: ", df['position_long'].isna().sum())
	print("Number of times both position_lat and position_long missing: ", 
		((df['position_lat'].isna()) & (df['position_long'].isna())).sum())

	# speed missing and position missing
	print("Number of missing values speed: ", df['speed'].isna().sum())
	print("Number of missing values speed and position_lat: ",
		(((df['speed'].isna()) & df['position_lat'].isna())).sum())
	# conclusion: if both are equal then when speed is missing, position_lat is missing as well
	
	# gps_accuracy missing and ( position missing or speed missing or speed zero or distance decreasing)
	print("Number of missing values gps_accuracy: ", df['gps_accuracy'].isna().sum())
	print("Number of missing values gps_accuracy and position_lat: ",
		((df['gps_accuracy'].isna()) & (df['position_lat'].isna())).sum())
	print("Number of missing values gps_accuracy and speed: ",
		((df['gps_accuracy'].isna()) & (df['speed'].isna())).sum())
	print("Number of missing values gps_accuracy and speed zero: ",
		((df['gps_accuracy'].isna()) & (df['speed'] == 0)).sum())
	print("Number of missing values gps_accuracy and decreasing distance: ",
		((df['gps_accuracy'].isna()) & (df['distance'].diff() < 0)).sum())


	# conclusion: all moments when position
	# cannot plot difference in GPS levels because gps accuracy is also nan when the rest is nan

	# plot difference in GPS levels when position is missing


	# plot difference in GPS levels when speed is missing



	# speed zero
	print("Is zero speed associated with missing values position?")
	print("Number of zero speed and missing values position_lat: ", 
		((df['position_lat'].isna()) & (df['speed'] == 0)).sum())
	# position missing and speed zero



	# TODO check where speed was zero (remove from file?), remove first entries before having measurements
	# check where speed is zero
	print("Speed zero at %s timestamps"%(df['speed'] == 0).sum())
	print("Speed zero and first timestamp of training: ",
		((df.index.isin(start_timestamps.index)) & (df['speed'] == 0)).sum())
	print("Speed zero and second timestamp of training: ",
		((np.concatenate(([False], df.index.isin(start_timestamps.index)[:-1]))) & (df['speed'] == 0)).sum())
	
	# plot difference in GPS levels when speed is zero, and when 

	# check if speed is zero and position missing


	# TODO: distance is negative 235 times even though timediff is only 1 min, why??
	print("Negative distance diffs: ", 
		((df['diff_timestamp (min)'] <= 1) & (df['distance'].diff() < 0)).sum())
	# could it be that I make a mistake in identifying new trainings?


	print("First entries of file: speed zero and (not) start_timestamps")
	# TODO: check where battery level was 0 or gps accuracy was 0

	# TODO: filter out other devices


	df.to_csv(merge_path+'clean/'+str(i)+'.csv')
	#df = pd.read_csv(merge_path+'clean/'+str(i)+'.csv')
	#df['timestamp'] = pd.to_datetime(df['timestamp'])
	#df.set_index('timestamp', drop=True, inplace=True)

	# -------------------- Feature engineering

	# -------------------- Impute
	# TODO: different type of interpolation?
	# IDEA: don't use interpolation but GP/HMM to find true value of data
	# missing values
	print("Missing values:")
	for col in df.columns:
		print(col, ": ", df[col].isna().sum())

	# features that we should not interpolate
	# because they contain info about when there was missing data,
	# or because they are too sparse (contain too many missing values)
	cols_noimpute = set(['date', 'index_training', 'time_training (min)', 
		'device_glucose', 'device_glucose_serial_number', 
		'time_from_course', 'zwift', 'calories', 'ascent'])

	# TODO: find out how to interpolate digitized signal (for temperature, battery_soc, gps_accuracy)
	interp_dict = {	'position_lat'				: 'time',
					'position_long'				: 'time',
					'gps_accuracy'				: 'time',
					'distance'					: 'time',
					'heart_rate'				: 'akima', #akima? (looks fine)
					'cadence'					: 'akima', #akima?
					'speed'						: 'akima', #akima?
					'power'						: 'akima', #akima?
					'altitude'					: 'time', #slinear?
					'grade'						: 'akima', #akima?
					'battery_soc'				: 'time',
					'left_right_balance'		: 'time', #mean? #TODO: fix zeros
					'left_pedal_smoothness'		: 'time', #mean?
					'right_pedal_smoothness'	: 'time', #mean?
					'left_torque_effectiveness'	: 'time', #mean?
					'right_torque_effectiveness': 'time', #mean?
					'temperature'				: 'time', #TODO
					'glucose (mg/dL)'			: 'pchip'}

	# interpolate features
	# TODO: do it on second basis
	# only within a day!! (probably also works with groupby)
	# now: can only fill forward, and cannot extrapolate (so nans should always be surrounded by valid numbers)
	for col in set(df.columns) - cols_noimpute:
		print(col)
		df[col+'_interp'] = np.nan
		for idx in df.index_training.unique():
			try:
				df.loc[df.index_training == idx, col+'_interp'] = \
				df.loc[df.index_training == idx, col]\
				.interpolate(method=interp_dict[col], limit_direction='forward')
			except ValueError as e:
				if str(e) == "The number of derivatives at boundaries does not match: expected 1, got 0+0"\
					or str(e) == "The number of derivatives at boundaries does not match: expected 2, got 0+0":
					print("Removing %s rows on %s due to too few %s values"\
						%((df.index_training == idx).sum(), 
							np.unique(df[df.index_training == idx].date)[0].strftime("%d-%m-%Y"),col))
					#df = df[df.date != date] # remove all data from this date
					# TODO: fix, do we even want to remove this?? 
					#df = df[df.index_training != idx] # remove all data from this date
					df.loc[df.index_training == idx, col+'_interp'] = \
					df.loc[df.index_training == idx, col]
				else:
					print("Error not caught for %s"%np.unique(df[df.index_training == idx].date)[0].strftime("%d-%m-%Y"))
					break

		# plot features for each training session
		pfirst = 10#len(df.index_training.unique())#200
		cmap = matplotlib.cm.get_cmap('viridis', len(df.index_training.unique()[:pfirst]))
		ax = plt.subplot()
		for c, idx in enumerate(df.index_training.unique()[:pfirst]): #for date in df.date.unique():
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
			df[df.index_training == idx].plot(ax=ax, x='time_training (min)', y=col+'_interp',
				color=cmap(c), legend=False, alpha=.5)
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col,
			df[df.index_training == idx].plot(ax=ax, x='time_training (min)', y=col,
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
		plt.ylabel(col)
		plt.show()
		plt.close()
		
		df[col+'_interp_ONLY'] = df.loc[df[col].isna(), col+'_interp']
		ax = plt.subplot()
		for c, idx in enumerate(df.index_training.unique()[:pfirst]): #for date in df.date.unique():
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
			df[df.index_training == idx].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=15.)
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col,
			df[df.index_training == idx].plot(ax=ax, x='time_training (min)', y=col,
				color=cmap(c), legend=False, alpha=.5)
		plt.ylabel(col)
		plt.show()
		plt.close()
		
	# todo: plot features for each day so that we can see how to impute missing values



"""
# Calculate derivatives
diff_xls = np.array(['@ascent', 'elapseddistance', '@calories'])
for col in diff_xls:
	df_xls[col+'_diff'] = df_xls[col].diff()
"""


"""
	df_lv['glucose_present'] = True
	df_lv_glucose = df_lv[['Device Timestamp', 'glucose_present']]

	# create an array with dates and whether the glucose is present in the file
	df_lv_glucose['date'] = pd.to_datetime(df_lv_glucose['Device Timestamp'], dayfirst=True).dt.date
	df_lv_glucose.drop(['Device Timestamp'], axis=1, inplace=True)
	df_lv_glucose = df_lv_glucose.groupby(['date']).sum()
	df_lv_glucose_measurements = df_lv_glucose.copy()
	df_lv_glucose['glucose_present'] = df_lv_glucose['glucose_present'].astype(bool)

	# plot calendar with glucose availaibility
	df_lv_glucose_calendar = create_calendar_array(df_lv_glucose.copy(), 'glucose_present')
	df_lv_glucose_calendar_measurements = create_calendar_array(df_lv_glucose_measurements.copy(), 'glucose_present')

	binary_mapping_lv = {np.nan:0, True:1}
	df_lv_glucose_calendar = df_lv_glucose_calendar.applymap(lambda x:binary_mapping_lv[x])

	# set calendar items up to first item to nan
	start_lv = df_lv_glucose.index.min()
	for j in range(1,start_lv.day):
		df_lv_glucose_calendar.loc[month_mapping[start_lv.month],j] = np.nan

	# transform dtype to int
	df_lv_glucose_calendar_measurements.fillna(0, inplace=True)
	df_lv_glucose_calendar_measurements = df_lv_glucose_calendar_measurements.astype(int)


	try:
		df_tp_glucose = df_tp[['timestamp', 'glucose']]
	except KeyError:
		continue
	df_tp_glucose.drop_duplicates(ignore_index=True, inplace=True)

	"""
	# make a list similar to the one from libreview with all glucose entries
	df_tp_glucose_list = df_tp_glucose[~df_tp_glucose['glucose'].isna()]
	"""

	df_tp_glucose['glucose_present'] = ~df_tp_glucose['glucose'].isna()
	df_tp_glucose.drop('glucose', axis=1, inplace=True)

	# create an array with dates and whether the glucose is present in the file
	df_tp_glucose['date'] = pd.to_datetime(df_tp_glucose['timestamp']).dt.date
	df_tp_glucose.drop(['timestamp'], axis=1, inplace=True)
	df_tp_glucose = df_tp_glucose.groupby(['date']).sum()
	df_tp_glucose_measurements = df_tp_glucose.copy()
	df_tp_glucose['glucose_present'] = df_tp_glucose['glucose_present'].astype(bool)

	# plot calendar with glucose availaibility
	df_tp_glucose_calendar = create_calendar_array(df_tp_glucose.copy(), 'glucose_present')
	df_tp_glucose_calendar_measurements = create_calendar_array(df_tp_glucose_measurements.copy(), 'glucose_present')

	binary_mapping_tp = {False:-1, np.nan:0, True:1} # False = cycling no glucose ; nan = no cycling ; True = cycling glucose
	df_tp_glucose_calendar = df_tp_glucose_calendar.applymap(lambda x:binary_mapping_tp[x])
"""