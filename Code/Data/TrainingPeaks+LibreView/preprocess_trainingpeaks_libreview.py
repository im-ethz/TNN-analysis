# List of TODOS - overview
# - Change resampling: for now don't do it, but just calculate summary statistics with the sliding window approach over the last X minutes of training data (with sec frequency)
# - Interpolate historic glucose
# - Add elevation gain as a feature (= diff altitude), ignore ascent feature
# - Calculate HbA1C
# - Does BUBBLE include a 15 min lag from blood glucose to interstitial glucose
# - Find out what happened when there are large gaps in the data
# - Find out what happened when there are small gaps in the data
# - Check GPS accuracy and battery level
# - Possibly remove first entries if the speed = 0
# - Imputation
# - Is there a way that I can have missed data from Libre if there are timestamps missing in TP?
# - IDEA: smooth instead of resampling for more data
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

lv_path = '../LibreView/clean/'
tp_path = '../TrainingPeaks/clean2/'
merge_path = './'

shift_historic = [15, 30, 45, 60]
shift_scan = [3, 5, 10, 15]
shift_bubble = [3, 5, 10, 15]

if not os.path.exists(merge_path+'raw/'):
	os.mkdir(merge_path+'raw/')
if not os.path.exists(merge_path+'1sec/'):
	os.mkdir(merge_path+'1sec/')
if not os.path.exists(merge_path+'resample_1min/'):
	os.mkdir(merge_path+'resample_1min/')
if not os.path.exists(merge_path+'resample_1min/dropna/'):
	os.mkdir(merge_path+'resample_1min/dropna/')

# FOR NOW: ONLY SELECT ATHLETES THAT ARE IN LIBREVIEW

lv_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(lv_path) if i.endswith('.csv')])

# Merge
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

	df_tp.rename(columns={'glucose':'Bubble Glucose mg/dL', 'Zwift':'zwift'}, inplace=True)

	df_tp['timestamp'] = pd.to_datetime(df_tp['timestamp'])
	df_tp['local_timestamp'] = pd.to_datetime(df_tp['local_timestamp'])

	# Remove data before Libre was used
	df_tp = df_tp[df_tp.local_timestamp >= first_date_libre \
		- datetime.timedelta(hours = first_date_libre.time().hour,
							 minutes = first_date_libre.time().minute,
							 seconds=first_date_libre.time().second)]
	del first_date_libre ; gc.collect()

	# -------------------- Merge
	df = pd.merge(df_tp, df_lv, how='left', left_on='local_timestamp', right_on='Device Timestamp')
	df.drop('Device Timestamp', axis=1, inplace=True)

	del df_tp ; gc.collect()

	# -------------------- Shift glucose values
	for s in shift_historic:
		df = shift_glucose(df, df_lv, 'Historic Glucose mg/dL', s)

	for s in shift_scan:
		df = shift_glucose(df, df_lv, 'Scan Glucose mg/dL', s)

	for s in shift_bubble:
		try:
			df_shift = df.set_index('local_timestamp')['Bubble Glucose mg/dL']\
			.shift(periods=-s, freq='min').dropna().reset_index()\
			.rename(columns={'Bubble Glucose mg/dL':'Bubble Glucose mg/dL (shift-%s)'%str(s)})
			df = pd.merge(df, df_shift, how='left', on='local_timestamp')
		except KeyError as e:
			print("KeyError: ", e)
			continue

	df.set_index('local_timestamp', drop=True, inplace=True)

	if not df.empty:
		df.to_csv(merge_path+'raw/'+str(i)+'.csv')

	del df, df_lv; gc.collect()

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(merge_path+'raw/') if i.endswith('.csv')])

# Clean
for i in athletes:
	# -------------------- Clean LibreView (glucose)
	df = pd.read_csv(merge_path+'raw/'+str(i)+'.csv', index_col=0)
	df.index = pd.to_datetime(df.index)

	# -------------------- Backwards-fill + interpolate + shift glucose values
	# backwards fill glucose
	df = fill_glucose(df)
	# shift glucose
	for s in shift_historic:
		df = fill_glucose(df, s)

	PlotPreprocess('../Descriptives/').plot_hist_glucose(df, ['Historic Glucose mg/dL', 'Historic Glucose mg/dL (filled)'])
	# TODO

	# -------------------- Zeros

	# TODO: clean data first for weird zeros at places
	# possibly make extra features without zeros

	# -------------------- Time training
	# check where time training > 1 and there is no change in file_id

	# TODO: find out what happened when there are large gaps in the data


	# -------------------- Temperature -TODO: move to preprocess_trainingpeaks.py
	# smooth temperature (because of binned values) (with centering implemented manually)
	# rolling mean of {temp_window} seconds, centered
	temp_window = 200 #in seconds
	df_temperature_smooth = df['temperature']\
		.rolling('%ss'%temp_window, min_periods=1).mean()\
		.shift(-temp_window/2, freq='s').rename('temperature_smooth')
	df = pd.merge(df, df_temperature_smooth, on='local_timestamp', how='left')
	del df_temperature_smooth ; gc.collect()

	print("Fraction of rows for which difference between original temperature and smoothed temperature is larger than 0.5: ",
		((df['temperature_smooth'] - df['temperature']).abs() > .5).sum() / df.shape[0])

	PlotPreprocess('../Descriptives/').plot_smoothing(df, 'temperature', kwargs=dict(alpha=.5, linewidth=2))

	# note that 200s should probably be more if you look at the distributions
	sns.histplot(df, x='temperature', kde=True)
	sns.histplot(df, x='temperature_smooth', kde=True, color='red')
	plt.show()

	# -------------------- Battery level
	# find out if/when battery level is not monotonically decreasing
	# TODO: why does this happen?
	battery_monotonic = pd.Series()
	for idx in df.file_id.unique():
		battery_monotonic.loc[idx] = (df.loc[df.file_id == idx, 'battery_soc'].dropna().diff().fillna(0) <= 0).all()

	if (~battery_monotonic).sum() > 0:
		print("WARNINGNumber of trainings for which battery level is not monotonically decreasing: ",
			(~battery_monotonic).sum())
		battery_notmonotonic_index = df.file_id.unique()[~battery_monotonic]
	
		# plot battery levels
		cmap = matplotlib.cm.get_cmap('viridis', len(battery_notmonotonic_index))
		ax = plt.subplot()
		for c, idx in enumerate(battery_notmonotonic_index): #for date in df.date.unique():
			df[df.file_id == idx].plot(ax=ax, x='time_training', y='battery_soc',
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
		plt.ylabel('battery_soc')
		plt.show()
		plt.close()

	# linearly interpolate battery level (with only backwards direction)
	for idx in df.file_id.unique():
		df.loc[df.file_id == idx, 'battery_soc_ilin'] = \
		df.loc[df.file_id == idx, 'battery_soc']\
		.interpolate(method='time', limit_direction='forward')

	PlotPreprocess('../Descriptives/').plot_interp(df, 'battery_soc', kwargs=dict(alpha=.5), 
		ikwargs=dict(alpha=.5, kind='scatter', s=10.))

	# -------------------- Distance
	# negative distance diffs (meaning that the athletes is moving backwards)
	print("Negative distance diffs: ", 
		((df.time_training.diff() == 1) & (df['distance'].diff() < 0)).sum())
	# only for athlete 10 there are two negative distance diffs

	# -------------------- Position
	# check if position long and lat are missing at the same time or not 
	print("Number of missing values position_lat: ", df['position_lat'].isna().sum())
	print("Number of missing values position_long: ", df['position_long'].isna().sum())
	print("Number of times both position_lat and position_long missing: ", 
		(df['position_lat'].isna() & df['position_long'].isna()).sum())
	# not really relevant because we're not going to do anything with it anymore

	# -------------------- Speed
	# TODO: create acceleration column with df.speed.diff()
	df['acceleration'] = np.nan
	for f in df.file_id.unique():
		df.loc[df.file_id == f, 'acceleration'] = df.loc[df.file_id == f, 'speed'].diff()

	"""
	def correlation_variables(df_mask1, text1, df_mask2, text2):
		print("Number of ", text1, ": ", df_mask1.sum())
		print("Number of ", text2, ": ", df_mask2.sum())
		print("Number of ", text1, " and ", text2, ": ", (df_mask1 & df_mask2).sum())
		if df_mask1.sum() == df_mask2.sum() and df_mask1.sum() == (df_mask1 & df_mask2).sum():
			print(text1, " happens when ", text2, " and vice versa.")
		if 
			# df_mask2.sum() is not restrictive
		if 

	# speed missing and position missing
	print("Number of missing values speed: ", df['speed'].isna().sum())
	print("Number of missing values speed and position_lat: ",
		(df['speed'].isna() & df['position_lat'].isna()).sum())
	print("Number of missing values speed and missing GPS accuracy: ",
		(df['speed'].isna() & df['gps_accuracy'].isna()).sum())
	print("Number of missing values speed and GPS accuracy one: ",
		(df['speed'].isna() & (df['gps_accuracy'] == 1)).sum())
	# conclusion: if both are equal then when speed is missing, position_lat is missing as well
	
	# speed zero
	print("Number of zero speed: ", (df['speed'] == 0).sum())
	print("Number of zero speed and position_lat: ",
		((df['speed'] == 0) & df['position_lat'].isna()).sum())
	print("Number of zero speed and missing GPS accuracy: ",
		((df['speed'] == 0) & df['gps_accuracy'].isna()).sum())
	print("Number of zero speed and GPS accuracy one: ",
		((df['speed'] == 0) & (df['gps_accuracy'] == 1)).sum())

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

	# -------------------- GPS accuracy
	# gps_accuracy missing and ( position missing or speed missing or speed zero or distance decreasing)
	print("Number of missing values gps_accuracy: ", df['gps_accuracy'].isna().sum())
	print("Number of missing values gps_accuracy and position_lat: ",
		((df['gps_accuracy'].isna()) & (df['position_lat'].isna())).sum())
	print("Number of missing values gps_accuracy and speed: ",
		((df['gps_accuracy'].isna()) & (df['speed'].isna()) & (df.time_training.diff() == 1)).sum())
	print("Number of missing values gps_accuracy and speed zero: ",
		((df['gps_accuracy'].isna()) & (df['speed'] == 0)).sum() & (df.time_training.diff() == 1))
	print("Number of missing values gps_accuracy and decreasing distance: ",
		((df['gps_accuracy'].isna()) & (df['distance'].diff() < 0) & (df.time_training.diff() == 1)).sum())


	# conclusion: all moments when position
	# cannot plot difference in GPS levels because gps accuracy is also nan when the rest is nan

	# plot difference in GPS levels when position is missing

	# plot difference in GPS levels when speed is missing

	print("First entries of file: speed zero and (not) start_timestamps")
	# TODO: check where battery level was 0 or gps accuracy was 0

	# TODO: filter out other devices

	# TODO: move
	# gaps in timestamps in one training (incl. first training)
	print("Number of gaps in timestamps in one training: ", 
	((df.index.to_series().diff().astype('timedelta64[s]').shift(-1) != 1) & ~df.file_id.duplicated()).sum() - 1)
	"""

	# -------------------- Save
	df.to_csv(merge_path+'1sec/'+str(i)+'.csv')


athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(merge_path+'1sec/') if i.endswith('.csv')])

# Resampling
for i in athletes:
	df = pd.read_csv(merge_path+'1sec/'+str(i)+'.csv', index_col=0)
	df.index = pd.to_datetime(df.index)
	# TODO: check if there are more measurements from libreview in one minute. 
	# in that case it's best not to take the mean twice, but mean over all types of measurements

	# get nancols before resampling
	cols_nan = list(set(df.columns) - set(df.dropna(axis=1, how='all').columns))
	print("Columns that are nan and will be dropped: ", cols_nan)
	df.drop(cols_nan, axis=1, inplace=True)

	# -------------------- Resampling
	# feature engineering for selected columns
	cols_feature_eng = ['distance', 'heart_rate', 'cadence', 'speed', 'acceleration', 'power', 
						'ascent', 'altitude', 'grade', 'temperature', 'temperature_smooth',
						'left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 
						'left_torque_effectiveness', 'right_torque_effectiveness']

	# note: we are not using entropy because difficult to calculate
	# other possibilities include: skewness and kurtosis, but don't know if this is relevant
	agg_dict = {}
	for col in cols_feature_eng:
		agg_dict.update({col: [np.sum, np.mean, np.median, np.std, sp.stats.iqr, np.min, np.max]})

	# glucose
	agg_dict.update({'Historic Glucose mg/dL'						: 'mean',
					 'Historic Glucose mg/dL (filled)'				: 'mean',
					 'Scan Glucose mg/dL'							: 'mean',
					 'Bubble Glucose mg/dL'							: 'mean'})
	agg_dict.update({'Historic Glucose mg/dL (shift-%s)'%s 			: 'mean' for s in shift_historic})
	agg_dict.update({'Historic Glucose mg/dL (shift-%s) (filled)'%s : 'mean' for s in shift_historic})
	agg_dict.update({'Scan Glucose mg/dL (shift-%s)'%s 				: 'mean' for s in shift_scan})
	agg_dict.update({'Bubble Glucose mg/dL (shift-%s)'%s 			: 'mean' for s in shift_bubble})

	# no feature engineering for remaining columns
	agg_dict.update({'file_id'							: 'first',
					'time_training'						: 'first', # check
					'position_lat'						: 'first', 
					'position_long'						: 'first',
					'gps_accuracy'						: 'mean',
					'battery_soc'						: 'mean',
					'battery_soc_ilin'					: 'mean',
					'time_from_course'					: 'mean', #??? first?
					'compressed_speed_distance'			: 'mean',
					'combined_pedal_smoothness'			: 'mean',
					'fractional_cadence'				: 'mean',
					'accumulated_power'					: 'mean',
					'cycle_length'						: 'mean',
					'resistance'						: 'mean',
					'calories'							: 'mean', # sum?? TODO: feature eng?
					'zwift'								: 'first',
					'device_ELEMNTBOLT'					: 'first',
					'device_ELEMNTROAM'					: 'first',
					'device_zwift'						: 'first',
					'Basal'								: 'first',
					'Nightscout'						: 'first',
					'Bloodglucose'						: 'mean',
					'CarbonBoard'						: 'sum',
					'InsulinOnBoard'					: 'sum,',
					'device_glucose'					: 'first',
					'device_glucose_serial_number'		: 'first',
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
	
	print("\nColumns that are in df and not included in the aggregation (and that are thus dropped): ",
		*tuple(set(df.columns) - set(agg_dict.keys())))
	print("Columns that are included in the aggregation but not in df (-> remove them from the aggregation list): ",
		*tuple(set(agg_dict.keys()) - set(df.columns)))
	for key in set(agg_dict.keys()) - set(df.columns):
		del agg_dict[key]

	# timestamps that should be in df after resampling to minute
	df_ts = df.index.floor('min').unique()

	# resampling to 1 min
	df = df.resample('1min').agg(agg_dict)

	# keep only timestamps that were in original TP file
	# (due to some weird issue with resampling)
	df = df[df.index.isin(df_ts)]

	df.to_csv(merge_path+'resample_1min/'+str(i)+'.csv')

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(merge_path+'resample_1min/') if i.endswith('.csv')])

# Cleaning after resampling
for i in athletes:
	df = pd.read_csv(merge_path+'resample_1min/'+str(i)+'.csv', header=[0,1], index_col=0)
	df.index = pd.to_datetime(df.index)

	df.rename(columns={'amin':'min', 'amax':'max'}, inplace=True)

	# --------------------- Clean and recalculate some features
	# recalculate time_training
	df[('time_training', 'first')] = np.nan
	for f in df[('file_id', 'first')].unique():
		df.loc[df[('file_id', 'first')] == f, ('time_training', 'first')] = df[df[('file_id', 'first')] == f].index - df[df[('file_id', 'first')] == f].index.min()
	df[('time_training', 'first')] = df[('time_training', 'first')] / np.timedelta64(1,'m')

	# create glucose id
	df = id_glucose(df)
	for s in shift_historic:
		df = id_glucose(df, s)

	# --------------------- Nans
	cols_feature = ['acceleration', 'altitude', 'cadence', 'distance', #'ascent',
					'heart_rate', 'left_pedal_smoothness', 'right_pedal_smoothness',
					'left_torque_effectiveness', 'right_torque_effectiveness', 'left_right_balance',
					'power', 'speed', 'temperature']

	#print("Number of nans \n", df.isna().sum().unstack())
	# TODO: obs: for temperature smooth, there are more nans than for temperature

	#print("Max number of nans per col type: \n", 
	#	df.isna().sum().unstack().max(axis=1).sort_values())

	# save file where all nans are dropped
	# don't consider ascent as it has too many nan values
	df_dropna = df.dropna(how='any', subset=[(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'min', 'max', 'std', 'sum']])
	df_dropna.to_csv(merge_path+'resample_1min/dropna/'+str(i)+'.csv')

	# TODO: drop if nanvalue is first of file
	# TODO: when do we want to delete the entire row? Also if sth like iqr is missing?
	for f in df[('file_id', 'first')]:
		if df.loc[df[('file_id', 'first')] == f].iloc[0][cols_feature].isna().sum() != 0:


	# TODO: find out how large the gaps are. Only impute for small gaps, else remove
	for f in df[('file_id', 'first')]:
		gap_ts = pd.Series(df[df[('file_id', 'first')] == f].index, index=df[df[('file_id', 'first')] == f].index).diff().astype('timedelta64[m]') > 1


	for c in cols_feature:
		for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']:
			for f in df[('file_id', 'first')]:
				df.loc[df[('file_id', 'first')] == f, (c,x)].isna()

	# use linear/spline imputation if there is no seasonality and a trend

	# -------------------- Clean

	# find out where missing data is located: per training
	print("Data per training:\n",
		df.groupby('file_id').count())
	data_completion = df.groupby('file_id').count().div(df.groupby('file_id').count().max(axis=1), axis=0)
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
	data_completion_first = df.loc[start_timestamps_first].groupby('file_id').count()\
		.div(df.loc[start_timestamps_first].groupby('file_id').count().max(axis=1), axis=0)
	ax = sns.heatmap(data_completion_first, cmap="YlGnBu_r", cbar_kws={'label': 'Fraction complete'})
	ax.set_xticks(np.arange(df.shape[1]))
	ax.set_xticklabels(df.columns, rotation='vertical')
	ax.xaxis.set_ticks_position('top')
	plt.title('Data completion in first %s timestamps of each training'%first_x)
	plt.show()
	data_completion_nfirst = df.loc[~df.index.isin(start_timestamps_first)].groupby('file_id').count()\
		.div(df.loc[~df.index.isin(start_timestamps_first)].groupby('file_id').count().max(axis=1), axis=0)
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
	cols_noimpute = set(['date', 'file_id', 'time_training (min)', 
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
		for idx in df.file_id.unique():
			try:
				df.loc[df.file_id == idx, col+'_interp'] = \
				df.loc[df.file_id == idx, col]\
				.interpolate(method=interp_dict[col], limit_direction='forward')
			except ValueError as e:
				if str(e) == "The number of derivatives at boundaries does not match: expected 1, got 0+0"\
					or str(e) == "The number of derivatives at boundaries does not match: expected 2, got 0+0":
					print("Removing %s rows on %s due to too few %s values"\
						%((df.file_id == idx).sum(), 
							np.unique(df[df.file_id == idx].date)[0].strftime("%d-%m-%Y"),col))
					#df = df[df.date != date] # remove all data from this date
					# TODO: fix, do we even want to remove this?? 
					#df = df[df.file_id != idx] # remove all data from this date
					df.loc[df.file_id == idx, col+'_interp'] = \
					df.loc[df.file_id == idx, col]
				else:
					print("Error not caught for %s"%np.unique(df[df.file_id == idx].date)[0].strftime("%d-%m-%Y"))
					break

		# plot features for each training session
		pfirst = 10#len(df.file_id.unique())#200
		cmap = matplotlib.cm.get_cmap('viridis', len(df.file_id.unique()[:pfirst]))
		ax = plt.subplot()
		for c, idx in enumerate(df.file_id.unique()[:pfirst]): #for date in df.date.unique():
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
			df[df.file_id == idx].plot(ax=ax, x='time_training (min)', y=col+'_interp',
				color=cmap(c), legend=False, alpha=.5)
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col,
			df[df.file_id == idx].plot(ax=ax, x='time_training (min)', y=col,
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
		plt.ylabel(col)
		plt.show()
		plt.close()
		
		df[col+'_interp_ONLY'] = df.loc[df[col].isna(), col+'_interp']
		ax = plt.subplot()
		for c, idx in enumerate(df.file_id.unique()[:pfirst]): #for date in df.date.unique():
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
			df[df.file_id == idx].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=15.)
			#df[df.date == date].plot(ax=ax, x='time_training (min)', y=col,
			df[df.file_id == idx].plot(ax=ax, x='time_training (min)', y=col,
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