import numpy as np
import pandas as pd
import datetime
import os
import gc
import matplotlib

from plot import *
from helper import *

lv_path = 'Data/LibreView/clean/'
tp_path = 'Data/TrainingPeaks/clean/'
merge_path = 'Data/TrainingPeaks+LibreView/'

if not os.path.exists(merge_path):
	os.mkdir(merge_path)
if not os.path.exists(merge_path+'raw/'):
	os.mkdir(merge_path+'raw/')
if not os.path.exists(merge_path+'raw_resampled/'):
	os.mkdir(merge_path+'raw_resampled/')
if not os.path.exists(merge_path+'clean/'):
	os.mkdir(merge_path+'clean/')

# FOR NOW: ONLY SELECT ATHLETES THAT ARE IN LIBREVIEW

lv_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(lv_path) if i.endswith('.csv')])

for i in lv_athletes:
	print(i)
	# -------------------- Libreview
	df_lv = pd.read_csv(lv_path+str(i)+'.csv')

	df_lv['Device Timestamp'] = pd.to_datetime(df_lv['Device Timestamp'], dayfirst=True)
	df_lv.sort_values('Device Timestamp', inplace=True)

	df_lv.rename(columns={'Device':'device_glucose', 'Serial Number':'device_glucose_serial_number'}, inplace=True)

	first_date_libre = df_lv['Device Timestamp'].min()

	# -------------------- TrainingPeaks
	df_tp = pd.read_csv(tp_path+str(i)+'/'+str(i)+'_data.csv', index_col=0)
	df_tp.drop('Unnamed: 0', axis=1, inplace=True)

	df_tp.rename(columns={'glucose':'glucose_tp', 'Zwift':'zwift'}, inplace=True)

	df_tp['timestamp'] = pd.to_datetime(df_tp['timestamp'])
	df_tp['local_timestamp'] = pd.to_datetime(df_tp['local_timestamp'])

	df_tp = df_tp.sort_values('timestamp')

	# Remove data before Libre was used
	df_tp = df_tp[df_tp.timestamp >= first_date_libre]

	# -------------------- Merge
	## CHECK IF THIS SHOULD BE LOCAL TIMESTAMP
	"""
	dates_tp_glucose = df_tp.timestamp[df_tp.glucose_tp.notna()].dt.date.unique()

	for date in dates_tp_glucose:
		df_tp_example = df_tp[df_tp.timestamp.dt.date == date][['timestamp','local_timestamp', 'glucose_tp']]
		df_lv_example = df_lv[df_lv.timestamp.dt.date == date][['Device Timestamp','Historic Glucose mg/dL', 'Scan Glucose mg/dL', 'Strip Glucose mg/dL']]
		df_example = df_merge[df_merge.index.date == date][['local_timestamp', 'glucose_tp', 'Historic Glucose mg/dL', 'Scan Glucose mg/dL', 'Strip Glucose mg/dL']]

		plt.scatter(df_tp_example['timestamp'], df_tp_example['glucose_tp'], label='tp')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Historic Glucose mg/dL'], label='lv_hist')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Scan Glucose mg/dL'], label='lv_scan')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Strip Glucose mg/dL'], label='lv_strip')
		plt.legend()
		plt.show()

		plt.scatter(df_tp_example['local_timestamp'], df_tp_example['glucose_tp'], label='tp')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Historic Glucose mg/dL'], label='lv_hist')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Scan Glucose mg/dL'], label='lv_scan')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Strip Glucose mg/dL'], label='lv_strip')
		plt.legend()
		plt.show()
	"""
	df_merge = pd.merge(df_tp, df_lv, how='left', left_on='local_timestamp', right_on='Device Timestamp')

	df_merge.set_index('timestamp', drop=True, inplace=True)

	df_merge.to_csv(merge_path+'raw/'+str(i)+'.csv')


	# -------------------- Resample

	# already get nancols before resampling
	cols_nan = list(set(df_merge.columns) - set(df_merge.dropna(axis=1, how='all').columns))	

	# clean data first for weird zeros at places
	# possibly make extra features without zeros

	# rolling before resampling?

	# todo: also check min, max, iqr, and std(var)
	# TODO: check if there are more measurements from libreview in one minute. 
	# in that case it's best not to take the mean twice, but mean over all types of measurements

	# resample to 1 min
	agg_dict = {'position_lat'						: 'first', 
				'position_long'						: 'first',
				'gps_accuracy'						: 'mean',
				'distance'							: 'mean', #?? change to last?
				'heart_rate'						: 'mean',
				'cadence'							: 'mean',
				'enhanced_speed'					: 'mean',
				'speed'								: 'mean',
				'power'								: 'mean',
				'enhanced_altitude'					: 'mean',
				'altitude'							: 'mean',
				'grade'								: 'mean',
				'battery_soc'						: 'mean',
				'left_right_balance'				: 'mean',
				'left_pedal_smoothness'				: 'mean',
				'right_pedal_smoothness'			: 'mean',
				'left_torque_effectiveness'			: 'mean',
				'right_torque_effectiveness'		: 'mean',
				'temperature'						: 'mean',
				#'local_timestamp'					: 'first',
				'glucose_tp'						: 'mean',
				'time_from_course'					: 'mean', #??? first?
				'compressed_speed_distance'			: 'mean',
				'cycle_length'						: 'mean',
				'resistance'						: 'mean',
				'calories'							: 'mean', # sum??
				'ascent'							: 'mean', # sum??
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
				'User Change Insulin (units)'		: 'sum'}
	
	print("Columns that are not included in the aggregation (and that are thus dropped): ",
		*tuple(set(df_merge.columns) - set(agg_dict.keys())))

	# timestamps that should be in df_merge after resampling to minute
	df_merge_ts = df_merge.index.floor('min').unique()

	# resampling to 1 min
	df_merge = df_merge.resample('1min').agg(agg_dict)

	# keep only timestamps that were in original TP file
	# (due to some weird issue with resampling)
	df_merge = df_merge[df_merge.index.isin(df_merge_ts)]

	# drop columns that are nan
	print("Columns that are nan and will be dropped: ", cols_nan)
	df_merge.drop(cols_nan, axis=1, inplace=True)

	df_merge.to_csv(merge_path+'raw_resampled/'+str(i)+'.csv')

	# -------------------- Clean
	# merge scan glucose and historic glucose
	try:
		df_merge['glucose (mg/dL)'] = df_merge[['Scan Glucose mg/dL', 'Historic Glucose mg/dL', 'Strip Glucose mg/dL']].mean(axis=1)
		df_merge.drop(['Scan Glucose mg/dL', 'Historic Glucose mg/dL', 'Strip Glucose mg/dL'], axis=1, inplace=True)
	except KeyError:
		df_merge['glucose (mg/dL)'] = df_merge[['Scan Glucose mg/dL', 'Historic Glucose mg/dL']].mean(axis=1)
		df_merge.drop(['Scan Glucose mg/dL', 'Historic Glucose mg/dL'], axis=1, inplace=True)

	# merge glucose_tp and glucose
	print("Number of timestamps for which there is glucose data in TP and in LV: ",
		((df_merge['glucose (mg/dL)'].notna()) & (df_merge['glucose_tp'].notna())).sum())
	print("Overlap TP and LV:\n",
		df_merge[(df_merge['glucose (mg/dL)'].notna()) & (df_merge['glucose_tp'].notna())][['glucose (mg/dL)', 'glucose_tp']])
	# TODO: does BUBBLE include a 15 min lag from blood glucose to interstitial glucose
	# for now just take the mean
	# TODO: Figure out what the difference is between LibreView glucose and glucose from BUBBLE
	# merge LV (BUBBLE) glucose and TP glucose
	df_merge['glucose (mg/dL)'] = df_merge[['glucose (mg/dL)', 'glucose_tp']].mean(axis=1)
	df_merge.drop('glucose_tp', axis=1, inplace=True)

	# print duplicate columns
	df_merge.T.duplicated(keep=False)

	# check if enhanced_altitude equals altitude
	eq_altitude = ((df_merge['enhanced_altitude'] != df_merge['altitude']) & 
    	(df_merge['enhanced_altitude'].notna()) & (df_merge['altitude'].notna())).sum()
	if eq_altitude == 0:
		print("GOOD: enhanced_altitude equals altitude")
		df_merge.drop('enhanced_altitude', axis=1, inplace=True)
	else:
		print("WARNING: enhanced_altitude does not equal altitude %s times"%eq_altitude)

	# check if enhanced_speed equals speed
	eq_speed = ((df_merge['enhanced_speed'] != df_merge['speed']) & 
    	(df_merge['enhanced_speed'].notna()) & (df_merge['speed'].notna())).sum()
	if eq_speed == 0:
		print("GOOD: enhanced_speed equals speed")
		df_merge.drop('enhanced_speed', axis=1, inplace=True)
	else:
		print("WARNING: enhanced_speed does not equal altitude %s times"%eq_speed)

	# TODO check if altitude is the same as ascent
	# cross-correlation
	[df_merge.altitude.corr(df_merge.ascent.shift(lag)) for lag in np.arange(-10,10,1)]
	[df_merge.altitude.diff().corr(df_merge.ascent.shift(lag)) for lag in np.arange(-10,10,1)]
	[df_merge.altitude.corr(df_merge.ascent.diff().shift(lag)) for lag in np.arange(-10,10,1)]
	# answer: no??

	# TODO: use distance to create new training
	# create column with training number
	# identify based on difference between timestamps (larger than 1 minute)
	timediff = 1 # TODO: check if the timediff should be longer
	df_merge['diff_timestamp (min)'] = pd.Series(df_merge.index, index=df_merge.index).diff().astype('timedelta64[m]')
	df_merge['diff_distance'] = df_merge['distance'].diff()
	df_merge['index_training'] = np.nan
	index_training_bool = (df_merge['diff_timestamp (min)'] > timediff) & (df_merge['diff_distance'] < 0)
	df_merge.loc[index_training_bool, 'index_training'] = np.arange(index_training_bool.sum())+1
	df_merge.loc[df_merge.index.min(), 'index_training'] = 0
	df_merge.index_training.ffill(inplace=True)
	df_merge.drop('diff_timestamp (min)', axis=1, inplace=True)

	# create column date and time in training
	df_merge['date'] = df_merge.index.date
	start_timestamps = df_merge[~df_merge.index_training.duplicated()]['index_training']
	df_merge['time_training (min)'] = np.nan
	for ts, idx in start_timestamps.iteritems():
		ts_mask = df_merge.index_training == idx
		df_merge.loc[ts_mask, 'time_training (min)'] = (df_merge.index - ts)[ts_mask]
	df_merge['time_training (min)'] = df_merge['time_training (min)'].astype('timedelta64[s]').astype('timedelta64[m]')

	# TODO: distance is negative 235 times even though timediff is only 1 min, why??


	# TODO check where speed was zero (remove from file?), remove first entries before having measurements

	# TODO: check where battery level was 0 or gps accuracy was 0

	# TODO: filter out other devices


	df_merge.to_csv(merge_path+'clean/'+str(i)+'.csv')
	#df_merge = pd.read_csv(merge_path+'clean/'+str(i)+'.csv')
	#df_merge['timestamp'] = pd.to_datetime(df_merge['timestamp'])
	#df_merge.set_index('timestamp', drop=True, inplace=True)

	# -------------------- Feature engineering

	# -------------------- Impute
	# TODO: different type of interpolation?
	# IDEA: don't use interpolation but GP/HMM to find true value of data
	# missing values
	print("Missing values:")
	for col in df_merge.columns:
		print(col, ": ", df_merge[col].isna().sum())

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
	for col in set(df_merge.columns) - cols_noimpute:
		print(col)
		df_merge[col+'_interp'] = np.nan
		for idx in df_merge.index_training.unique():
			try:
				df_merge.loc[df_merge.index_training == idx, col+'_interp'] = \
				df_merge.loc[df_merge.index_training == idx, col]\
				.interpolate(method=interp_dict[col], limit_direction='forward')
			except ValueError as e:
				if str(e) == "The number of derivatives at boundaries does not match: expected 1, got 0+0"\
					or str(e) == "The number of derivatives at boundaries does not match: expected 2, got 0+0":
					print("Removing %s rows on %s due to too few %s values"\
						%((df_merge.index_training == idx).sum(), 
							np.unique(df_merge[df_merge.index_training == idx].date)[0].strftime("%d-%m-%Y"),col))
					#df_merge = df_merge[df_merge.date != date] # remove all data from this date
					# TODO: fix, do we even want to remove this?? 
					#df_merge = df_merge[df_merge.index_training != idx] # remove all data from this date
					df_merge.loc[df_merge.index_training == idx, col+'_interp'] = \
					df_merge.loc[df_merge.index_training == idx, col]
				else:
					print("Error not caught for %s"%np.unique(df_merge[df_merge.index_training == idx].date)[0].strftime("%d-%m-%Y"))
					break

		# plot features for each training session
		pfirst = 10#len(df_merge.index_training.unique())#200
		cmap = matplotlib.cm.get_cmap('viridis', len(df_merge.index_training.unique()[:pfirst]))
		ax = plt.subplot()
		for c, idx in enumerate(df_merge.index_training.unique()[:pfirst]): #for date in df_merge.date.unique():
			#df_merge[df_merge.date == date].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
			df_merge[df_merge.index_training == idx].plot(ax=ax, x='time_training (min)', y=col+'_interp',
				color=cmap(c), legend=False, alpha=.5)
			#df_merge[df_merge.date == date].plot(ax=ax, x='time_training (min)', y=col,
			df_merge[df_merge.index_training == idx].plot(ax=ax, x='time_training (min)', y=col,
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=10.)
		plt.ylabel(col)
		plt.show()
		plt.close()
		
		df_merge[col+'_interp_ONLY'] = df_merge.loc[df_merge[col].isna(), col+'_interp']
		ax = plt.subplot()
		for c, idx in enumerate(df_merge.index_training.unique()[:pfirst]): #for date in df_merge.date.unique():
			#df_merge[df_merge.date == date].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
			df_merge[df_merge.index_training == idx].plot(ax=ax, x='time_training (min)', y=col+'_interp_ONLY',
				color=cmap(c), legend=False, alpha=.5, kind='scatter', s=15.)
			#df_merge[df_merge.date == date].plot(ax=ax, x='time_training (min)', y=col,
			df_merge[df_merge.index_training == idx].plot(ax=ax, x='time_training (min)', y=col,
				color=cmap(c), legend=False, alpha=.5)
		plt.ylabel(col)
		plt.show()
		plt.close()
		
	# todo: plot features for each day so that we can see how to impute missing values

"""
.interpolate(method='', limit_direction='forward', axis=1)

# Impute glucose data
# not working: 'nearest', 'zero', 'barycentric', 'krogh'
interp_list = ('time', 'slinear', 'quadratic', 'cubic', 'spline', 'polynomial', 'pchip', 'akima')
for i in interp_list:
	df_xls['glucose_'+i] = df_xls['@glucose'].interpolate(method=i, order=2)

p.plot_interp_allinone(df_xls, 'glucose', interp_list, 'xls')
p.plot_interp_subplots(df_xls, 'glucose', interp_list, (2,4), 'xls')
for i in range(2):
	for j in range(4):
		p.plot_interp_individual(df_xls, 'glucose', interp_list[4*i+j])
# conclusion:
# not good: linear, slinear
# medium: pchip, akima, 2nd order spline
# good: quadratic, cubic, 2nd order polynomial
df_xls.drop(['glucose_' + i for i in (set(interp_list) - {'polynomial'})], axis=1, inplace=True)
df_xls.rename(columns={'glucose_polynomial':'glucose'}, inplace=True)
"""

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