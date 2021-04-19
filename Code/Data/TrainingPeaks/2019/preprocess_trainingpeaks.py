# TODO: include units in column names (not possible for now, maybe saved units wrong..)
# TODO: remove duplicate timestamps
# TODO: instead of deleting data for which both local timestamps are not incorrect, try finding out which one is the right one (if you overlap it with Libre)
import os
import sys
sys.path.append(os.path.abspath('../../../'))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
from pandas_profiling import ProfileReport

import datetime
import pytz
from tzwhere import tzwhere
tzwhere = tzwhere.tzwhere()

from plot import *
from helper import *

import gc

path = './'
if not os.path.exists(path+'combine/'):
	os.mkdir(path+'combine/')
if not os.path.exists(path+'clean/'):
	os.mkdir(path+'clean/')
if not os.path.exists(path+'clean2/'):
	os.mkdir(path+'clean2/')

verbose = 2

def cleaning_per_file(df_data, df_info, j):
	# convert latitude and longitude from semicircles to degrees
	if 'position_long' in df_data:
		df_data['position_long'] = df_data['position_long'] * (180 / 2**31)
	if 'position_lat' in df_data:
		df_data['position_lat'] = df_data['position_lat'] * (180 / 2**31)

	# get local timestamp
	df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])
	try:
		# calculate timezone
		tz = (pd.to_datetime(df_info.loc['activity'].loc['local_timestamp'])
			- pd.to_datetime(df_info.loc['activity'].loc['timestamp']))[0]
		if abs(tz) > datetime.timedelta(days=1): # sometimes the local_timestamp is someday in 1989, so now I ignored it
			df_data['local_timestamp'] = np.nan
		else:
			df_data['local_timestamp'] = df_data['timestamp'] + tz
	except KeyError:
		df_data['local_timestamp'] = np.nan

	# get local timestamp from location
	if 'position_lat' in df_data and 'position_long' in df_data:
		tz_loc_name = tzwhere.tzNameAt(df_data['position_lat'].dropna().iloc[0], df_data['position_long'].dropna().iloc[0])
		if tz_loc_name is not None:
			tz_loc = pytz.timezone(tz_loc_name).utcoffset(df_data['timestamp'][0])
			df_data['local_timestamp_loc'] = df_data['timestamp'] + tz_loc
	else:
		df_data['local_timestamp_loc'] = np.nan

	# remove local_timestamp_loc if device is zwift
	if ('device_0', 'manufacturer') in df_info.T:
		if df_info.loc[('device_0', 'manufacturer')][0] == 'zwift':
			df_data['local_timestamp_loc'] = np.nan
	elif ('session', 'sub_sport') in df_info.T:
		if df_info.loc[('session', 'sub_sport')][0] == 'virtual_activity':
			df_data['local_timestamp_loc'] = np.nan

	# create column filename
	df_data['file_id'] = j

	return df_data, df_info

def isnan(x):
	return (x != x)

def product_info(x, col0):
	try:
		manufacturer = x[(col0, 'manufacturer')]
	except KeyError:
		manufacturer = np.nan
	try:
		product_name = x[(col0, 'product_name')]
	except KeyError:
		product_name = np.nan
	if isnan(manufacturer) and isnan(product_name):
		return np.nan
	else:
		return str(manufacturer) + ' ' + str(product_name)

def print_times_dates(text, df, df_mask, ts='timestamp'):
	print("\n", text)
	print("times: ", df_mask.sum())
	print("days: ", len(df[df_mask][ts].dt.date.unique()))
	if verbose == 2:
		print("file ids: ", df[df_mask].file_id.unique())

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'csv/')])

# merge all csv files
for i in athletes:
	print("\n------------------------------- Athlete ", i)
	if not os.path.exists(path+'clean/'+str(i)):
		os.mkdir(path+'clean/'+str(i))
	if not os.path.exists(path+'combine/'+str(i)):
		os.mkdir(path+'combine/'+str(i))

	filename_date = {}
	fileid_filename = {}

	df = pd.DataFrame()
	df_nan = pd.DataFrame()
	df_information = pd.DataFrame()

	# -------------------- Normal
	files = sorted(os.listdir(path+'csv/'+str(i)+'/data'))
	for j, f in enumerate(files):
		name = f.rstrip('_data.csv')

		df_data = pd.read_csv(path+'csv/'+str(i)+'/data/'+f, index_col=0)
		df_info = pd.read_csv(path+'csv/'+str(i)+'/info/'+name+'_info.csv', index_col=(0,1))

		df_data, df_info = cleaning_per_file(df_data, df_info, j)

		filename_date.update({f:df_data['timestamp'].dt.date.unique()}) # update filelist
		fileid_filename.update({j:f})

		# create df_information file
		df_info = df_info.T.rename(index={'0':j})
		df_information = df_information.append(df_info, verify_integrity=True)

		# open all other files and check for missing info
		try:
			df_n = pd.read_csv(path+'csv/'+str(i)+'/nan/'+name+'_nan.csv', index_col=0)
			df_nan = df_nan.append(df_n, ignore_index=True, verify_integrity=True)
		except FileNotFoundError:
			pass

		df = df.append(df_data, ignore_index=True, verify_integrity=True)

	# -------------------- Move nan-info from df to df_nan
	# remove unknown columns from into nans file TODO: process later
	cols_unk = df.columns[df.columns.str.startswith('unknown')]
	df_nan = df_nan.append(df[['timestamp']+list(cols_unk)].dropna(subset=cols_unk, how='all'))
	df.drop(cols_unk, axis=1, inplace=True)

	df_nan.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_nan.csv')

	# -------------------- Save mappings
	# save file with mapping of fileid to filename
	pd.DataFrame.from_dict(fileid_filename, orient='index').to_csv(path+'clean/'+str(i)+'/'+str(i)+'_fileid_filename.csv')

	# create dict with dates that refers to filenames
	date_filename = pd.DataFrame.from_dict(filename_date, orient='index').stack().reset_index()\
		.rename(columns={'level_0':'file', 0:'date'}).drop('level_1', axis=1)\
		.groupby('date').apply(lambda x:np.unique(x))
	date_filename = pd.DataFrame(date_filename, columns=['files'])
	date_filename['N'] = date_filename['files'].apply(lambda x:len(x))
	date_filename.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_datefilename.csv')

	# ------------------- Clean df_information	
	# drop columns that only have zeros in them
	for col in df_information.columns:
		if (df_information[col].dropna() == 0).all():
			df_information.drop(col, axis=1, inplace=True)
			if verbose == 2:
				print('DROPPED: ', str(col))

	# identify columns with a lot of nans (outside of "device")
	cols_nan = []
	for col in df_information.columns:
		if df_information[col].isna().sum() / df_information.shape[0] > 0.7 and col[0][:6] != 'device':
			cols_nan.append(col)
	if verbose == 2:
		print("WARNING: Cols outside of 'device' that have a high number of nans, and that we should look at: ", cols_nan)

	# check if device_0 equals file_id device
	cols0_device = [c for c in df_information.columns.get_level_values(0).unique() if c[:6] == 'device']
	
	nan_dev = df_information[('file_id', 'product')].isna() | df_information[('device_0', 'product')].isna()
	print("device_0 manufacturer is file_id manufacturer: ", 
		(df_information[('file_id', 'manufacturer')] == df_information[('device_0', 'manufacturer')]).all())
	print("device_0 product is file_id product: ", 
		(df_information[~nan_dev][('file_id', 'product')] == df_information[~nan_dev][('device_0', 'product')]).all())

	# combine all devices into a list
	df_information[('device_summary', '0')] = df_information.apply(lambda x: product_info(x, 'device_0'), axis=1)
	df_information[('device_summary', '1')] = df_information.apply(lambda x: sorted([product_info(x, col0) for col0 in cols0_device[1:]\
													if not isnan(product_info(x, col0))]), axis=1)
	df_information.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_info.csv')

	df.to_csv(path+'combine/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)

# first stage cleaning
for i in athletes:
	print("\n------------------------------- Athlete ", i)
	df = pd.read_csv(path+'combine/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)
	df_information = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_info.csv', header=[0,1], index_col=0)

	# -------------------- Clean df_data
	print("\n--------------- NAN")
	print("\n-------- Remove rows fully nan")
	cols_ignore = set(['timestamp', 'local_timestamp', 'local_timestamp_loc', 'file_id'])

	# drop completely empty rows (usually first or last rows of a file)
	nanrows = df.shape[0] - df.dropna(how='all', subset=set(df.columns)-cols_ignore).shape[0]
	if nanrows > 0:
		print("DROPPED: %s rows dropped due to being empty"%nanrows)
		nanrows_index = df.index[~df.index.isin(df.dropna(how='all', subset=set(df.columns)-cols_ignore).index)]
		if verbose == 2:
			print("Timestamps before and after nanrows: ")
			for idx in nanrows_index:
				print(df.iloc[df.index.get_loc(idx)-1:df.index.get_loc(idx)+2])
		# Conclusion: seems to be the first (or last) row in a file
		df.drop(nanrows_index, inplace=True)
	else:
		print("GOOD: No rows have to be dropped due to being empty")

	print("\n--------------- DEVICE")
	# include device in df
	df = pd.merge(df, df_information[('device_summary', '0')].str.strip("nan").str.strip().rename('device_0'),
					left_on='file_id', right_index=True, how='left')
	df = pd.merge(df, df_information[('file_id', 'serial_number')].rename('device_0_serialnumber'), 
					left_on='file_id', right_index=True, how='left')

	df.replace({'device_0':{'wahoo_fitness ELEMNT':'ELEMNT',
							'wahoo_fitness ELEMNT BOLT':'ELEMNTBOLT',
							'wahoo_fitness ELEMNT ROAM':'ELEMNTROAM',
							'garmin':'GARMIN',
							'zwift':'ZWIFT',
							'bkool BKOOL Website':'BKOOL',
							'virtualtraining Rouvy':'ROUVY'}}, inplace=True)
	df = pd.concat([df, pd.get_dummies(df['device_0'], prefix='device', dtype=bool)], axis=1)
	print("Devices used:\n", df.groupby('file_id').first().device_0.value_counts())

	# sort by device and timestamp for duplicates later
	df = df.sort_values(by=['device_0', 'device_0_serialnumber', 'local_timestamp'], 
		key=lambda x: x.map({'ELEMNTBOLT':0, 'ELEMENTROAM':1, 'ELEMNT':2, 'ZWIFT':3}))
	df.reset_index(drop=True, inplace=True)
	devices = [x for x in df.columns[df.columns.str.startswith('device_')] if x != 'device_0' and x != 'device_0_serialnumber']

	# select devices to keep: ELEMNT (BOLT/ROAM) and ZWIFT
	# note that often they also use garmin devices, but there is a large difference between the files of garmin and of ELEMNT, 
	# so maybe we should use an entirely different package to parse them
	keep_devices = ['device_ELEMNT', 'device_ELEMNTBOLT', 'device_ELEMNTROAM', 'device_ZWIFT']
	keep_devices = set(keep_devices) & set(devices)
	df['keep_devices'] = df[keep_devices].sum(axis=1) > 0
	
	print("Fraction of data dropped with device selection: ", 
		(~df['keep_devices']).sum()/df.shape[0])
	if 'glucose' in df:
		print("Are there glucose values in the data that will be dropped with the device selection? ",
			not df[~df['keep_devices']]['glucose'].dropna().empty)
	else:
		print("There are no glucose values in the data at all.")

	print("\n--------------- TIMESTAMP")
	# fix local timestamp
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df['local_timestamp_loc'] = pd.to_datetime(df['local_timestamp_loc'])

	# fix zwift timestamps
	print("\n-------- Timestamp: impute zwift nan")
	if 'device_ZWIFT' in df:
		ts_zwift = df.loc[df['device_ZWIFT'] & df['local_timestamp'].isna() & df['local_timestamp_loc'].isna(), 'timestamp'].dt.date.unique()
		print("Get timezone of Zwift files for the following dates: ")
		for ts in ts_zwift:
			print(ts)
			ts_range = [ts-datetime.timedelta(days=2), ts-datetime.timedelta(days=1), ts, ts+datetime.timedelta(days=1), ts+datetime.timedelta(days=2)]
			df_ts_zwift = df[df['timestamp'].dt.date.isin(ts_range)]
			tz_zwift = pd.concat([(df_ts_zwift['local_timestamp'] - df_ts_zwift['timestamp']),
								  (df_ts_zwift['local_timestamp_loc'] - df_ts_zwift['timestamp'])], axis=1)
			tz_meta = pd.DataFrame(tz_zwift[0].unique()).dropna().to_numpy()[0]
			tz_loc = pd.DataFrame(tz_zwift[1].unique()).dropna().to_numpy()[0]
			# make sure no difference in ts_range and we have tz both from meta and loc
			if len(tz_meta) == 1 and len(tz_loc) == 1:
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'local_timestamp_loc'] = \
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'timestamp'] + tz_loc[0]
				print("local_timestamp_loc changed to UTC offset: ", tz_loc[0].astype('timedelta64[h]'))
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'local_timestamp'] = \
				df.loc[df['device_ZWIFT'] & (df.timestamp.dt.date == ts), 'timestamp'] + tz_meta[0]
				print("local_timestamp changed to UTC offset: ", tz_meta[0].astype('timedelta64[h]'))
			else:
				print("local_timestamp_loc not changed because of difference in timezones in the range of two days before until two days after")

	# distribution of timezones
	plt.hist([(df.local_timestamp_loc - df.timestamp).astype('timedelta64[h]').to_numpy(),
			  (df.local_timestamp - df.timestamp).astype('timedelta64[h]').to_numpy()],
			  label=['position', 'metadata'])
	plt.xlabel('UTC offset (hours)')
	plt.legend()
	plt.savefig('../../../Descriptives/hist_timezone_%s.pdf'%i, bbox_inches='tight')
	plt.savefig('../../../Descriptives/hist_timezone_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.close()

	# print nans local_timestamp and local_timestamp_loc
	print("\n-------- Timestamp: nan")
	print_times_dates("local timestamp not available", 
		df, df['local_timestamp'].isna())
	print_times_dates("local timestamp from location not available", 
		df, df['local_timestamp_loc'].isna())
	print_times_dates("local timestamp OR local timestamp from location not available", 
		df, df['local_timestamp'].isna() | df['local_timestamp_loc'].isna())
	print_times_dates("local timestamp AND local timestamp from location not available", 
		df, df['local_timestamp'].isna() & df['local_timestamp_loc'].isna())
	
	#df['drop_nan_timestamp'] = df['local_timestamp'].isna() & df['local_timestamp_loc'].isna()
	df_nan_ts = df[df['local_timestamp'].isna() & df['local_timestamp_loc'].isna()]
	df.drop(df_nan_ts.index, inplace=True) 
	print("DROPPED: %s files with %s nan local timestamps (both meta and loc)"%(len(df_nan_ts.file_id.unique()), len(df_nan_ts)))
	
	print("\n-------- Timestamp error")
	# print how often local_timestamp does not equal local_timestamp_loc
	# Note: this is mainly around the time that the clock is switched from summertime to wintertime. Find out what to do with it!
	nan_llts = df['local_timestamp'].isna() | df['local_timestamp_loc'].isna()
	print_times_dates("local timestamp does not equal local timestamp from location (excl. nans)", 
		df[~nan_llts], df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc'])
	print("Dates for which this happens: ", 
		df[~nan_llts][df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc']].timestamp.dt.date.unique())

	df['error_local_timestamp'] = df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc']
	#df.drop(df[~nan_llts][df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc']].index, inplace=True)
	#print("DROPPED: discrepancy local timestamps")

	print("\n-------- Timestamp local combine")
	# combine both local timestamps
	# keep timestamp from location as the primary timestamp
	df['local_timestamp_loc'].fillna(df['local_timestamp'], inplace=True)
	df.drop('local_timestamp', axis=1, inplace=True)
	df.rename(columns={'local_timestamp_loc':'local_timestamp'}, inplace=True)

	print("\n--------------- DUPLICATES")
	print("\n-------- Entire row duplicate")
	# check if one of the files continues on, or if it's simply a complete file that we drop
	len_dupl = {}
	dupl = df.duplicated(set(df.columns)-set(['file_id']), keep=False)
	for f in df[dupl].file_id.unique():
		len_dupl.update({f:[len(df[df.file_id == f]), 
							len(df[dupl & (df.file_id == f)])]})
	print("CHECK if we remove entire file by removing duplicated entries")
	print("Dupl file_id: [len file, len dupl ts of file]\n",
		len_dupl, "\n", dict(zip(len_dupl, [l==m for key, [l,m] in len_dupl.items()])))

	print("DROPPED: %s duplicate entries"%df.duplicated(set(df.columns)-set(['file_id'])).sum())
	df.drop_duplicates(subset=set(df.columns)-set(['file_id']), keep='first', inplace=True)

	print("\n-------- Timestamp duplicate")
	print_times_dates("duplicated timestamps", 
		df[df['timestamp'].notna()], df[df['timestamp'].notna()]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df, df['local_timestamp'].duplicated(), ts='local_timestamp')

	dupl_timestamp_both = df['local_timestamp'].duplicated(keep=False)
	dupl_timestamp = df['local_timestamp'].duplicated(keep='first')

	# check if dupl timestamps cover entire file, or only part of it
	len_dupl_ts = {}
	for f in df[dupl_timestamp_both].file_id.unique():
		len_dupl_ts[f] = [len(df[df.file_id == f]), 
						  len(df[dupl_timestamp_both & (df.file_id == f)])]
	print("CHECK if we remove entire file by removing duplicated timestamps")
	print("Dupl file_id: [len file, len dupl ts of file]\n",
		len_dupl_ts, "\n", dict(zip(len_dupl_ts, [l==m for key, [l,m] in len_dupl_ts.items()])))

	# check percentage of nan columns for each file with duplicated timestamps
	len_colnan_dupl_ts = {}
	colnan_dupl_ts = {}
	for f in df[dupl_timestamp_both].file_id.unique():
		len_colnan_dupl_ts[f] = df[df.file_id == f].isna().all().sum() / len(df.columns)
		colnan_dupl_ts[f] = df.columns[df[df.file_id == f].isna().all()]
	print("Percentage of missing columns per file\n", len_colnan_dupl_ts)
	print("Missing columns per file\n", colnan_dupl_ts)

	# check if duplicate timestamps are caused by using multiple devices at the same time
	print("Duplicate timestamps in devices:")
	for dev in devices:
		print_times_dates(dev, df, dupl_timestamp_both & df[dev], ts='local_timestamp')

	print("Duplicate timestamps in serialnumbers:")
	for ser in df['device_0_serialnumber'].unique():
		print_times_dates(str(ser), df, dupl_timestamp_both & (df['device_0_serialnumber'] == ser), ts='local_timestamp')

	print("Duplicate timestamps after potentially dropping devices: ")
	for dev in keep_devices:
		print_times_dates(dev, df, dupl_timestamp_both & df[dev], ts='local_timestamp')

	print("Do the duplicate timestamps have the same data?")
	print(df[dupl_timestamp_both & df['keep_devices']].sort_values('local_timestamp'))
	# Answer: sometimes. Maybe find way to select faulty values and remove them
	# For now, keep identifyer in them so that we know we should do something with it
	# Note: what can happen here is that two devices are recording at the same time for some reason
	# but one of the two devices may not contain all the information.

	# TODO: look into dropping duplicate timestamps for which the local timestamp is incorrect
	# maybe sort by error timestamps as well?
	#print("Error timestamps in duplicate timestamps: ",
	#	df[dupl_timestamp_both & df['keep_devices']].error_local_timestamp.sum() != 0)
	#print("DROPPED: duplicate timestamps for which the local timestamp is possibly incorrect")
	# TODO: for each file, check whether it's duplicate has more nans (so for missing files, check percentage of nans)

	# Drop entire file associated with a duplicate timestamp
	# Due to the ordering of the data by device, the data from less desired devices is dropped
	# Copy duplicate timestamps to a separate files, in case we want to use it for imputation
	df_dupl = df[df.file_id.isin(df[dupl_timestamp].file_id.unique())]
	df_dupl.drop(['device_0', 'device_0_serialnumber'], axis=1, inplace=True)
	df_dupl.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_data_dupl.csv', index_label=False)
	df.drop(df_dupl.index, inplace=True)
	print("DROPPED: %s files with %s duplicate timestamps"%(len(df_dupl.file_id.unique()), len(df_dupl)))

	df.drop(['device_0', 'device_0_serialnumber'], axis=1, inplace=True)

	# save df to file
	df.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)

	# create pandas profiling report
	profile = ProfileReport(df, title='pandas profiling report', minimal=True)
	profile.to_file(path+'clean/%s/%s_report.html'%(i,i))

	del df, df_information

# TODO: check if there are training sessions that should be merged
# second stage cleaning
for i in athletes:
	print("\n------------------------------- Athlete ", i)

	if not os.path.exists(path+'clean2/'+str(i)):
		os.mkdir(path+'clean2/'+str(i))

	df = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	df = df.sort_values('local_timestamp')

	print("\n-------- select devices")
	# -------------------- Select devices
	cols_devices = df.columns[df.columns.str.startswith('device_')]
	drop_devices = set(cols_devices) - set(['device_ELEMNT', 'device_ELEMNTBOLT', 'device_ELEMNTROAM', 'device_ZWIFT'])

	print("DROPPED: %s files with %s entries from devices %s"\
		%(len(df[~df['keep_devices']].file_id.unique()), (~df['keep_devices']).sum(), drop_devices))
	df = df[df['keep_devices']]
	df.drop(['keep_devices', *list(drop_devices)], axis=1, inplace=True)

	print("\n-------- cleaning features")
	# -------------------- Empty cols
	empty_cols = ['fractional_cadence', 'time_from_course', 'compressed_speed_distance', 'resistance', 'cycle_length', 'accumulated_power']
	for col in empty_cols:
		if col in df:
			df.drop(col, axis=1, inplace=True)
			print("DROPPED: ", col, " (empty)")

	# -------------------- Left-right balance
	# there are some strings in this column for some reason (e.g. 'mask', 'right')
	df.left_right_balance = pd.to_numeric(df.left_right_balance, errors='coerce')
	print("CLEAN: left-right balance")

	# -------------------- Zeros power meter
	# TODO: figure out what to do with zeros from power meter
	# CHECK: is the previous value for a nan always a zero: answer NO
	# CHECK: is the a zero always followed by a nan: answer more NO
	# Note: sometimes the power is nan if there is a shift in timestamps
	# Note: zero power often happens for negative grades, so maybe it is actually when they stop pedalling
	# TODO: ask how this works in real life
	# TODO: nan seems to happen a lot also on a specific day, remove that file

	# -------------------- Enhanced altitude
	# check if enhanced_altitude equals altitude
	print("CHECK: enhanced altitude does not equal altitude %s times"\
		%((df['enhanced_altitude'] != df['altitude']) & 
		(df['enhanced_altitude'].notna()) & (df['altitude'].notna())).sum())
	print("DROPPED: altitude (equals enhanced_altitude)")
	df.drop('altitude', axis=1, inplace=True)
	df.rename({'enhanced_altitude':'altitude'}, axis=1, inplace=True)

	# -------------------- Enhanced speed
	# check if enhanced_speed equals speed
	print("CHECK: enhanced speed does not equal speed %s times"\
		%((df['enhanced_speed'] != df['speed']) & 
		(df['enhanced_speed'].notna()) & (df['speed'].notna())).sum())
	print("DROPPED: speed (equals enhanced_speed)")
	df.drop('speed', axis=1, inplace=True)
	df.rename({'enhanced_speed':'speed'}, axis=1, inplace=True)

	# -------------------- Elevation gain
	df['elevation_gain'] = df.groupby('file_id')['altitude'].transform(lambda x: x.diff())
	print("CREATED: elevation gain")
	# TODO: remove extreme values

	# -------------------- Acceleration
	df['acceleration'] = df.groupby('file_id')['speed'].transform(lambda x: x.diff())
	print("CREATED: acceleration")
	# TODO: remove extreme values

	# -------------------- Acceleration
	df['accumulated_power'] = df.groupby('file_id')['power'].transform(lambda x: x.sum())
	print("CREATED: accumulated power")

	# -------------------- Time in training
	# create column time in training
	df['time_training'] = np.nan
	for fid in df.file_id.unique():
		df.loc[df.file_id == fid, 'time_training'] = df[df.file_id == fid].local_timestamp - df[df.file_id == fid].local_timestamp.min()
	df['time_training'] = df['time_training'] / np.timedelta64(1,'s')
	print("CREATED: time training")

	# length training
	length_training = df.groupby('file_id').count().max(axis=1) / 60
	PlotPreprocess('../../../Descriptives/', athlete=i).plot_hist(length_training, 'length_training (min)')

	print("Max training length (min): ", length_training.max())
	print("Number of trainings that last shorter than 10 min: ", (length_training <= 10).sum())
	print("Number of trainings that last shorter than 20 min: ", (length_training <= 20).sum())
	del length_training

	# -------------------- Distance
	# negative distance diffs (meaning that the athletes is moving backwards)
	print("CHECK: Negative distance diffs: ", 
		((df.time_training.diff() == 1) & (df['distance'].diff() < 0)).sum())
	# only for athlete 10 there are two negative distance diffs

	df.to_csv(path+'clean2/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)

	# create pandas profiling report
	profile = ProfileReport(df, title='pandas profiling report', minimal=True)
	profile.to_file(path+'clean2/%s/%s_report.html'%(i,i))

	del df

# TODO: filter out training sessions for which a whole column is missing
# TODO: remove training sessions for which there is no distance at all

# TODO: find out what happened when there are large gaps in the data
# TODO: calculate values (statistics) on a training-level
# TODO: elevation_gain, acceleartion and accumulated power now contain extreme values when a large shift in time is made within a training
# imputation
for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(path+'clean2/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	# -------------------- Position
	# check if position long and lat are missing at the same time or not 
	print("Number of missing values position_lat: ", df['position_lat'].isna().sum())
	print("Number of missing values position_long: ", df['position_long'].isna().sum())
	print("Number of times both position_lat and position_long missing: ", 
		(df['position_lat'].isna() & df['position_long'].isna()).sum())
	# not really relevant because we're not going to do anything with it anymore

	"""
	print("\n-------- Remove first rows mostly nan")
	# TODO: move this to the next stage
	# for each file, go over the first rows and check the percentage of nans
	df['percentage_nan'] = df[set(df.columns) - cols_ignore].isna().sum(axis=1) / len(set(df.columns) - cols_ignore)
	count_drop_firstrows = 0
	for f in df.file_id.unique():
		for j, row in df[df.file_id == f].iterrows():
			if row['percentage_nan'] > .75:
				df.drop(j, inplace=True)
				count_drop_firstrows += 1
			else:
				break
	print("DROPPED: {:g} first rows with more than 75\% nans".format(count_drop_firstrows))
	"""

	# -------------------- Impute nans
	# TODO fill up nans with duplicates
	# df_dupl = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_data_dupl.csv', index_col=0)


	# -------------------- Temperature - TODO maybe move this to before
	# smooth temperature (because of binned values) (with centering implemented manually)
	# rolling mean of {temp_window} seconds, centered
	temp_window = 200 #in seconds
	df_temperature_smooth = df.set_index('local_timestamp')['temperature']\
		.rolling('%ss'%temp_window, min_periods=1).mean()\
		.shift(-temp_window/2, freq='s').rename('temperature_smooth')
	df = pd.merge(df, df_temperature_smooth, left_on='local_timestamp', right_index=True, how='left')
	del df_temperature_smooth ; gc.collect()
	print("CREATED: temperature smooth")

	print("Fraction of rows for which difference between original temperature and smoothed temperature is larger than 0.5: ",
		((df['temperature_smooth'] - df['temperature']).abs() > .5).sum() / df.shape[0])

	PlotPreprocess('../Descriptives/', athlete=i).plot_smoothing(df, 'temperature', kwargs=dict(alpha=.5, linewidth=2))

	# note that 200s should probably be more if you look at the distributions
	sns.histplot(df, x='temperature', kde=True)
	sns.histplot(df, x='temperature_smooth', kde=True, color='red')
	plt.show()
	plt.close()

	# -------------------- Battery level
	# find out if/when battery level is not monotonically decreasing
	# TODO: why does this happen?
	battery_monotonic = pd.Series()
	for idx in df.file_id.unique():
		battery_monotonic.loc[idx] = (df.loc[df.file_id == idx, 'battery_soc'].dropna().diff().fillna(0) <= 0).all()

	if (~battery_monotonic).sum() > 0:
		print("WARNING: Number of trainings for which battery level is not monotonically decreasing: ",
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
	print("CREATED: linearly interpolated battery level")

	PlotPreprocess('../Descriptives/', athlete=i).plot_interp(df, 'battery_soc', kwargs=dict(alpha=.5), 
		ikwargs=dict(alpha=.5, kind='scatter', s=10.))

	# TODO: clean more features here, and add some as well

	df.to_csv(path+'clean3/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)