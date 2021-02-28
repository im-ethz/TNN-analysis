# TODO: include units in column names (not possible for now, maybe saved units wrong..)
# TODO: remove duplicate timestamps
# TODO: keep in mind that with filling the glucose values, we still have some duplicate timestamps in there
# TODO: instead of deleting data for which both local timestamps are not incorrect, try finding out which one is the right one (if you overlap it with Libre)
# TODO: there is something wrong with df['duplicate_timestamp']!!! Because it's not actually equal to df.local_timestamp.duplicated()
import os
import sys
sys.path.append(os.path.abspath('../../../'))

import numpy as np
import pandas as pd

import datetime
import pytz
from tzwhere import tzwhere
tzwhere = tzwhere.tzwhere()

from plot import *
from helper import *

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
	try:
		tz_loc = pytz.timezone(tzwhere.tzNameAt(df_data['position_lat'].dropna().iloc[0], df_data['position_long'].dropna().iloc[0])).utcoffset(df_data['timestamp'][0])
		df_data['local_timestamp_loc'] = df_data['timestamp'] + tz_loc
	except KeyError:
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

# Clean df
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
		print("DROPPED: Number of rows dropped due to being empty: ", nanrows)
		nanrows_index = df.index[~df.index.isin(df.dropna(how='all', subset=set(df.columns)-cols_ignore).index)]
		if verbose == 2:
			print("Timestamps before and after nanrows: ")
			for idx in nanrows_index:
				print(df.iloc[df.index.get_loc(idx)-1:df.index.get_loc(idx)+2])
		# Conclusion: seems to be the first (or last) row in a file
		df.drop(nanrows_index, inplace=True)
	else:
		print("GOOD: No rows have to be dropped due to being empty")

	print("\n-------- Remove first rows mostly nan")
	# TODO: move this to the next stage
	# for each file, go over the first rows and check the percentage of nans
	count_drop_firstrows = 0
	for f in df.file_id.unique():
		for j, row in df[df.file_id == f].iterrows():
			if row[set(df.columns) - cols_ignore].isna().sum() / len(set(df.columns) - cols_ignore) > .75:
				df.drop(j, inplace=True)
				count_drop_firstrows += 1
			else:
				break
	print("DROPPED: {:g} first rows with more than 75\% nans".format(count_drop_firstrows))

	print("\n--------------- DEVICE")
	# TODO: can we move this one section down?
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
	df.drop(df[df['local_timestamp'].isna() & df['local_timestamp_loc'].isna()].index, inplace=True)
	print("DROPPED: NaN local timestamps (if both are NaN)")
	
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
	print("CHECK if we remove entire file by removing duplicated entries, or if the file somehow continues on")
	print("Dupl file_id: [len(df for file_id), len(df_dupl for file_id)]\n",
		len_dupl)

	print("Number of duplicated entries: ", df.duplicated(set(df.columns)-set(['file_id'])).sum())
	df.drop_duplicates(subset=set(df.columns)-set(['file_id']), keep='first', inplace=True)
	print("DROPPED: duplicate entries")

	print("\n-------- Timestamp duplicate")
	print_times_dates("duplicated timestamps", 
		df[df['timestamp'].notna()], df[df['timestamp'].notna()]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df, df['local_timestamp'].duplicated(), ts='local_timestamp')

	dupl_timestamp_both = df['local_timestamp'].duplicated(keep=False)
	dupl_timestamp = df['local_timestamp'].duplicated(keep='first')

	len_dupl_ts = {}
	for f in df[dupl_timestamp_both].file_id.unique():
		len_dupl_ts.update({f:[len(df[df.file_id == f]), 
							   len(df[dupl_timestamp_both & (df.file_id == f)])]})
	print("CHECK if we remove entire file by removing duplicated timestamps, or if the file somehow continues on")
	print("Dupl file_id: [len(df for file_id), len(df_dupl for file_id)]\n",
		len_dupl_ts)

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

	# TODO: check if there are training sessions that should be merged

	# TODO: calculate values (statistics) on a training-level

	# save df to file
	df.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)
	del df, df_information

# Second stage cleaning
for i in athletes:
	print("\n------------------------------- Athlete ", i)

	if not os.path.exists(path+'clean2/'+str(i)):
		os.mkdir(path+'clean2/'+str(i))

	df = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	df = df.sort_values('local_timestamp')

	# select device
	df = df[df['keep_devices']]
	df.drop('keep_devices', axis=1, inplace=True)

	devices = df.columns[df.columns.str.startswith('device_')]
	keep_devices = ['device_ELEMNT', 'device_ELEMNTBOLT', 'device_ELEMNTROAM', 'device_ZWIFT']
	drop_devices = set(devices) - set(keep_devices)
	df.drop(drop_devices, axis=1, inplace=True)
	print("DROPPED: all data of devices: ", drop_devices)

	# create column date and time in training
	df['time_training'] = np.nan
	for fid in df.file_id.unique():
		df.loc[df.file_id == fid, 'time_training'] = df[df.file_id == fid].local_timestamp - df[df.file_id == fid].local_timestamp.min()
	df['time_training'] = df['time_training'] / np.timedelta64(1,'s')

	# length training
	length_training = df.groupby('file_id').count().max(axis=1) / 60
	print("Max training length (min):\n", length_training.max())
	length_training.hist(bins=40)
	plt.xlabel('length_training (min)')
	plt.show()

	print("Number of trainings that last shorter than 10 min: ",
		(length_training <= 10).sum())
	print("Number of trainings that last shorter than 20 min: ",
		(length_training <= 20).sum())
	del length_training
	# TODO: look into this at some point

	# check for negative distances
	print("Negative distance diffs: ", 
		((df.time_training.diff() == 1) & (df['distance'].diff() < 0)).sum())

	# ------------------- cleaning features
	# check if enhanced_altitude equals altitude
	eq_altitude = ((df['enhanced_altitude'] != df['altitude']) & 
    	(df['enhanced_altitude'].notna()) & (df['altitude'].notna())).sum()
	if eq_altitude == 0:
		print("GOOD: enhanced_altitude equals altitude")
		df.drop('enhanced_altitude', axis=1, inplace=True)
	else:
		print("WARNING: enhanced_altitude does not equal altitude %s times"%eq_altitude)

	# check if enhanced_speed equals speed
	eq_speed = ((df['enhanced_speed'] != df['speed']) & 
    	(df['enhanced_speed'].notna()) & (df['speed'].notna())).sum()
	if eq_speed == 0:
		print("GOOD: enhanced_speed equals speed")
		df.drop('enhanced_speed', axis=1, inplace=True)
	else:
		print("WARNING: enhanced_speed does not equal altitude %s times"%eq_speed)

	# TODO: clean more features here, and add some as well

	df.to_csv(path+'clean2/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)