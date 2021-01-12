# TODO: include units in column names (not possible for now, maybe saved units wrong..)
# TODO: remove duplicate timestamps
# TODO: keep in mind that with filling the glucose values, we still have some duplicate timestamps in there
# TODO: instead of deleting data for which both local timestamps are not incorrect, try finding out which one is the right one (if you overlap it with Libre)
import numpy as np
import pandas as pd

import datetime
import pytz
from tzwhere import tzwhere
tzwhere = tzwhere.tzwhere()

import os

from plot import *
from helper import *

path = 'Data/TrainingPeaks/'
if not os.path.exists(path+'clean/'):
	os.mkdir(path+'clean/')
if not os.path.exists(path+'combine/'):
	os.mkdir(path+'combine/')

verbose = 1

def cleaning_per_file(df_data, df_info, j):
	# drop first col
	df_data.drop('Unnamed: 0', axis=1, inplace=True)

	# convert latitude and longitude from semicircles to degrees
	try:
		df_data['position_long'] = df_data['position_long'] * (180 / 2**31)
		df_data['position_lat'] = df_data['position_lat'] * (180 / 2**31)
	except KeyError:
		pass

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
	print(i)
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

		df_data = pd.read_csv(path+'csv/'+str(i)+'/data/'+f)
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

	df['Zwift'] = False

	# -------------------- Virtual (Zwift)
	if os.path.exists(path+'csv/'+str(i)+'/Zwift/data'):
		files_virtual = sorted(os.listdir(path+'csv/'+str(i)+'/Zwift/data'))
		for j, f in enumerate(files_virtual):
			name = f.rstrip('_data.csv')

			df_data = pd.read_csv(path+'csv/'+str(i)+'/Zwift/data/'+f)
			df_info = pd.read_csv(path+'csv/'+str(i)+'/Zwift/info/'+name+'_info.csv', index_col=(0,1))

			df_data, df_info = cleaning_per_file(df_data, df_info, j+len(files))

			filename_date.update({f:df_data['timestamp'].dt.date.unique()}) # update filelist
			fileid_filename.update({j+len(files):f})

			# create df_information file
			df_info = df_info.T.rename(index={'0':j+len(files)})
			df_information = df_information.append(df_info, verify_integrity=True)

			# open all other files and check for missing info
			try:
				df_n = pd.read_csv(path+'csv/'+str(i)+'/Zwift/nan/'+name+'_nan.csv', index_col=0)
				df_nan = df_nan.append(df_n, ignore_index=True, verify_integrity=True)
			except FileNotFoundError:
				pass

			df = df.append(df_data, ignore_index=True, verify_integrity=True)

		df['Zwift'].fillna(True, inplace=True)

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

# include second stage, so we don't have to run the entire code again if sth goes wrong here
for i in athletes:
	print("\n------------------------------- Athlete ", i)
	df = pd.read_csv(path+'combine/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)
	df_information = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_info.csv', header=[0,1], index_col=0)
	fileid_filename = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_fileid_filename.csv', index_col=0)

	# -------------------- Clean df_data
	# drop empty rows
	cols_ignore = set(['timestamp', 'local_timestamp', 'local_timestamp_loc', 'file_id', 'Zwift'])
	nanrows = df.shape[0] - df.dropna(how='all', subset=set(df.columns)-cols_ignore).shape[0]
	if nanrows > 0:
		print("DROPPED: Number of rows dropped due to being empty: ", nanrows)
		nanrows_index = df.index[~df.index.isin(df.dropna(how='all', subset=set(df.columns)-cols_ignore).index)]
		if verbose == 2:
			print("Timestamps before and after nanrows: ")
			for idx in nanrows_index:
				print(df.iloc[df.index.get_loc(idx)-1:df.index.get_loc(idx)+2])
		# Conclusion: seems to be the first row in a file
		df.drop(nanrows_index, inplace=True)
	else:
		print("GOOD: No rows have to be dropped due to being empty")

	# include device in df
	#df_information[('nunique','')] = 1
	#df_information.groupby([('device_summary', '0'), ('file_id', 'serial_number'), ('device_0', 'serial_number')]).sum()[('nunique','')]
	print("Devices used: ", df_information[('device_summary', '0')].str.strip("nan").str.strip().unique())
	print("Fileid devices used: ", df_information.apply(lambda x: product_info(x, 'file_id'), axis=1).str.strip("nan").str.strip().unique())
	df = pd.merge(df, df_information[('device_summary', '0')].str.strip("nan").str.strip().rename('device_0'),
					left_on='file_id', right_index=True, how='left')
	df = pd.merge(df, df_information[('file_id', 'serial_number')].rename('device_0_serialnumber'), 
					left_on='file_id', right_index=True, how='left')

	df = pd.concat([df, pd.get_dummies(df['device_0'], prefix='device', dtype=bool)], axis=1)
	df.rename(columns={'device_virtualtraining Rouvy':'device_Rouvy', 
					   'device_wahoo_fitness ELEMNT BOLT':'device_ELEMNTBOLT',
					   'device_wahoo_fitness ELEMNT':'device_ELEMNT',
					   'device_wahoo_fitness ELEMNT ROAM':'device_ELEMNTROAM',
					   'device_bkool BKOOL Website':'device_bkool'}, inplace=True)

	# check if df['Zwift'] equals df['device_0'] == 'zwift'
	if not (df['Zwift'] == (df['device_0'] == 'zwift')).all():
		print("WARNING: zwift not correctly identified")
		print("Number of times filename is called Zwift: ", df['Zwift'].sum())
		print("Number of times device is Zwift: ", (df['device_0'] == 'zwift').sum())
		if df['Zwift'].sum() > (df['device_0'] == 'zwift').sum():
			print("Devices when filename is zwift and device is not zwift: ", df[(df['Zwift']) & (df['device_0'] != 'zwift')].device_0.value_counts())
		else:
			print("Filenames when device is zwift and filename is not zwift: ", [fileid_filename.loc[x][0] for x in df[(~df['Zwift']) & (df['device_0'] == 'zwift')].file_id.unique()])
	# TODO: change bool Zwift to zwift device?

	df = df.sort_values(by=['device_0', 'device_0_serialnumber', 'local_timestamp'], key=lambda x: x.map({'ELEMNTBOLT':0, 'zwift':1})) # sort for duplicates later
	df.reset_index(drop=True, inplace=True)
	devices = [x for x in df.columns[df.columns.str.startswith('device_')] if x != 'device_0' and x != 'device_0_serialnumber']

	# fix local timestamp
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df['local_timestamp_loc'] = pd.to_datetime(df['local_timestamp_loc'])

	# print nans local_timestamp and local_timestamp_loc
	print("\n--------------- Timestamp: NAN")
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
	
	# print how often local_timestamp does not equal local_timestamp_loc
	# TODO: find other solution than dropping
	# TODO: this is mainly around the time that the clock is switched from summertime to wintertime. Find out what to do with it!
	print("\n--------------- Timestamp ERROR")
	nan_llts = df['local_timestamp'].isna() | df['local_timestamp_loc'].isna()
	print_times_dates("local timestamp does not equal local timestamp from location (excl. nans)", 
		df[~nan_llts], df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc'])
	print("Dates for which this happens: ", 
		df[~nan_llts][df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc']].timestamp.dt.date.unique())

	#df['drop_error_ltimestamp'] = df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc']
	df.drop(df[~nan_llts][df[~nan_llts]['local_timestamp'] != df[~nan_llts]['local_timestamp_loc']].index, inplace=True)
	print("DROPPED: discrepancy local timestamps")

	# combine both local timestamps
	df['local_timestamp'].fillna(df['local_timestamp_loc'], inplace=True)
	df.drop('local_timestamp_loc', axis=1, inplace=True)

	# print duplicate timestamps
	print("Number of duplicated entries: ", df.duplicated().sum())

	print("\n--------------- Timestamp DUPLICATE")
	print_times_dates("duplicated timestamps", 
		df[df['timestamp'].notna()], df[df['timestamp'].notna()]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df, df['local_timestamp'].duplicated(), ts='local_timestamp')

	dupl_timestamp_both = df['local_timestamp'].duplicated(keep=False)
	dupl_timestamp_first = df['local_timestamp'].duplicated(keep='first')
	dupl_timestamp_last = df['local_timestamp'].duplicated(keep='last')

	print("Duplicate timestamps in devices:")
	for dev in devices:
		print_times_dates(dev, df, dupl_timestamp_last & df[dev], ts='local_timestamp')
		#print(df[dupl_timestamp_last & df[dev]].file_id.unique())

	print("Duplicate timestamps in serialnumbers:")
	for ser in df['device_0_serialnumber'].unique():
		print_times_dates(str(ser), df, dupl_timestamp_first & (df['device_0_serialnumber'] == ser), ts='local_timestamp')

	# select devices to keep: ELEMNT BOLT and zwift
	keep_devices = ['device_ELEMNTBOLT', 'device_ELEMNTROAM', 'device_zwift']
	df['keep_devices'] = False
	drop_devices = []
	for k in keep_devices:
		try:
			df['keep_devices'] |= df[k]
		except KeyError:
			drop_devices.append(k)
			continue
	keep_devices = list(set(keep_devices) - set(drop_devices))
	
	print("Fraction of data dropped with device selection: ", 
		(~df['keep_devices']).sum()/df.shape[0])
	try:
		print("Are there glucose values in the data that will be dropped with the device selection? ",
			not df[~df['keep_devices']]['glucose'].dropna().empty)
	except KeyError:
		pass

	print("Duplicate timestamps after dropping devices: ")
	for dev in keep_devices:
		print_times_dates(dev, df, dupl_timestamp_first & df[dev] & df['keep_devices'], ts='local_timestamp')
		#print(df[dupl_timestamp_first & df[dev]].file_id.unique())

	print("Do the duplicate timestamps have the same data?")
	print(df[dupl_timestamp_both & df['keep_devices']].sort_values('local_timestamp'))
	# Answer: no, not at all, much different altitude for example. Maybe find way to select faulty values and remove them
	# For now, keep identifyer in them so that we know we should do something with it
	df['duplicate_timestamp'] = dupl_timestamp_both
	print("Fraction of times with duplicate timestamps remaining after device selection: ", 
		(df['duplicate_timestamp'] & df['keep_devices']).sum()/df['keep_devices'].sum())

	df.drop(['device_0', 'device_0_serialnumber'], axis=1, inplace=True)

	# save df to file
	df.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)
	del df

	"""
	OLD: use this when we don't remove nans for local timestamps, or when we don't combine both timestamps
	nan_ts = df['timestamp'].notna()
	nan_lts = df['local_timestamp'].notna()
	nan_ltsl = df['local_timestamp_loc'].notna()

	print_times_dates("duplicated timestamps", 
		df[nan_ts], df[nan_ts]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts], df[nan_lts]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl], df[nan_ltsl]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')

	dupl_timestamp = df[nan_ts][df[nan_ts]['timestamp'].duplicated()]
	dupl_timestamp_first = df[nan_ts]['timestamp'].duplicated(keep='first')
	dupl_timestamp_last = df[nan_ts]['timestamp'].duplicated(keep='last')
	dupl_timestamps_keep = df[nan_ts][df[nan_ts]['timestamp'].duplicated(keep=False)].sort_values('timestamp')

	# TODO: find out a way to combine this info. 
	print("Are there glucose values in the duplicate data that is not dropped? ",
		not df[nan_ts][df[nan_ts]['timestamp'].duplicated(keep=False)].sort_values('timestamp').glucose.dropna().empty)
	print("Are there glucose values in the duplicate that that will be dropped? ",
		not df[nan_ts][df[nan_ts]['timestamp'].duplicated()].sort_values('timestamp').glucose.dropna().empty)

	dupl_devices = []
	for dev in devices:
		print("Duplicate timestamps in %s: "%dev, df[nan_ts][dupl_timestamp_first][dev].sum())
		if df[nan_ts][dupl_timestamp_first][dev].sum() != 0:
			dupl_devices.append(dev)
	# TODO: make sure that this happens in the right order, and that not the original duplicate is removed
	# TODO: find out why garmin is double and if data is the same then (latter: not)

	df['drop_dupl_timestamp'] = df[dupl_devices].sum(axis=1).astype(bool)
	"""

	"""
	print("\n----------------- SELECT: device_0 == ELEMNT")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & df['device_ELEMNTBOLT']], 
		df[nan_ts & df['device_ELEMNTBOLT']]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & df['device_ELEMNTBOLT']], 
		df[nan_lts & df['device_ELEMNTBOLT']]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & df['device_ELEMNTBOLT']], 
		df[nan_ltsl & df['device_ELEMNTBOLT']]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	# TODO: check if duplicated local timestamps were in error_timestamps_fileid

	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_ltsl | (df['device_ELEMNTBOLT'] | df['device_zwift'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	
	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift or Rouvy")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')

	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift or Rouvy or Garmin")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	"""

	"""
	df_dupl = df[df['timestamp'].duplicated(keep=False)]
	if not df_dupl.empty:
		df_dupl['date'] = df_dupl['timestamp'].dt.date
		df_dupl[df_dupl['timestamp'] == df_dupl.iloc[0]['timestamp']]

		print("Dates with multiple files but not duplicate timestamps:\n",
			*tuple(set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist()) 
				- set(df_dupl['date'].unique())))
		print("Dates with duplicate timestamps but not multiple files:\n",
			*tuple(set(df_dupl['date'].unique()) 
				- set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist())))
		# conclusion: duplicate timestamps must be because there are duplicate files
	"""