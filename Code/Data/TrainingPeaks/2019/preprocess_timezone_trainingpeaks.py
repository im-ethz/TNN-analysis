"""
Create file with timezones extracted from each TrainingPeaks files
Get location from coordinates of TrainingPeaks file, using reverse geo
Fill up unknown timezones and locations from previous and following timezones and locations

Note: getting home locations for Zwift does not seem to work, they seem to take it when travelling
"""
import os
import sys
sys.path.append(os.path.abspath('../../../'))

import numpy as np
import pandas as pd

import datetime
import geopy

from config import rider_mapping
from plot import *
from helper import *

import gc

path = './clean/'
if not os.path.exists('timezone'):
	os.mkdir('timezone')

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path)])

rider_mapping_inv = {v:k for (k,v) in rider_mapping.items()}

# --------------------- extract tz from trainingpeaks files
for i in athletes:
	print("\n------------------------------- Athlete ", i)
	df_i = pd.read_csv(path+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df_i['timestamp'] = pd.to_datetime(df_i['timestamp'])
	df_i['local_timestamp'] = pd.to_datetime(df_i['local_timestamp'])
	df_i['local_timestamp_loc'] = pd.to_datetime(df_i['local_timestamp_loc'])

	df_i = df_i.sort_values('timestamp')

	print("\n-------- export timezone")
	df = df_i[['timestamp', 'position_lat', 'position_long', 'local_timestamp', 'local_timestamp_loc', 'error_local_timestamp', 'file_id', 'device_0']]
	df['timezone'] = df['local_timestamp'] - df['timestamp']
	df['timezone_loc'] = df['local_timestamp_loc'] - df['timestamp']
	df['date'] = df['timestamp'].dt.date
	df = df.groupby(['file_id']).agg({'timestamp'				: ['min', 'max'],
									  'local_timestamp'			: ['min', 'max'],
									  'local_timestamp_loc'		: ['min', 'max'],
									  'error_local_timestamp'	: 'first',
									  'device_0'				: 'first',
									  'position_lat'			: 'first',
									  'position_long'			: 'first',
									  'timezone'				: 'first',
									  'timezone_loc'			: 'first'})
	df.to_csv(path+str(i)+'/'+str(i)+'_timezone.csv')

# --------------------- get locations
df = pd.concat({i: pd.read_csv(path+str(i)+'/'+str(i)+'_timezone.csv', index_col=[0], header=[0,1]) for i in athletes}, names=['RIDER', 'file_id'])
df.columns = [i[0]+'_'+i[1] if 'timestamp' in i[0] else i[0] for i in df.columns]
df.rename(columns={'error_local_timestamp_first':'error_local_timestamp'}, inplace=True)

df['timestamp_min'] = pd.to_datetime(df['timestamp_min'])
df['timestamp_max'] = pd.to_datetime(df['timestamp_max'])
df['local_timestamp_min'] = pd.to_datetime(df['local_timestamp_min'])
df['local_timestamp_max'] = pd.to_datetime(df['local_timestamp_max'])
df['local_timestamp_loc_min'] = pd.to_datetime(df['local_timestamp_loc_min'])
df['local_timestamp_loc_max'] = pd.to_datetime(df['local_timestamp_loc_max'])
df['timezone'] = pd.to_timedelta(df['timezone'])
df['timezone_loc'] = pd.to_timedelta(df['timezone_loc'])

df.reset_index(inplace=True)

# sort by rider and date
df = df.sort_values(['RIDER', 'timestamp_min'])
df.reset_index(drop=True, inplace=True)

# get locations
geo = geopy.geocoders.Nominatim(user_agent="trainingpeaks_locations")
for idx, (lat, lon) in df[['position_lat', 'position_long']].iterrows():
	if not np.isnan(lat) and not np.isnan(lon):
		df.loc[idx, 'location'] = geo.reverse(query=str(lat)+", "+str(lon),
			language='en', zoom=0).raw['address']['country']
print("Number of files without location: ", df.location.isna().sum())

# remove locations for zwift files
df.loc[df.device_0 == 'ZWIFT', 'location'] = np.nan 

# TODO: change this in preprocess_trainingpeaks
print("Note: location found with geopy, but no timezone found with pytz: ",
	df[df.location.notna() & df.timezone_loc.isna()])

# add date (useful for later)
df['date'] = df['timestamp_min'].dt.date

df.to_csv('./timezone/timezone_raw.csv')

# --------------------- check raw file Fede
# TODO: check if timestamp_min is sorted
df_check = df[['RIDER', 'date', 'position_lat', 'position_long', 'location', 'timezone', 'timezone_loc', 'error_local_timestamp', 'device_0']]
df_check.rename(columns={'timezone'				:'timezone_from_file', 
						 'timezone_loc'			:'timezone_from_location', 
						 'error_local_timestamp':'error_timezone',
						 'device_0'				:'device'}, inplace=True)
df_check.RIDER = df_check.RIDER.replace(rider_mapping_inv)
df_check.to_csv('./timezone/timezone_check_raw.csv')


# --------------------- preprocess tz
df = pd.read_csv('./timezone/timezone_raw.csv', index_col=0)
df['date'] = pd.to_datetime(df['date'])
df['timestamp_min'] = pd.to_datetime(df['timestamp_min'])
df['timestamp_max'] = pd.to_datetime(df['timestamp_max'])

print("\n---------------- Fill-up NAN timezones (and country) from two days before and after ----------------")

for name, col in zip(['tz_meta', 'tz_loc', 'country'], ['timezone', 'timezone_loc', 'location']):
	print("NAN %s: \n from"%name, df[col].isna().sum())
	idx_nan = df.loc[df[col].isna(), ['RIDER', 'date']]
	for idx, (i, d) in idx_nan.iterrows():
		# take time window of two days before and after the date
		mask_prev = (df.RIDER == i) & (df.date >= d-datetime.timedelta(days=2)) & (df.date <= d)
		mask_next = (df.RIDER == i) & (df.date <= d+datetime.timedelta(days=2)) & (df.date >= d)

		col_prev = df[mask_prev][col].dropna()
		col_next = df[mask_next][col].dropna()

		# if there is at least one entry before and one entry after within the time window
		# and the "col" of these entries is the same
		if len(col_prev) >= 1 and len(col_next) >= 1 and len(pd.concat([col_prev, col_next]).unique()) == 1:
			df.loc[idx, col] = pd.concat([col_prev, col_next]).unique()[0]
	print(" to  ", df[col].isna().sum())

print("\n---------------- Fill-up NAN timezones (and country) from file before and after ----------------")

for name, col in zip(['tz_meta', 'tz_loc', 'country'], ['timezone', 'timezone_loc', 'location']):
	# fill up zwift (file before and file after)
	print("NAN %s: \n from"%name, df[col].isna().sum())
	idx_nan = df.loc[df[col].isna()].index
	for idx in idx_nan:

		i_prev = df.loc[:idx-1, col].last_valid_index()
		i_next = df.loc[idx+1:, col].first_valid_index()

		col_prev = df.loc[i_prev][col]
		col_next = df.loc[i_next][col]

		# if previous "col" equals next "col" (provided they are from the same rider)
		if col_prev == col_next and df.loc[i_prev, 'RIDER'] == df.loc[i_next, 'RIDER']:
			df.loc[idx, col] = col_prev
	print(" to  ", df[col].isna().sum())

df.to_csv('./timezone/timezone_filled.csv')

print("\n---------------- Remaining NANs ----------------")

idx_nan_remain = df.loc[df.location.isna() | df.timezone_loc.isna() | df.timezone.isna()].index
print(idx_nan_remain)

print("\n---------------- Combine timezones ----------------")
# when timezone_loc == timezone
df.loc[df['timezone'] == df['timezone_loc'], 'timezone_combine'] = df['timezone']
print("Number of timezone entries to figure out: ", df['timezone_combine'].isna().sum())
df.to_csv('./timezone/timezone_combine.csv')

# it seems the device needs some time to update its timezone, even though the timezone has already been changed
# (so the change first happens in the timezone_loc column, and then in the timezone column)
# therefore just use the timezone_loc in the cases that we don't know

# take the timezone from location if we don't know
df['timezone_combine'] = df['timezone_combine'].fillna(df['timezone_loc'])
print("Number of timezone entries to figure out: ", df['timezone_combine'].isna().sum())

df.to_csv('./timezone/timezone_combine2.csv')


# EXCEPTIONS:

# 3 (clancy) 2019-10-19 366 can't travel to japan spain japan in one day
# 12 (lozano) 2019-09-26 a lot of travelling between spain and italy, is this possible?
# 12 (lozano) 2019-10-13 can't travel between china spain and back to china in one day

# --------------------- save files for Fede to check

# file to send to Fede for checking
df_check = df[['RIDER', 'timestamp_min', 'date', 'position_lat', 'position_long', 'location', 'timezone', 'timezone_loc', 'timezone_combine', 'error_local_timestamp', 'device_0']]
df_check.rename(columns={'timezone'				:'timezone_from_file', 
						 'timezone_loc'			:'timezone_from_location',
						 'timezone_combine'		:'timezone_final',
						 'error_local_timestamp':'error_timezone',
						 'device_0'				:'device'}, inplace=True)
df_check.drop_duplicates(['RIDER', 'date', 'timezone_final'], inplace=True)
df_check.reset_index(drop=True, inplace=True)
df_check.drop('timestamp_min', axis=1, inplace=True)
df_check.RIDER = df_check.RIDER.replace(rider_mapping_inv)
df_check.to_csv('./timezone/timezone_check_filled.csv')

# easier file format for Fede to check
df_check['travel'] = df_check.groupby('RIDER')['location'].transform(lambda x: x.shift() != x)
# create n_location column
for i in df_check.RIDER.unique():
	df_check.loc[df_check.RIDER == i, 'n_location'] = df_check[(df_check.RIDER == i) & df_check['travel']].reset_index().reset_index().set_index('index')['level_0']
df_check['n_location'] = df_check['n_location'].fillna(method='ffill')

df_check = df_check.groupby(['RIDER', 'n_location']).agg({'date'					:['min', 'max'],
														  'location'				:'min',
														  'position_lat'			:'min',
														  'position_long'			:'min',
														  'timezone_from_file'		:lambda x: list(x.unique()),# if len(x.unique()) > 1 else x,
														  'timezone_from_location'	:lambda x: list(x.unique()),# if len(x.unique()) > 1 else x,
														  'timezone_final'			:lambda x: list(x.unique()),
														  'error_timezone'			:'max',
														  'device'					:lambda x: list(x.unique())})
df_check.columns = [i[0]+'_'+i[1] if i[0] == 'date' else i[0] for i in df_check.columns]
df_check = df_check.reset_index().drop('n_location', axis=1)
df_check.to_csv('./timezone/timezone_check_filled_easy.csv')

print("\n---------------- Final file ----------------")
df = pd.read_csv('./timezone/timezone_combine2.csv', index_col=0)

df = df[['RIDER', 'file_id', 'timestamp_min', 'timestamp_max', 'date', 'timezone_combine', 'location']]
df.rename(columns={'timezone_combine':'timezone'}, inplace=True)

df.to_csv('./timezone/timezone_list_final.csv')

df['date'] = pd.to_datetime(df['date'])
df['timestamp_min'] = pd.to_datetime(df['timestamp_min'])
df['timestamp_max'] = pd.to_datetime(df['timestamp_max'])
df['timezone'] = pd.to_timedelta(df['timezone'])

df.dropna(subset=['timezone'], inplace=True)

df['local_timestamp_min'] = df['timestamp_min'] + df['timezone']
df['local_timestamp_max'] = df['timestamp_max'] + df['timezone']



# MANUAL drop timestamps that seem very odd (i.e. travelling from Japan to Spain and back in one day)
df.drop([1040,2827,2832], inplace=True)
# TODO 1040: check the date of the Gramin device for 3
# because now it says Spain in between to Japan trips
# so either the date is not correct of the Garmin and everything has to be shifted
# or he has to reset the Garmin manually, and did not do it yet after travelling.
# Note: there is also no timezone known - maybe look into this individual file to see what is going on?
# Note: it seems that the timezone of 1041 and 1042 is incorrect instead (when looking at when the dexcom was censored)

# TODO: Lozano: check 
# 2809 2019-09-28 Spain
# 2811 2019-09-30 Spain
# 2813 2019-10-02 Spain
# 2816 2019-10-04 Spain
# 2817 2019-10-05 Spain
# 2827 2019-10-13 Spain
# 2832 2019-10-19 China
# and further! This whole month of data is weird

# TODO: check if we should also drop them in general!!!

# create n_location column
df['travel'] = df.groupby('RIDER')['timezone'].transform(lambda x: x.shift() != x)
for i in df.RIDER.unique():
	df.loc[df.RIDER == i, 'n'] = df[(df.RIDER == i) & df['travel']].reset_index().reset_index().set_index('index')['level_0']
df['n'] = df['n'].fillna(method='ffill')
df = df.groupby(['RIDER', 'n']).agg({'timezone'				:'first',
									 'date'					:['min', 'max'], 
									 'timestamp_min'		:'min',
									 'timestamp_max'		:'max',
									 'local_timestamp_min'	:'min',
									 'local_timestamp_max'	:'max'})
df.columns = [i[0]+'_'+i[1] if i[0] == 'date' else i[0] for i in df.columns]
df.to_csv('./timezone/timezone_final.csv')