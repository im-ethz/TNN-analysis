# TODO: clean out extreme values in script before
# TODO: first remove training sessions with little info
# NOTE: aggregation is by LOCAL timestamp date
import os
import numpy as np
import pandas as pd
import gc

import datetime
import pycountry

from calc import combine_pedal_smoothness
from calc import calc_hr_zones, calc_power_zones
from calc import agg_power, agg_zones, agg_stats
from calc import chronic_training_load, acute_training_load, training_stress_balance

from config import DATA_PATH

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in long_scalars")

SAVE_PATH = DATA_PATH+'agg/'

date_range = pd.date_range(start='2014-01-01', end='2021-12-31', freq='1d')

# ----------------------- info
info = pd.read_csv(DATA_PATH+'info/info.csv')

# get age on 01-01-2019
info['age'] = 2018 - info['birthyear']
info['diabetes_duration'] = info['age'] - info['age_diagnosis']
info = info.drop(['name', 'athlete_type', 'dob', 'birthyear', 'age_diagnosis'], axis=1)

# ----------------------- fitness
# read in fit variables
fitness = pd.read_csv(DATA_PATH+'fitness/fitness.csv', header=[0,1], index_col=[0,1])
fitness = fitness.reset_index()

# take average for beginning, mid and end of season
fitness = fitness.groupby('RIDER').mean()
fitness = fitness.reset_index()

# select right columns
cols = {('RIDER', '')						: 'RIDER',
		('ID and ANTROPOMETRY', 'Height')	: 'height', #cm
		('ID and ANTROPOMETRY', 'Weight')	: 'weight', #kg
		('ID and ANTROPOMETRY', 'bf(%)')	: 'bf(%)', #%
		('ID and ANTROPOMETRY', 'HbA1C')	: 'HbA1c', #% (?)
		('VT2 (RCP)', 'W')					: 'FTP',#W
		('VT2 (RCP)', 'HR')					: 'LTHR', #bpm
		('VO2peak', 'HR')					: 'HRmax', #bpm
		('VO2peak', 'VO2/Kg')				: 'VO2max', #mL/min/kg
		}
fitness = fitness[cols.keys()]
fitness.columns = ['_'.join(c) for c in fitness.columns]
fitness = fitness.rename(columns={'_'.join(k):v for k,v in cols.items()})

info = pd.merge(info, fitness, how='outer', on='RIDER')
info['height'] = info['height_y'].fillna(info['height_x'])
info = info.drop(['height_x', 'height_y'], axis=1)
info = info.set_index('RIDER')
info.to_csv(SAVE_PATH+'info.csv', index_label=False)

# ----------------------- zones
# calculate HR and Power zones
hr_zones = info['LTHR'].apply(calc_hr_zones)
power_zones = info['FTP'].apply(calc_power_zones)

# ----------------------- calendar
# race
race = pd.read_csv(DATA_PATH+'calendar/procyclingstats.csv', index_col=0)
race['date'] = pd.to_datetime(race['date'])
race = race[['RIDER', 'date']]
race['race'] = True
race = race.drop_duplicates()

# travel
timezones = pd.read_csv(DATA_PATH+'timezone.csv', index_col=0)
timezones['date'] = pd.to_datetime(timezones['date'])
travel = timezones.loc[timezones['travel'], ['RIDER', 'date', 'travel']]

timezones = timezones[['RIDER', 'date', 'travel', 'country']]
timezones = timezones.set_index(['RIDER', 'date'])

# ----------------------- country/carbs info
country_nutrients = pd.read_csv(DATA_PATH+'carbs/country_nutrients.csv')
country_carbs = country_nutrients.set_index('code')['carbohydrates (kcal)'].to_dict()

# ----------------------- aggregation
df_agg = {}

athletes = sorted([int(i) for i in os.listdir(DATA_PATH+'TrainingPeaks/clean/')])

for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(DATA_PATH+f'TrainingPeaks/clean/{i}/{i}_data4.csv', index_col=0)
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df['date'] = df['local_timestamp'].dt.date

	# remove zeros before calculating averages
	df['power'] = df['power'].replace({0:np.nan})
	df['heart_rate'] = df['heart_rate'].replace({0:np.nan})

	# remove zeros if all values of a feature in a file are zero
	cols_zero = ('distance', 'speed', 'grade', 'acceleration', 'elevation_gain', 'ascent', 'cadence',
		'left_pedal_smoothness', 'right_pedal_smoothness', 'combined_pedal_smoothness',
		'left_torque_effectiveness', 'right_torque_effectiveness', 'left_right_balance')

	for col in df.columns[df.columns.isin(cols_zero)]:
		frac_zero = df.groupby('file_id')[col].apply(lambda x: x[x==0].count() / x.count())
		df.loc[df.file_id.isin(frac_zero[frac_zero == 1].index), col] = np.nan

	cols_nan = cols_zero + ('temperature', 'altitude', 'position_lat', 'position_long', 'power', 'heart_rate')
	df = df.dropna(subset=df.columns[df.columns.isin(cols_nan)], how='all')
	"""
	# combine pedal smoothness
	if 'combined_pedal_smoothness' not in df:
		df['combined_pedal_smoothness'] = df[['left_pedal_smoothness', 'right_pedal_smoothness', 'left_right_balance']].apply(combine_pedal_smoothness)
	else:
		df['combined_pedal_smoothness'] = df['combined_pedal_smoothness'].fillna( 
			df[['left_pedal_smoothness', 'right_pedal_smoothness', 'left_right_balance']].apply(combine_pedal_smoothness))
	"""
	"""
	# remove zeros for some features
	for col in ['power', 'combined_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness',
				'cadence', 'left_pedal_smoothness', 'right_pedal_smoothness']:
		try:
			df[col+'_n0'] = df[col].replace({0:np.nan})
		except KeyError:
			pass
	"""
	# split out columns in ascent and descent
	df['descent'] = df.groupby('file_id')['altitude'].transform(lambda x: x.interpolate(method='linear').diff() < 0)
	for col in ['distance', 'elevation_gain']:#'altitude', 'speed', 'heart_rate', 'power', 'cadence', 'acceleration']:
		df[col+'_up'] = df.loc[~df['descent'], col]
		df[col+'_down'] = df.loc[df['descent'], col]
	df = df.drop('descent', axis=1)

	# ---------- times
	df_times = df.groupby('date').agg({'timestamp'			: ['count'],
									   'local_timestamp'	: ['min', 'max'],
									   'file_id'			: lambda x: len(np.unique(x))})
	df_times = df_times.rename(columns={'<lambda>':'unique_count'})
	df_times.columns = ['_'.join(col) for col in df_times.columns]

	# ---------- carbs
	# get country code
	df_country = df.groupby('date')['country'].first()
	df_country = df_country.fillna(timezones.loc[i, 'country']).fillna(info.loc[i, 'nationality'])
	df_country = df_country.replace({'South Korea'	: 'Korea, Republic of',
									 'North Korea'	: "Korea, Democratic People's Republic of",
									 'Taiwan'		: 'Taiwan, Province of China',
									 'Luzon'		: 'Philippines',
									 'Russia'		: 'Russian Federation'})
	df_country = df_country.to_frame()
	df_country['code'] = df_country['country'].apply(lambda x: pycountry.countries.get(name=x).alpha_3)

	# get carbs per country
	df_carbs = df_country['code'].map(country_carbs).rename('country_carbs').to_frame()
	del df_country

	# ---------- stats
	# calculate flirt statistics
	col_stats = set(df.columns)-set(['country', 'position_long', 'position_lat', 'device_0', 'local_timestamp', 'timestamp', 'time_session', 'file_id'])
	df_stats = df[col_stats].groupby('date').apply(agg_stats)
	df_stats.columns = ['_'.join(col) for col in df_stats.columns]
	df_stats = df_stats.dropna(how='all', axis=1) #empty cols

	# ---------- zones
	# calculate hr and power zones
	df_zones = df.groupby('date').apply(agg_zones, hr_zones=hr_zones.loc[i], power_zones=power_zones.loc[i])
	
	# ---------- power
	# calculate power statistics
	df = df.set_index('timestamp')

	df_power = df.groupby('date').apply(agg_power, FTP=info.loc[i, 'FTP'])

	# fill up dates for which we don't have an entry to get ewm
	dates = df_power.index
	df_power = df_power.reindex(date_range)

	# calculate ctl, atl and tsb
	df_power['chronic_training_load'] = chronic_training_load(df_power['training_stress_score'])
	df_power['acute_training_load'] = acute_training_load(df_power['training_stress_score'])
	df_power['training_stress_balance'] = training_stress_balance(df_power['chronic_training_load'], df_power['acute_training_load'])

	# get back to indices for which there is a training session
	df_power = df_power.loc[dates]

	df_agg[i] = pd.concat([df_times, df_carbs, df_zones, df_power, df_stats], axis=1)

	del df, df_times, df_zones, df_carbs, df_power, df_stats ; gc.collect()

df_agg = pd.concat(df_agg).reset_index().rename(columns={'level_0':'RIDER'})
df_agg['date'] = pd.to_datetime(df_agg['date'])

df_agg = pd.merge(df_agg, race, on=['RIDER', 'date'], how='left')
df_agg = pd.merge(df_agg, travel, on=['RIDER', 'date'], how='left')

df_agg['race'] = df_agg['race'].fillna(False)
df_agg['travel'] = df_agg['travel'].fillna(False)

# fill up dates for which we don't have an entry to do some aggregations
date_range = pd.date_range(start='2014-01-01', end='2021-12-31', freq='1d')
date_index = pd.MultiIndex.from_product([df_agg.RIDER.unique(), date_range], names=['RIDER', 'date']).to_frame().reset_index(drop=True)
df_agg = pd.merge(df_agg, date_index, how='right', on=['RIDER', 'date'])

df_agg['travel_3d_any'] = df_agg.groupby('RIDER').rolling(3, min_periods=1)['travel'].agg(lambda x: x.any()).astype(bool).reset_index(drop=True)
df_agg['travel_7d_any'] = df_agg.groupby('RIDER').rolling(7, min_periods=1)['travel'].agg(lambda x: x.any()).astype(bool).reset_index(drop=True)

df_agg['race_3d_mean'] = df_agg.groupby('RIDER').rolling(3, min_periods=1)['race'].agg(lambda x: x.mean()).reset_index(drop=True)
df_agg['race_7d_mean'] = df_agg.groupby('RIDER').rolling(7, min_periods=1)['race'].agg(lambda x: x.mean()).reset_index(drop=True)

df_agg = df_agg.dropna(subset=['local_timestamp_min'])

df_agg = df_agg.set_index(['RIDER', 'date'])

# modalities
modalities = {}

modalities['TIME'] = ['timestamp_count', 'local_timestamp_min', 'local_timestamp_max']

modalities['CALENDAR'] = ['race', 'travel', 'race_3d_mean', 'race_7d_mean', 'travel_3d_any', 'travel_7d_any', 'country_carbs']

modalities['HR'] = df_agg.columns[df_agg.columns.str.startswith(('time_in_hr', 'heart_rate'))]

modalities['POWER'] = df_agg.columns[df_agg.columns.str.startswith(('time_in_power', 'power', 'left_', 'right_', 'combined', 'cadence'))].to_list() + \
	['normalised_power', 'intensity_factor', 'training_stress_score', 'variability_index', 'efficiency_factor', 
	'chronic_training_load', 'acute_training_load', 'training_stress_balance']

modalities['LOC'] = df_agg.columns[df_agg.columns.str.startswith(('grade', 'altitude', 'distance', 'speed', 'acceleration', 'temperature', 'elevation_gain'))]

modalities = {v:k for k, values in modalities.items() for v in values}

# sort columns
df_agg = df_agg[modalities.keys()]
df_agg.columns = pd.MultiIndex.from_tuples([(v,k) for k, v in modalities.items()])
df_agg.to_csv(SAVE_PATH+'trainingpeaks_day.csv')