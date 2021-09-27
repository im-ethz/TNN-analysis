# TODO: put features into different modalities
# TODO: clean out extreme values in script before
# TODO: first remove training sessions with little info
# TODO: it seems sth is wrong with removing zeros
# then remove variables with little info
# NOTE: aggregation is by LOCAL timestamp date
import os
import sys
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd

from calc import calc_hr_zones, calc_power_zones
from calc import agg_power, agg_zones, agg_stats
from calc import chronic_training_load, acute_training_load, training_stress_balance

from config import DATA_PATH

import gc

SAVE_PATH = './'

# ----------------------- fitness
# read in fitness variables
fitness = pd.read_csv(DATA_PATH+'fitness.csv', index_col=[0,1], header=[0,1])

# determine functional threshold power
FTP = fitness[[("VT2 (RCP)", 'W')]].dropna().reset_index().groupby('RIDER').mean()
LTHR = fitness[[("VT2 (RCP)", 'HR')]].loc[pd.IndexSlice[:, 'Jan_2019'],:].reset_index().drop('date', axis=1).set_index('RIDER')

# calculate HR and Power zones
hr_zones = LTHR.apply(calc_hr_zones, axis=1)
power_zones = FTP.apply(calc_power_zones, axis=1)

# ----------------------- calendar
# race calendar
cal_race = pd.read_csv(DATA_PATH+'calendar.csv', index_col=0)
cal_race = cal_race[cal_race.type == 'R'] # filter races
cal_race.drop('type', axis=1, inplace=True)
cal_race['start_date'] = pd.to_datetime(cal_race['start_date'])
cal_race['end_date'] = pd.to_datetime(cal_race['end_date'])

# travel calendar
cal_travel = pd.read_csv(DATA_PATH+'travel.csv', index_col=[0,1])
cal_travel.reset_index(inplace=True)
cal_travel['local_timestamp_min'] = pd.to_datetime(cal_travel['local_timestamp_min'])
cal_travel['local_timestamp_max'] = pd.to_datetime(cal_travel['local_timestamp_max'])
cal_travel['date_min'] = pd.to_datetime(cal_travel['local_timestamp_min'].dt.date) 
cal_travel['date_max'] = pd.to_datetime(cal_travel['local_timestamp_max'].dt.date - pd.to_timedelta('1d'))
cal_travel = cal_travel[['RIDER', 'date_min', 'date_max']]

# ----------------------- aggregation
df_agg = {}

athletes = set(sorted([int(i.rstrip('.csv').rstrip('_info').rstrip('_data')) for i in os.listdir(DATA_PATH+'trainingpeaks/')]))

for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(DATA_PATH+'trainingpeaks/'+str(i)+'_data.csv', index_col=0)
	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df['date'] = df['local_timestamp'].dt.date

	# calculate time statistics
	df_times = df.groupby('date').agg({'timestamp'			: ['min', 'max', 'count'],
									   'local_timestamp'	: ['min', 'max']})
	df_times.columns = [c1+'_'+c2 for c1, c2 in df_times.columns]

	# combine pedal smoothness
	# TODO: check if correct
	try:
		df['combined_pedal_smoothness'].fillna(df['left_pedal_smoothness']*(df['left_right_balance'].clip(0,100)/100)
			+ df['right_pedal_smoothness']*(1-df['left_right_balance'].clip(0,100)/100), inplace=True)
	except KeyError:
		df['combined_pedal_smoothness'] = df['left_pedal_smoothness']*(df['left_right_balance'].clip(0,100)/100)\
			+ df['right_pedal_smoothness']*(1-df['left_right_balance'].clip(0,100)/100)

	# split out columns in ascent and descent
	df['descent'] = df.groupby('file_id')['altitude'].transform(lambda x: x.interpolate(method='linear').diff() < 0)
	for col in ['altitude', 'speed', 'distance', 'heart_rate', 'power', 'cadence']:
		df[col+'_up'] = df.loc[~df['descent'], col]
		df[col+'_down'] = df.loc[df['descent'], col]
	df.drop('descent', axis=1, inplace=True)

	# calculate features with removing zeros
	for col in ['power', 'combined_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness',
				'cadence', 'left_pedal_smoothness', 'right_pedal_smoothness']:
		try:
			df[col+'_n0'] = df[col].replace({0:np.nan})
		except KeyError:
			pass

	# calculate flirt statistics
	col_stats = set(df.columns)-set(['position_long', 'position_lat', 'gps_accuracy', 'battery_soc', 'device_0', 'local_timestamp', 'timestamp', 'time_training', 'file_id'])
	df_stats = df[col_stats].groupby('date').apply(agg_stats)
	df_stats.columns = [c1+'_'+c2 for c1, c2 in df_stats.columns]

	# clean columns
	df_stats.drop(df_stats.columns[df_stats.notna().sum() == 0], axis=1, inplace=True) #empty cols

	# calculate power statistics and hr and power zones
	df.set_index('timestamp', inplace=True)

	df_zones = df.groupby('date').apply(agg_zones, hr_zones=hr_zones.loc[i], power_zones=power_zones.loc[i])
	df_power = df.groupby('date').apply(agg_power, FTP=FTP.loc[i].item())

	df_power['chronic_training_load'] = chronic_training_load(df_power['training_stress_score'])
	df_power['acute_training_load'] = acute_training_load(df_power['training_stress_score'])
	df_power['training_stress_balance'] = training_stress_balance(df_power['chronic_training_load'], df_power['acute_training_load'])

	# read calories from info file
	df_files = df[['file_id', 'date']].drop_duplicates().set_index('file_id')
	df_info = pd.read_csv(DATA_PATH+'trainingpeaks/'+str(i)+'_info.csv', index_col=0)
	df_calos = df_files.join(df_info['total_calories']).groupby('date').sum()

	df_agg[i] = pd.concat([df_times, df_zones, df_power, df_calos, df_stats], axis=1)

	del df, df_times, df_zones, df_power, df_calos, df_stats, df_files, df_info

df_agg = pd.concat(df_agg)
df_agg = df_agg.reset_index().rename(columns={'level_0':'RIDER'})

# merge with race info
df_agg['race'] = False
for _, (i, d_start, d_end) in cal_race.iterrows():
	df_agg.loc[(df_agg.RIDER == i) & (df_agg.date >= d_start) & (df_agg.date <= d_end), 'race'] = True

# merge with travel info
df_agg['travel'] = True
for _, (i, d_start, d_end) in cal_travel.iterrows():
	df_agg.loc[(df_agg.RIDER == i) & (df_agg.date > d_start) & (df_agg.date < d_end), 'travel'] = False

df_agg.to_csv(SAVE_PATH+'trainingpeaks_day.csv', index_label=False)

# ----------------------- modalities
# TODO: split up and down?
df_agg = df_agg.set_index(['RIDER', 'date'])

modalities = {}

# TIME
modalities.update({k:'TIME' for k in ['timestamp_min', 'timestamp_max', 'timestamp_count', 'local_timestamp_min', 'local_timestamp_max']})

# CALENDAR
modalities.update({k:'CALENDAR' for k in ['race', 'travel']})

# HR
modalities.update({k:'HR' for k in df_agg.columns[df_agg.columns.str.startswith('time_in_hr')]})
modalities.update({k:'HR' for k in df_agg.columns[df_agg.columns.str.startswith('heart_rate')]})
modalities['efficiency_factor'] = 'HR'

# POWER
modalities.update({k:'POWER' for k in df_agg.columns[df_agg.columns.str.startswith('time_in_power')]})
modalities.update({k:'POWER' for k in df_agg.columns[df_agg.columns.str.startswith('power')]})
modalities.update({k:'POWER' for k in df_agg.columns[df_agg.columns.str.startswith('left_')]})
modalities.update({k:'POWER' for k in df_agg.columns[df_agg.columns.str.startswith('right_')]})
modalities.update({k:'POWER' for k in df_agg.columns[df_agg.columns.str.startswith('combined_')]})
modalities.update({k:'POWER' for k in df_agg.columns[df_agg.columns.str.startswith('cadence')]})
modalities['normalised_power'] = 'POWER'
modalities['intensity_factor'] = 'POWER'
modalities['training_stress_score'] = 'POWER'
modalities['variability_index'] = 'POWER'
modalities['chronic_training_load'] = 'POWER'
modalities['acute_training_load'] = 'POWER'
modalities['training_stress_balance'] = 'POWER'
modalities['total_calories'] = 'POWER'

# LOC
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('grade')]})
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('altitude')]})
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('distance')]})
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('speed')]})
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('acceleration')]})
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('temperature')]})
modalities.update({k:'LOC' for k in df_agg.columns[df_agg.columns.str.startswith('elevation_gain')]})

# sort columns
df_agg = df_agg[modalities.keys()]
df_agg.columns = pd.MultiIndex.from_tuples([(v,k) for k, v in modalities.items()])
df_agg.to_csv(SAVE_PATH+'trainingpeaks_day_modalities.csv')