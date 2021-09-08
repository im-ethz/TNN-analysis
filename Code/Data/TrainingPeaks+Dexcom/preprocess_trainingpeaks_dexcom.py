# TODO: should I filter by source device ID?
import os
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import scipy as sp
import pandas as pd
import datetime
import gc
import matplotlib

from plot import *
from helper import *
from calc import *

verbose = 1

path_dexcom = '../Dexcom/'
path_trainingpeaks = '../TrainingPeaks/2019/clean2/'
path_merge = './'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path_trainingpeaks)])

# --------------------- merge trainingpeaks with dexcom
if not os.path.exists(path_merge+'trainingpeaks_dexcom.csv'):
	# read trainingpeaks data
	df = pd.concat({i: pd.read_csv(path_trainingpeaks+str(i)+'/'+str(i)+'_data.csv', index_col=0) for i in athletes})
	df = df.reset_index().rename(columns={'level_0':'RIDER'}).drop('level_1', axis=1)

	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
	df.drop('timestamp', axis=1, inplace=True)

	# read dexcom data
	df_glucose = pd.read_csv(path_dexcom+'dexcom_clean.csv', index_col=0)
	df_glucose['local_timestamp'] = pd.to_datetime(df_glucose['local_timestamp'])

	# select glucose measurements
	df_glucose = df_glucose[df_glucose['Event Type'] == 'EGV']
	df_glucose.drop(['Event Subtype', 'Insulin Value (u)', 'Carb Value (grams)', 'Duration (hh:mm:ss)'], axis=1, inplace=True)
	df_glucose.drop(['Source Device ID', 'Transmitter Time (Long Integer)', 'Transmitter ID'], axis=1, inplace=True)

	df_merge = pd.merge(df, df_glucose, how='outer', on=['RIDER', 'local_timestamp'], validate='one_to_one')
	
	df_merge.to_csv(path_merge+'trainingpeaks_dexcom.csv')
	del df, df_glucose, df_merge

"""
# --------------------- remove trainingsessions without glucose entries
# TODO: edit this for glucose values in TP
if not os.path.exists(path_merge+'trainingpeaks_dexcom_dropnaglucose.csv'):
	df = pd.read_csv(path_merge+'trainingpeaks_dexcom.csv', index_col=0)

	# remove training sessions that have no glucose entries
	mask_nan = df.groupby(['RIDER', 'file_id'])['Glucose Value (mg/dL)'].count() == 0
	idx_nan = mask_nan[~mask_nan].rename('filter').reset_index()
	df = pd.merge(df, idx_nan, how='inner', on=['RIDER', 'file_id'], validate='many_to_one')
	df.drop('filter', axis=1, inplace=True)

	df.to_csv(path_merge+'trainingpeaks_dexcom_dropnaglucose.csv')
	del df
"""

# move all trainingpeaks_dexcom csv files to OLD
# scp gpu04:/local/home/evanweenen/bicyclediabetes/trainingpeaks_dexcom_left_sess.csv

# get time in hypoL2, time in hypoL1, etc.

# --------------------- aggregate per training session
def time_in_hypoL2(x):
	return ((x >= glucose_levels['hypo L2'][0]) & (x <= glucose_levels['hypo L2'][1])).sum()

def time_in_hypoL1(x):
	return ((x >= glucose_levels['hypo L1'][0]) & (x <= glucose_levels['hypo L1'][1])).sum()

def time_in_normal(x):
	return ((x >= glucose_levels['normal'][0]) & (x <= glucose_levels['normal'][1])).sum()

def time_in_hyperL1(x):
	return ((x >= glucose_levels['hyper L1'][0]) & (x <= glucose_levels['hyper L1'][1])).sum()

def time_in_hyperL2(x):
	return ((x >= glucose_levels['hyper L2'][0]) & (x <= glucose_levels['hyper L2'][1])).sum()

def til_to_pil(df, col):
	levels = ['hypoL2', 'hypoL1', 'normal', 'hyperL1', 'hyperL2']
	for l in levels:
		df.loc[:, (col,'perc_in_'+l)] = df.loc[:, (col, 'time_in_'+l)] / (df.loc[:, ('time_training', 'max')]/60/5) * 100
	return df

# TODO: impute first before doing this

# TODO: feature engineering for all power-related features with zeros in there
col_features = ['altitude', 'distance', 'speed', 'grade',
				'power', 'combined_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness',
				'cadence', 'left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 
				'temperature', 'heart_rate', 'glucose', 'Glucose Value (mg/dL)']

agg_dict = {col: [np.sum, np.mean, np.median, np.std, sp.stats.iqr, np.min, np.max] for col in col_features}
#agg_dict['Glucose Value (mg/dL)'] += [time_in_hypoL2, time_in_hypoL1, time_in_normal, time_in_hyperL1, time_in_hyperL2]
#agg_dict['glucose'] += [time_in_hypoL2, time_in_hypoL1, time_in_normal, time_in_hyperL1, time_in_hyperL2]
agg_dict.update({'file_id'				:'first',
				 'time_training'		:'max',
				 'local_timestamp'		:['first', 'last'],
				 'position_lat'			:'first',
				 'position_long'		:'first',
				 'gps_accuracy'			:'mean',
				 'battery_soc'			:'mean',
				 'device_ELEMNT'		:'first',
				 'device_ELEMNTBOLT'	:'first',
				 'device_ELEMNTROAM'	:'first',
				 'device_ZWIFT'			:'first',
				 'error_local_timestamp':'first',
				 'Event Type'			:'first'})

# aggregate training session
df = pd.read_csv(path_merge+'trainingpeaks_dexcom.csv', index_col=0)
df.local_timestamp = pd.to_datetime(df.local_timestamp)

df_session = df.groupby(['RIDER', 'file_id']).agg(agg_dict)

#df_session = til_to_pil(df_session, 'glucose')
#df_session = til_to_pil(df_session, 'Glucose Value (mg/dL)')

df_session.to_csv(path_merge+'trainingpeaks_dexcom_sess.csv')

# calculate training session statistics
fitness = pd.read_csv('../fitness/fitness_analysis_2019_anonymous.csv', index_col=[0,1], header=[0,1])
FTP = fitness[("60' power", 'WKO4')].groupby('RIDER').last()

def session_stats(x):
	d = {}
	d['normalised_power'] = normalised_power(x.set_index('local_timestamp')['power'])
	d['functional_threshold_power'] = FTP.loc[x['RIDER'].min()]
	d['intensity_factor'] = intensity_factor(d['normalised_power'], d['functional_threshold_power'])
	d['training_stress_score'] = training_stress_score(x['time_training'].max(), d['intensity_factor'])
	d['variability_index'] = variability_index(d['normalised_power'], x['power'])
	d['efficiency_factor'] = efficiency_factor(d['normalised_power'], x['heart_rate'])
	return pd.Series(d)

df_session_stats = df.groupby(['RIDER', 'file_id']).apply(session_stats)

df_session_stats['chronic_training_load'] = df_session_stats.groupby('RIDER')['training_stress_score'].transform(chronic_training_load)
df_session_stats['acute_training_load'] = df_session_stats.groupby('RIDER')['training_stress_score'].transform(acute_training_load)
df_session_stats['training_stress_balance'] = training_stress_balance(df_session_stats['chronic_training_load'], df_session_stats['acute_training_load'])

df_session_stats.columns = pd.MultiIndex.from_product([df_session_stats.columns, ['']])
df_session = df_session.join(df_session_stats)

df_session.to_csv(path_merge+'trainingpeaks_dexcom_sess_stats.csv')



# sliding window approach


# resample to 1 min
df_1min = df.groupby(['RIDER', pd.Grouper(key='local_timestamp', freq='1min')]).agg(agg_dict)
df_5min = df.groupby(['RIDER', pd.Grouper(key='local_timestamp', freq='5min')]).agg(agg_dict)

df_1min.to_csv(path_merge+'trainingpeaks_dexcom_1min.csv')
df_5min.to_csv(path_merge+'trainingpeaks_dexcom_5min.csv')

# --------------------- merge glucose from trainingpeaks with glucose from dexcom
df = pd.read_csv(path_merge+'trainingpeaks_dexcom.csv', index_col=0)
print("CHECK: %s glucose entries from TP"%df.glucose.notna().sum())
print("CHECK: %s glucose entries from Dexcom"%df['Glucose Value (mg/dL)'].notna().sum())
# TODO: they will only overlap if they are at the exact second
#print("CHECK: %s entries with both glucose from TP and Dexcom"\
#	%(df.glucose.notna() & df['Glucose Value (mg/dL)'].notna()).sum())
#print("CHECK: %s entries where glucose from TP and Dexcom differ"\
#	%(df.glucose.notna() & df['Glucose Value (mg/dL)'].notna() & (df.glucose != df['Glucose Value (mg/dL)'])).sum())
#df.glucose.fillna(df['Glucose Value (mg/dL)'], inplace=True)
#df.drop('Glucose Value (mg/dL)', axis=1, inplace=True)
