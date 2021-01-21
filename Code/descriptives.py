# TODO: descriptives for minute data
# TODO: include acceleration
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import os
import gc
import matplotlib

from plot import *
from helper import *

# remove if moved to plot
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from statsmodels.graphics.tsaplots import plot_acf

sns.set()
sns.set_context('paper')
sns.set_style('white')

path = 'Data/TrainingPeaks+LibreView/'
savedir = 'Descriptives/'
if not os.path.exists(savedir):
	os.mkdir(savedir)

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'1sec/') if i.endswith('.csv')])

# 1 sec data
for i in athletes:
	df = pd.read_csv(path+'1sec/'+str(i)+'.csv', index_col=0)
	df.index = pd.to_datetime(df.index)

	cols_plot = np.array([['time_training', 'position_lat', 'position_long', 'altitude', 'grade'],
				 		  ['distance', 'speed', 'acceleration', 'cadence', 'power'],
						  ['left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness'],
				 		  ['heart_rate', 'glucose_BUBBLE', 'Historic Glucose mg/dL', 'Historic Glucose mg/dL (filled)', 'Scan Glucose mg/dL'],
						  ['gps_accuracy', 'temperature', 'temperature_smooth', 'battery_soc', 'battery_soc_ilin']])

	p = PlotData(savedir)

	# plot histogram subplots of all features
	p.plot_feature_distr_subplots(df, i, cols_plot.flatten(), figsize=(20,20))

	# plot diagonal correlation matrix
	p.plot_feature_correlation(df, i, cols_plot.flatten())

	# for one training print all features
	cols_ts = np.array(['altitude', 'speed', 'cadence', 'acceleration', 'power', 'temperature_smooth',
						'heart_rate', 'Historic Glucose mg/dL (filled)', 'Scan Glucose mg/dL'])#,
						#'left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 
						#'left_torque_effectiveness', 'right_torque_effectiveness'])

	df_i = df[df.file_id == df.file_id[0]]

	p.plot_feature_timeseries_parasite(df_i, i, cols_ts, cols_ts, 
		ylims = [(0,1000), (-0,65), (-300,500), (-5,15), (-1500,1500), (17, 55), (-150, 220), (0,220), (0,220)],
		axlocs = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l', 'l'],#, 'r', 'r', 'r', 'r', 'r'],
		lws = [.7, .7, .7, .7, .7, .7, .7, 3., 1.],#, .7, .7, .7, .7, .7],
		alphas = [.7, .7, .7, .7, .7, .7, .7, .85, .85],#, .7, .7, .7, .7, .7], 
		figsize=(20,8),	legend=False)

# 1 min data
for i in athletes:
	df = pd.read_csv(path+'1min/'+str(i)+'.csv', index_col=0, header=[0,1])
	df.index = pd.to_datetime(df.index)

	cols = df.columns
	df.columns = [x[0] + '_' + x[1] for x in df.columns]

	cols_plot = np.array([['time_training_first', 'position_lat_first', 'position_long_first', 'altitude_mean', 'grade_mean'],
				 		  ['distance_mean', 'speed_mean', 'acceleration_mean', 'cadence_mean', 'power_mean'],
						  ['left_right_balance_mean', 'left_pedal_smoothness_mean', 'right_pedal_smoothness_mean', 'left_torque_effectiveness_mean', 'right_torque_effectiveness_mean'],
				 		  ['heart_rate_mean', 'glucose_BUBBLE_mean', 'Historic Glucose mg/dL_mean', 'Historic Glucose mg/dL (filled)_mean', 'Scan Glucose mg/dL_mean'],
						  ['gps_accuracy_mean', 'temperature_mean', 'temperature_smooth_mean', 'battery_soc_mean', 'battery_soc_ilin_mean']])

	p = PlotData(savedir)

	# plot histogram subplots of all features
	p.plot_feature_distr_subplots(df, i, cols_plot.flatten(), figsize=(20,20))

	# plot diagonal correlation matrix
	p.plot_feature_correlation(df, i, cols_plot.flatten())
	cols_excl = ['file_id_first', 'time_training_first', 'zwift_first', 'glucose_BUBBLE_mean', 
				 'device_ELEMNTBOLT_first', 'device_ELEMNTROAM_first', 'device_zwift_first', 
				 'device_glucose_first', 'device_glucose_serial_number_first', 'acceleration_entropy']
	cols_incl = df.columns[~df.columns.isin(cols_excl)]
	mcols_incl = cols[~df.columns.isin(cols_excl)]
	ticks, locs = np.unique(mcols_incl.get_level_values(0), return_index=True)
	ticks, locs = ticks[np.argsort(locs)], locs[np.argsort(locs)]
	p.plot_feature_correlation(df, i, cols_incl, ticks, locs)

	# plot clustermap correlation
	p.plot_feature_clustermap(df, i, cols_incl, cols_incl)
	# TODO: is unfinished..

	# for one training print all features
	cols_ts = np.array(['altitude_mean', 'speed_mean', 'cadence_mean', 'acceleration_mean', 'power_mean', 'temperature_smooth_mean',
						'heart_rate_mean', 'Historic Glucose mg/dL (filled)_mean', 'Scan Glucose mg/dL_mean'])#,
						#'left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 
						#'left_torque_effectiveness', 'right_torque_effectiveness'])

	df_i = df[df.file_id_first == df.file_id_first[0]]

	p.plot_feature_timeseries_parasite(df_i, i, cols_ts, [c.rstrip('mean').rstrip('_') for c in cols_ts], 
		ylims = [(0,1000), (-10,80), (-300,500), (-1,2), (-1500,1500), (17, 55), (-150, 220), (0,220), (0,220)],
		axlocs = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l', 'l'],#, 'r', 'r', 'r', 'r', 'r'],
		lws = [.7, .7, .7, .7, .7, .7, .7, 3., 1.],#, .7, .7, .7, .7, .7],
		alphas = [.7, .7, .7, .7, .7, .7, .7, .85, .85],#, .7, .7, .7, .7, .7], 
		figsize=(20,8),	legend=False)


	# TODO: this plot does not really work out
	# for all trainings, print the mean and std of all features
	df_c = df[cols_ts.tolist() + ['time_training_first']]
	cols_ts = [c.rstrip('mean').rstrip('_') for c in cols_ts]
	df_c.columns = [c.rstrip('mean').rstrip('_') for c in df_c.columns]
	df_c = df_c.groupby('time_training_first').agg({c:['mean', 'std'] for c in cols_ts})

	p.plot_feature_timeseries_agg_parasite(df_c, i, cols_ts, cols_ts, 
		ylims = [(0,5000), (-10,80), (-300,500), (-1,2), (-1500,1500), (0, 100), (-200, 250), (-1500,500), (0,220)],
		axlocs = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l', 'l'],#, 'r', 'r', 'r', 'r', 'r'],
		lws = [.7, .7, .7, .7, .7, .7, .7, 3., 1.],#, .7, .7, .7, .7, .7],
		alphas = [.7, .7, .7, .7, .7, .7, .7, .85, .85],#, .7, .7, .7, .7, .7], 
		figsize=(20,8),	legend=False)