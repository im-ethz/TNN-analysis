import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

import datetime as dt

from plot import *

loaddir = 'subject_data/'
savedir = 'Results/'

subject = 'x'
date = dt.date(2020, 9, 19)
time = dt.time(8, 45, 2)

p = PlotData(savedir)
# TODO: check if the file that sam sent is the same as the xls
# TODO: check missing values

# ------------------------------ Read data ------------------------------
df_csv = pd.read_csv(loaddir+subject+'/fit_csv/cycling_'+date.strftime("%Y-%m-%d")+'_'+time.strftime('%H-%M-%S')+'.csv')
#df_laps = pd.read_csv(loaddir+subject+'/fit_csv/cycling_'+date.strftime("%Y-%m-%d")+'_'+time.strftime('%H-%M-%S')+'_laps.csv') #empty
#df_starts = pd.read_csv(loaddir+subject+'/fit_csv/cycling_'+date.strftime("%Y-%m-%d")+'_'+time.strftime('%H-%M-%S')+'_starts.csv') #empty
df_xls = pd.read_excel(loaddir+subject+'/fit_files/SB_'+date.strftime("%d%m%Y")+'_sample data.xlsx')

# ------------------------------ Clean data ------------------------------
df_csv.drop('fractional_cadence', axis=1, inplace=True) # fractional_cadence is empty

# Drop similar features
df_csv.drop(['enhanced_speed', 'enhanced_altitude'], axis=1, inplace=True)

# Drop empty first row
df_xls.drop(0, inplace=True)
df_xls.reset_index(drop=True, inplace=True)
#TODO: check whether things are better overlapping now

# 'None's and NaNs in heart_rate and cadence
df_csv.replace('None', np.nan, inplace=True)
df_csv[['heart_rate', 'cadence']] = df_csv[['heart_rate', 'cadence']].astype(float)

# Replace random datetime objects in balance column
for i in df_xls.index:
	if type(df_xls.balance[i]) is dt.datetime:
		df_xls.balance[i] = np.nan

# Convert to Timestamp
df_csv.timestamp = pd.to_datetime(df_csv.timestamp)
df_xls['timestamp'] = df_csv.timestamp[0] + pd.to_timedelta(df_xls.elapsedtime.astype(str))

# Rename columns of csv to xls
df_csv.rename(columns={'position_lat':'latitude', 'position_long':'longitude', 
	'heart_rate':'heartrate', 'distance':'elapseddistance'}, inplace=True)

# Set index 
df_xls.set_index('timestamp', drop=True, inplace=True)
df_csv.set_index('timestamp', drop=True, inplace=True)

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

# ------------------------------ Feature engineering ------------------------------
# Transform balance to feature
df_xls[['balanceleft', 'balanceright']] = df_xls.balance.str.split('/', expand=True).astype(float)
# Create ratio features TODO: note correlation here
df_xls['balanceratio'] = df_xls.balanceleft / df_xls.balanceright
df_xls['effectivenessratio'] = df_xls.effectivenessleft / df_xls.effectivenessright
df_xls['smoothnessratio'] = df_xls.smoothnessleft / df_xls.smoothnessright

df_xls.replace([-np.inf, np.inf], np.nan, inplace=True)

df_xls.drop('balance', axis=1, inplace=True)

# Calculate derivatives
diff_xls = np.array(['@ascent', 'elapseddistance', '@calories'])
for col in diff_xls:
	df_xls[col+'_diff'] = df_xls[col].diff()

# Select features
features_xls = np.array([['latitude', 'longitude', 'elevation', '@ascent', 'elapseddistance', 'temperature'],
						['speed', 'cadence', 'power', 'heartrate', '@calories', '@glucose'],
						['balanceleft', 'balanceright', 'effectivenessleft', 'effectivenessright', 'smoothnessleft', 'smoothnessright'],
						['balanceratio', 'effectivenessratio', 'smoothnessratio', '@grade', '@gps_accuracy', '@battery_soc']])
df_xls = df_xls[features_xls.flatten()]

features_csv = np.array([['latitude', 'longitude', 'altitude', 'temperature'], 
						['elapseddistance', 'speed', 'cadence', 'heartrate']])
df_csv = df_csv[features_csv.flatten()]

# Smoothing
smooth_xls = np.array(['speed', 'cadence', 'power', 'balanceleft', 'balanceright', 
	'effectivenessleft', 'effectivenessright', 'smoothnessleft', 'smoothnessright', '@grade'])

#df_xls[smooth_xls].rolling(window=DateOffset(seconds=10), center=True).mean() #TODO: if we have missing observations
df_xls[[i+'_rm' for i in smooth_xls]] = df_xls[smooth_xls].rolling(window=100, center=True).mean()
df_xls[[i+'_ewm' for i in smooth_xls]] = df_xls[smooth_xls].ewm(alpha=.01).mean()

for i in smooth_xls:
	p.plot_smooth_individual(df_xls, i, ['ewm', 'rm'])

p.plot_smooth_subplots(df_xls, smooth_xls, ['ewm', 'rm'], (2,5), 'xls_long')
p.plot_smooth_subplots(df_xls, smooth_xls, ['ewm', 'rm'], (5,2), 'xls_wide')

# ------------------------------ Plotting ------------------------------
plot_timeseries_parasite(df_csv, ['altitude', 'speed', 'heartrate', 'cadence'],
	ylabels = ["Altitude (m)", "Speed (km/h?)", "Heartrate (bpm)", "Cadence"],
	ylims = [(150,600), (0,130), (0,180), (0,150)], 
	axlocs = ["left", "right", "right", "right"],
	lws = [.7, .7, .7, .7],
	alphas = [.7, .7, .7, .2],
	savetext='csv')

plot_timeseries_parasite(df_xls[list(features_xls.flatten())+['glucose']], 
	['elevation', 'speed', 'heartrate', 'cadence', 'power', 'glucose'],
	ylabels = ["Altitude (m)", "Speed (km/h?)", "Heartrate (bpm)", "Cadence", "Power", "Glucose"],
	ylims = [(150,600), (0,130), (0,180), (0,150), (0,1200), (100,400)],
	axlocs = ["left", "right", "right", "right", "right", "left"],
	lws = [.7, .7, .7, .7, .7, 2.],
	alphas = [.7, .7, .7, .2, .2, 1.],
	figsize=(15,4), savetext='xls')

plot_timeseries_parasite(df_xls, ['elevation', 'speed', 'heartrate', 'cadence', 'power', 'glucose'],
	ylabels = ["Altitude (m)", "Speed (km/h?)", "Heartrate (bpm)", "Cadence", "Power", "Glucose"],
	ylims = [(150,700), (0,140), (0,180), (0,150), (0,1500), (100,400)],
	axlocs = ["left", "right", "right", "right", "right", "left"],
	lws = [.8, .8, .8, .8, .8, 2.],
	alphas = [.7, .05, .7, .05, .05, 1.],
	figsize=(15,5), savetext='xls_tall_ewm', legend=False)

# Plot distributions
p.plot_feature_distr_subplots(df_xls, features_xls, (18,10), 'xls')
p.plot_feature_distr_subplots(df_csv, features_csv, (10,6), 'csv')

# Plot timeseries
p.plot_feature_timeseries_subplots(df_xls, features_xls, (18,10), 'xls')
p.plot_feature_timeseries_subplots(df_csv, features_csv, (10,6), 'csv')

# Scatter timeseries
p.plot_feature_timeseries_subplots(df_xls, features_xls, (18,10), 'xls', scatter=True)
p.plot_feature_timeseries_subplots(df_csv, features_csv, (10,6), 'csv', scatter=True)

# ------------------------------ Statistical Tests ------------------------------
# Check for autocorrelation
plot_acf(df_xls['balanceleft'], lags=1000) ; plt.show()

# Check for stationarity
# TODO: Dickey-Fuller test https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775

# ------------------------------ Preliminary analysis ------------------------------
# TODO: check window functions

# These analysis are for univariate time-series prediction
# Moving average
# (next observation is the mean of all past observations - window function)

# Exponential smoothing
# (similar to moving average, but with decreasing weight assigned to past observations)
# double: when there is a trend
# triple: seasonality

# SARIMA (Seasonal autoregressive integrated moving average model)

# Prophet?

# TOOD: train test split / walk_forward validation??

model_features_xls = ['elevation', '@ascent_diff', 'elapseddistance_diff']
df_xls.columns.drop(['latitude', 'longitude'])
