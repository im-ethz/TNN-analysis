import os

import numpy as np
import scipy as sp
import pandas as pd

import datetime
import gc

import matplotlib
matplotlib.use('Agg')
from pandas_profiling import ProfileReport

from plot import *
from helper import *

verbose = 1

path = 'Data/TrainingPeaks+Dexcom/'
savepath = 'Descriptives/'

# -------------------------- Read data
df = {	'1min': pd.read_csv(path+'trainingpeaks_dexcom_left_1min.csv', header=[0,1], index_col=[0,1]),
		'5min':	pd.read_csv(path+'trainingpeaks_dexcom_left_5min.csv', header=[0,1], index_col=[0,1]),
		'sess': pd.read_csv(path+'trainingpeaks_dexcom_sess.csv', header=[0,1], index_col=[0,1])}

for k in df.keys():
	df[k].columns = [i[0]+'_'+i[1] if i[1] != 'first' and i[1] != '' else i[0] for i in df[k].columns]
	if k == 'sess':
		df[k].drop('file_id', axis=1, inplace=True)
	df[k].reset_index(inplace=True)
	if k != 'sess':
		df[k].local_timestamp = pd.to_datetime(df[k].local_timestamp)

# -------------------------- Create pandas profiling report
for k in df.keys():
	profile = ProfileReport(df[k], title='pandas profiling report', minimal=True)
	profile.to_file(savepath+'report_%s.html'%k)

	for i in df[k].RIDER.unique():
		profile = ProfileReport(df[k][df[k].RIDER == i], title='pandas profiling report', minimal=True)
		profile.to_file(savepath+'report_%s_%s.html'%(k,i))

# -------------------------- Read training session data
df = pd.read_csv(path+'trainingpeaks_dexcom_sess_stats.csv', header=[0,1], index_col=[0,1])
df.loc[:, ('local_timestamp', 'first')] = pd.to_datetime(df.loc[:, ('local_timestamp', 'first')])
df.loc[:, ('local_timestamp', 'last')] = pd.to_datetime(df.loc[:, ('local_timestamp', 'last')])

df.columns = [i[0]+'_'+i[1] if (i[1] != 'first' and i[1] != '' and not 'Unnamed' in i[1]) or i[0] == 'local_timestamp' else  i[0] for i in df.columns]
df.rename(columns={'time_training_max':'length_training'}, inplace=True)

df.drop('file_id', axis=1, inplace=True)
df.reset_index(inplace=True)

# -------------------------- Training length
matplotlib.use('TkAgg')

sns.distplot(df.length_training/60, kde=False, bins=100)
plt.xlim((-5, 600))
plt.xlabel('Length training session (min)')
plt.savefig(savepath+'length_training (min).pdf', bbox_inches='tight')
plt.savefig(savepath+'length_training (min).png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# -------------------------- TP Availability
df['date'] = df.local_timestamp_first.dt.date

training_count = df.groupby(['RIDER', 'date'])['file_id'].nunique().unstack().fillna(0)

#TODO: there is a training file from after 1-12-2019 in there
training_count.drop(datetime.date(2019,12,1), axis=1, inplace=True)

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(training_count, cmap='Blues', vmax=2, cbar_kws=dict(extend='max', label='number of training sessions per day'))
fig.axes[1].collections[0].cmap.set_over('orange') 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig('Descriptives/trainingpeaks_availability_day_all.pdf', bbox_inches='tight')
plt.savefig('Descriptives/trainingpeaks_availability_day_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# -------------------------- Correlation plots
corr_glucose = pd.DataFrame(index=df.columns)
corr_glucose['mean'] = df.corrwith(df['Glucose Value (mg/dL)_mean'])
corr_glucose['std'] = df.corrwith(df['Glucose Value (mg/dL)_std'])
corr_glucose.drop(['Glucose Value (mg/dL)_sum', 'Glucose Value (mg/dL)_median', 'Glucose Value (mg/dL)_amin', 'Glucose Value (mg/dL)_amax', 
					'glucose_mean', 'glucose_median', 'glucose_std', 'glucose_amin', 'glucose_amax'], inplace=True)

th_corr = 0.2
cols_corr = corr_glucose[(corr_glucose.abs() > th_corr).any(axis=1)].index
df[cols_corr]

PlotData(savepath, savetext='sess').plot_feature_correlation(df, cols_corr, cols_corr, np.arange(len(cols_corr)))

col_features = ['altitude', 'distance', 'speed', 'grade',
				'power', 'combined_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness',
				'cadence', 'left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 
				'temperature', 'heart_rate', 'glucose', 'Glucose Value (mg/dL)']
