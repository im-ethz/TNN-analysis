import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *
from calc import *

path = 'Data/Dexcom/'
path_trainingpeaks = 'Data/TrainingPeaks/2019/clean2/'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path_trainingpeaks)])

df_glucose = pd.read_csv(path+'dexcom_clean.csv', index_col=0)
df_glucose['local_timestamp'] = pd.to_datetime(df_glucose['local_timestamp'])

# select glucose measurements
df_glucose = df_glucose[((df_glucose['Event Type'] == 'EGV') | (df_glucose['Event Type'] == 'Calibration'))]

# -------------------------- Glucose availability
# create calendar with glucose availability
dates_2019 = pd.date_range(start='12/1/2018', end='11/30/2019').date
glucose_avail = pd.DataFrame(index=df_glucose.RIDER.unique(), columns=dates_2019)
for i in df_glucose.RIDER.unique():
	glucose_avail.loc[i, df_glucose[df_glucose.RIDER == i].local_timestamp.dt.date.unique()] = 1
glucose_avail.fillna(0, inplace=True)

# plot glucose availability per day
plt.figure(figsize=(25,10))
ax = sns.heatmap(glucose_avail, cmap='Blues', cbar=False, linewidths=.01) 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig('Descriptives/glucose_availability_day_all.pdf', bbox_inches='tight')
plt.savefig('Descriptives/glucose_availability_day_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# aggregate to month level
glucose_avail.columns = pd.MultiIndex.from_arrays([pd.to_datetime(glucose_avail.columns).month, glucose_avail.columns])
glucose_avail = glucose_avail.T.groupby(level=0).sum().T
glucose_avail = glucose_avail[[glucose_avail.columns[-1]] + list(glucose_avail.columns[:-1])]
glucose_avail.columns = glucose_avail.columns.map(month_mapping)

# plot glucose availability per month
sns.heatmap(glucose_avail, annot=True, linewidth=.5, cmap='Greens', cbar_kws={'label': 'days'})
plt.xlabel('month')
plt.ylabel('rider')
plt.savefig('Descriptives/glucose_availability_month_all.pdf', bbox_inches='tight')
plt.savefig('Descriptives/glucose_availability_month_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# measurements per day per athlete
df_glucose['date'] = df_glucose['local_timestamp'].dt.date

glucose_count = df_glucose.groupby(['RIDER', 'date'])['local_timestamp'].nunique()

glucose_count = pd.merge(glucose_count, 
	df_glucose[df_glucose['Event Type'] == 'EGV'].groupby(['RIDER', 'date'])['local_timestamp'].nunique().rename('egv'),
	how='left', on=['RIDER', 'date'], validate='one_to_one')

glucose_count.index.set_levels(glucose_count.index.levels[1].date, level=1, inplace=True)

glucose_count = pd.merge(glucose_count, 
	df_glucose[df_glucose['Event Type'] == 'Calibration'].groupby(['RIDER', 'date'])['local_timestamp'].nunique().rename('calibration'),
	how='left', on=['RIDER', 'date'], validate='one_to_one')

max_readings = 24*60/5

print("Days with more than the max number of readings:\n",
	glucose_count[glucose_count['egv'] > max_readings])
print("Day after days with more than the max number of readings:\n",
	glucose_count[(glucose_count['egv'] > max_readings).shift().fillna(False)])
"""
overlap: 
6 2018-12-09
6 2019-03-22
6 2019-10-02
10 2019-04-18
"""
glucose_avail_perc = glucose_count['local_timestamp'].unstack() / max_readings
glucose_avail_perc.fillna(0, inplace=True)

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(25,10))
ax = sns.heatmap(glucose_avail_perc, cmap='Blues', vmax=1, cbar_kws=dict(extend='max', label='fraction of possible CGM readings per day'))
fig.axes[1].collections[0].cmap.set_over('orange') 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig('Descriptives/glucose_availability_day_perc_all.pdf', bbox_inches='tight')
plt.savefig('Descriptives/glucose_availability_day_perc_all.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# TODO: 70% selection?


# -------------------------- Select parts before, during and after training sessions
# TODO all per athlete
# TODO split up athlete, and time-ranges (i.e. during training, nocturnal, etc.)

df_training = {}
for i in athletes:
	df =  pd.read_csv(path_trainingpeaks+str(i)+'/'+str(i)+'_data.csv', index_col=0)
	df_training[i] = df.groupby('file_id')['local_timestamp'].agg(['first', 'last'])
df_training = pd.concat(df_training)
df_training = df_training.reset_index().rename(columns={'level_0':'athlete'})

# df glucose all
df_glucose

# ------------- glucose during training sessions
ts_training = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=first, end=last, freq='s')]).to_frame(index=False, name=['RIDER', 'local_timestamp']) \
	for _, (athlete, _, first, last) in df_training.iterrows()])

df_glucose_training = pd.merge(df_glucose, ts_training, how='inner', on=['RIDER','local_timestamp'], validate='one_to_one')
df_glucose_training.to_csv('Data/Dexcom/dexcom_clean_training.csv')

# ------------- glucose 4h after training session
ts_after = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=last, periods=4*3600, freq='s')]).to_frame(index=False, name=['RIDER', 'local_timestamp']) \
	for _, (athlete, _, first, last) in df_training.iterrows()])

# TODO: ask if this is the right approach to this division
# remove timestamps that are included in training sessions
ts_training['one'] = 1
ts_after = ts_after.merge(ts_training, how='left', on=['RIDER', 'local_timestamp'])
ts_after = ts_after[ts_after.one.isna()]
ts_training.drop('one', axis=1, inplace=True)
ts_after.drop('one', axis=1, inplace=True)

# drop duplicates
ts_after.drop_duplicates(inplace=True)

df_glucose_after = pd.merge(df_glucose, ts_after, how='inner', on=['RIDER','local_timestamp'], validate='one_to_one')
df_glucose_after.to_csv('Data/Dexcom/dexcom_clean_after.csv')

# ------------- glucose during the wake part of the day of the training sessions
df_training['first'] = pd.to_datetime(df_training['first'])
df_training['last'] = pd.to_datetime(df_training['last'])

ts_wake = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=datetime.datetime.combine(first.date(), datetime.time(6,0)), 
				  end=datetime.datetime.combine(first.date(), datetime.time(23,59,59)), freq='s')]).to_frame(index=False, name=['RIDER', 'local_timestamp']) \
	for _, (athlete, _, first, last) in df_training.iterrows()])

ts_wake.drop_duplicates(inplace=True)

df_glucose_wake = pd.merge(df_glucose, ts_wake, how='inner', on=['RIDER','local_timestamp'], validate='one_to_one')
df_glucose_wake.to_csv('Data/Dexcom/dexcom_clean_wake.csv')

# ------------- glucose during the sleep part of the day of the training sessions
ts_sleep = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=datetime.datetime.combine(first.date()+datetime.timedelta(days=1), datetime.time(0)), 
				  end=datetime.datetime.combine(first.date()+datetime.timedelta(days=1), datetime.time(5,59,59)), freq='s')]).to_frame(index=False, name=['RIDER', 'local_timestamp']) \
	for _, (athlete, _, first, last) in df_training.iterrows()])

ts_sleep.drop_duplicates(inplace=True)

df_glucose_sleep = pd.merge(df_glucose, ts_sleep, how='inner', on=['RIDER','local_timestamp'], validate='one_to_one')
df_glucose_sleep.to_csv('Data/Dexcom/dexcom_clean_sleep.csv')

# ------------- glucose on the entire day of the training session
ts_day = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=datetime.datetime.combine(first.date(), datetime.time(0)), 
				  end=datetime.datetime.combine(first.date(), datetime.time(23,59,59)), freq='s')]).to_frame(index=False, name=['RIDER', 'local_timestamp']) \
	for _, (athlete, _, first, last) in df_training.iterrows()])

ts_day.drop_duplicates(inplace=True)
df_glucose_day = pd.merge(df_glucose, ts_day, how='inner', on=['RIDER','local_timestamp'], validate='one_to_one')
df_glucose_day.to_csv('Data/Dexcom/dexcom_clean_day.csv')

# -------------------------- Descriptives

# TODO: also plot extremes

# select glucose data only on the days that there is a training

# descriptives
inclusion: minimum of 14 consecutive days of data with approximately 70% of possible CGM readings over those 14 days

calculate metrics for the following blocks:
- sleep (24-6)
- wake (6-24)
- all
- trainingpeaks

mean glucose
percentage time in hypo L2, L1, target, hyper L1, L2
glycemic variability (CV and SD)
eA1c (estimated HbA1c)

