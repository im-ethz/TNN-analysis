# GOAL
# We know where the players were at dates from the trainingpeaks data
# We know when they changed timezones with the dexcom devices
# Aim is to find out the actual travelling times, so we know exactly what the rider's own local time was

# Note that we cannot simply take the changes of dexcom because:
# - not all changes happen automatically, so sometimes they have to change the timezone manually (depending on the device)
#   and this change does not take place when they actually travelled
# - some changes also happen because of a device change instead of travelling

# PLAN
# We take the changes that we observe in the dexcom data
# Then we filter out the changes that were made because of device changes, and any other changes
# that most likely did not happen in real life.
# Then we connect each change in dexcom to a change in TrainingPeaks data.
# Sometimes it happens that a change (or timezone) in TrainingPeaks is not linked to any change (or timezone) in dexcom.
# This happens often because the dexcom is censored (not recording) at that time.

# We check for each change, if it was within the range that we know they travelled from the TrainingPeaks data
# If any changes due to intermediate stops happen in the dexcom, it is most likely reset automatically
# In addition to that, if all changes happen within the change timeframe of the TrainingPeaks data,
# the timezone changes most likely happen automatically as well 
# (unless the person is really good at resetting his phone while travelling, but that is unlikely).
# If the timezone changes happen later than in TrainingPeaks, this is normal, as they most likely 
# simply changed the clock a few days after they arrived at the destination.
# This means that they reset manually.
# If the timezone changes happen earlier than in TrainingPeaks, this either means that the data in 
# TrainingPeaks is wrong (i.e. maybe the date of one of the devices is wrong there),
# or the person anticipated the timezone change before travelling and already reset it. 
# In any case this is something that needs to be looked into.
# If the device starts with PL8 or ML, it is a receiver from Dexcom. As these devices do not have
# internet connection, their clock has to be set manually.

# For any changes that we know happened manually, find out what the actual change was. 
# This is done in three steps:
# - Look at the timewindow from the trainingpeaks data
# - Go online to TrainingPeaks and check if there was any travelling registered there or if there is
#   a day on which they did not cycle within the time window that we know.
# - Check if there is a gap in the dexcom data from not recording. I think it can happen that they 
#   take of the dexcom while travelling. It is also adviced not to go with the dexcom through an 
#   X-ray, so it could be that they simply take it off. Another option is maybe that there is no 
#   recording because there is no (bluetooth) connection with the phone?
#   If we don't know, we should guess the date to be somewhere in the middle of the time window.

# https://www.flightera.net/en/flight/KLM+Royal+Dutch+Airlines-Amsterdam-Tokyo/KL861/Oct-2019#flight_list
# to find flights

import os
import gc
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd

from plot import *
from helper import *
from calc import *
from config import rider_mapping

path = './'
path_dc = '../Dexcom/'
path_tp = '../TrainingPeaks/2019/timezone/'

if not os.path.exists(path+'timezone/'):
	os.mkdir(path+'timezone/')


# ------------------------- Create file with changes observed from dexcom -------------------------
# trainingpeaks
df_tp = pd.read_csv(path_tp+'timezone_final.csv', index_col=[0,1])
df_tp.timezone = pd.to_timedelta(df_tp.timezone)
df_tp.local_timestamp_min = pd.to_datetime(df_tp.local_timestamp_min)
df_tp.local_timestamp_max = pd.to_datetime(df_tp.local_timestamp_max)
df_tp.timestamp_min = pd.to_datetime(df_tp.timestamp_min)
df_tp.timestamp_max = pd.to_datetime(df_tp.timestamp_max)
df_tp.date_min = pd.to_datetime(df_tp.date_min).dt.date
df_tp.date_max = pd.to_datetime(df_tp.date_max).dt.date

# dexcom
df = pd.read_csv(path_dc+'dexcom_utc.csv', index_col=0)
df.local_timestamp = pd.to_datetime(df.local_timestamp)
df.timestamp = pd.to_datetime(df.timestamp)

df['censoring'] = df.timestamp.diff()

# rider change
df['rider_change'] = df.RIDER.diff() != 0

# device change
df['device_change'] = df['Source Device ID'] != df['Source Device ID'].shift(1)
df.loc[df['rider_change'], 'device_change'] = np.nan

# transmitter change
df['transmitter_change'] = df['Transmitter ID'] != df['Transmitter ID'].shift(1)
df.loc[df['rider_change'], 'transmitter_change'] = np.nan

# shift to get min timestamp
df['local_timestamp_shift'] = df['local_timestamp'].shift(1)
df['timestamp_shift'] = df['timestamp'].shift(1)

# calculate timezone from dexcom
df['timezone'] = (df['local_timestamp'] - df['timestamp']).round('min')
df['timezone_change'] = (df['timezone'].diff() != '0s') & df['timezone'].notna() 
df.loc[df['rider_change'], 'timezone_change'] = True # note here true, so that we get this change as well

df_changes = df.loc[df['timezone_change'], ['RIDER', 'timezone', 'timestamp', 'timestamp_shift', 'local_timestamp', 'local_timestamp_shift', 'Source Device ID', 'Transmitter ID', 'device_change', 'transmitter_change']]

df_changes['trust_time'] = (df['Source Device ID'].str[:3] != 'PL8') & (df['Source Device ID'].str[:3] != 'MX9')

# get min and max timestamp
df_changes[['timestamp_shift', 'local_timestamp_shift']] = df_changes.groupby('RIDER')[['timestamp_shift', 'local_timestamp_shift']].shift(-1)
df_changes.rename(columns={	'timestamp'				:'timestamp_min',
							'timestamp_shift'		:'timestamp_max',
							'local_timestamp'		:'local_timestamp_min',
							'local_timestamp_shift'	:'local_timestamp_max'}, inplace=True)

# reset index
df_changes['n'] = df_changes.groupby('RIDER').cumcount()
df_changes.reset_index(drop=False, inplace=True)
df_changes.set_index(['RIDER', 'n'], inplace=True)

# fill in last timestamp
n_max = df_changes.groupby('RIDER')['timezone'].count()
for i in df.RIDER.unique():
	df_changes.loc[(i,n_max[i]-1), 'timestamp_max'] = df.loc[(df.RIDER == i), 'timestamp'].iloc[-1]
	df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max'] = df.loc[(df.RIDER == i), 'local_timestamp'].iloc[-1]
	if pd.isnull(df_changes.loc[(i,n_max[i]-1), 'timestamp_max']):
		df_changes.loc[(i,n_max[i]-1), 'timestamp_max'] = df.loc[(df.RIDER == i) & (df['Event Type'] == 'EGV'), 'timestamp'].iloc[-1]
		df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max'] = df.loc[(df.RIDER == i) & (df['Event Type'] == 'EGV'), 'local_timestamp'].iloc[-1]

	print(i, df_changes.loc[(i,n_max[i]-1), 'timestamp_max'], df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max'])

df_changes.to_csv(path+'timezone/dexcom_changes.csv')

# ------------------------- Check automatic or manual changes -------------------------

def delete_rows(df, i, n):
	df.loc[(i,n[0]-1), 'timestamp_max'] = df.loc[(i,n[-1]), 'timestamp_max']
	df.loc[(i,n[0]-1), 'local_timestamp_max'] = df.loc[(i,n[-1]), 'local_timestamp_max']

	df.drop([(i,j) for j in n], inplace=True)
	return df

def keep_second(df, i, n):
	df.loc[(i,n[0]), 'timezone'] = df.loc[(i,n[1]), 'timezone']
	df.loc[(i,n[0]), 'timestamp_max'] = df.loc[(i,n[1]), 'timestamp_max']
	df.loc[(i,n[0]), 'local_timestamp_max'] = df.loc[(i,n[1]), 'local_timestamp_max']
	df.loc[(i,n[0]), 'Source Device ID'] = df.loc[(i,n[1]), 'Source Device ID']
	df.drop((i,n[1]), inplace=True)
	return df

def replace_row(dc, tp, i, n_dc:int, n_tp:list, keep_tz=False):
	row = dc.loc[(i,n_dc)]
	next_rows = dc[(dc.index.get_level_values(0) == i) & (dc.index.get_level_values(1) > n_dc)]

	dc.drop([(i,n_dc)], inplace=True)
	dc.drop(next_rows.index, inplace=True)

	for j, n in enumerate(n_tp):
		dc.loc[(i,n_dc+j), 'timezone'] = tp.loc[(i,n), 'timezone']
		if j == 0 and keep_tz:
			dc.loc[(i,n_dc+j), 'timestamp_min'] = row['timestamp_min']
			dc.loc[(i,n_dc+j), 'local_timestamp_min'] = row['local_timestamp_min']
		else:
			dc.loc[(i,n_dc+j), 'timestamp_min'] = tp.loc[(i,n), 'timestamp_min']
			dc.loc[(i,n_dc+j), 'local_timestamp_min'] = tp.loc[(i,n), 'local_timestamp_min']

		if j == len(n_tp) - 1 and keep_tz:
			dc.loc[(i,n_dc+j), 'timestamp_max'] = row['timestamp_max']
			dc.loc[(i,n_dc+j), 'local_timestamp_max'] = row['local_timestamp_max']
		else:
			dc.loc[(i,n_dc+j), 'timestamp_max'] = tp.loc[(i,n), 'timestamp_max']
			dc.loc[(i,n_dc+j), 'local_timestamp_max'] = tp.loc[(i,n), 'local_timestamp_max']

	for (i,n), n_row in next_rows.iterrows():
		dc.loc[(i,n+len(n_tp)-1),:] = n_row

	return dc

def check_minmax(dc, tp, n_dc, n_tp=None):
	# Note: it is not a problem if the dexcom is changed a few days later
	# This is common, as sometimes the riders need to make their changes manually
	# However, it would be weird maybe if the dexcom is changed earlier than the actual travel change
	# (Although this could also happen if they're already anticipating the timezone changed, and are doing
	# it while travelling..)
	# returns [whether dexcom min is not changed earlier than TP,
	# 		   whether dexcom max is not changed earlier than TP]
	# Note: it is only a problem if the previous one max is true and the current one min is true
	if n_tp is None:
		n_tp = n_dc
	print(n_dc[1], n_tp[1],
		" later ", 
		dc.loc[n_dc, 'timestamp_min'].date() > tp.loc[n_tp, 'date_min'],
		dc.loc[n_dc, 'timestamp_max'].date() > tp.loc[n_tp, 'date_max'],
		" earlier ",
		dc.loc[n_dc, 'timestamp_min'].date() < tp.loc[n_tp, 'date_min'],
		dc.loc[n_dc, 'timestamp_max'].date() < tp.loc[n_tp, 'date_max'])

# ------------------------- rider 1
# device changes (that were still set back in time)
n_del = df_changes[df_changes['device_change'] == 1].loc[1].index
df_changes = delete_rows(df_changes, 1, n_del)

# CHECK if timezones are correct
# timezones are correct, nothing missing

# recalculate n
df_changes.reset_index(inplace=True)
df_changes['n'] = df_changes.groupby('RIDER').cumcount()
df_changes.set_index(['RIDER', 'n'], inplace=True)
n_max = df_changes.groupby('RIDER')['timezone'].count()

# CHECK if timestamp of timezone changes are correct
for n in range(n_max[1]):
	check_minmax(df_changes, df_tp, (1,n))
# later happens quite a few times

# CONCLUSION timezone change manually

# ------------------------- rider 2
# CHECK if timezones are correct
# 2,9 is censored in dexcom

# CHECK if timestamp of timezone changes are correct
for n in range(n_max[2]):
	check_minmax(df_changes, df_tp, (2,n))

# CONCLUSION timezone change automatically
# We don't need a manual lookup

# ------------------------- rider 3
# CHECK if timezones are correct

# ['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']
# Missing travelling from Spain to Ireland and back to Spain between 2018-12-16 and 2019-12-19
df.loc[(df.RIDER == 3) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-01 04:41:40') & (df.timestamp <= '2019-03-31 00:52:42')]
df.loc[144743:144744, 'timestamp']
# Note that this is because of censoring between 2018-12-12 and 2019-01-14

# Missing travelling from +2 to +8 and back between 2019-05-21 and 2019-06-02
df.loc[(df.RIDER == 3) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-31 00:00:00') & (df.timestamp <= '2019-06-18 00:00:00')]
df.loc[150197:150198, 'timestamp']
# Note that this is because of censoring between 2019-04-05 and 2019-06-11

# CHECK if timestamp of timezone changes are correct
check_minmax(df_changes, df_tp, (3,0), (3,0))
check_minmax(df_changes, df_tp, (3,0), (3,2))
check_minmax(df_changes, df_tp, (3,1), (3,3))
check_minmax(df_changes, df_tp, (3,1), (3,5))
for n in range(2,n_max[1]):
	check_minmax(df_changes, df_tp, (3,n), (3,n+4))

# CONCLUSION timezone change automatically
# Only between 6-10 and 7-11 is too late, but maybe there's a mistake with the TrainingPeaks file there?
# TODO: Note that the device around this change is again Garmin, and it seems that the timezone
# of the Garmin is simply not reset. (this is loc 1041 and 1042)

# ------------------------- rider 4
# something weird going on with the phone
# cannot travel from +11 to +13 to +01 to +11 again in 2 days
df_changes = delete_rows(df_changes, 4, [2,3,4])

# new device still at old timestamp
df_changes = delete_rows(df_changes, 4, [10]) # TODO: check

# alternating between receiver and phone (receiver still at old time)
df_changes = delete_rows(df_changes, 4, range(15,27))

# CHECK if timezones are correct

# Missing travelling
df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-04-06 00:00:00') & (df.timestamp <= '2019-09-04 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
df.loc[215628:215629, 'timestamp']
df.loc[217164:217165, 'timestamp']
# Note that this is because of censoring between:
# - 2019-04-15 and 2019-05-09
# - 2019-05-15 and 2019-09-04

# CHECK if timestamp of timezone changes are correct
check_minmax(df_changes, df_tp, (4,0), (4,0))
check_minmax(df_changes, df_tp, (4,1), (4,1))
check_minmax(df_changes, df_tp, (4,5), (4,2))
check_minmax(df_changes, df_tp, (4,6), (4,3))
check_minmax(df_changes, df_tp, (4,7), (4,4))
check_minmax(df_changes, df_tp, (4,8), (4,5))
check_minmax(df_changes, df_tp, (4,9), (4,6))
check_minmax(df_changes, df_tp, (4,11), (4,9))
check_minmax(df_changes, df_tp, (4,12), (4,10)) # NOTE
check_minmax(df_changes, df_tp, (4,13), (4,10)) # NOTE
check_minmax(df_changes, df_tp, (4,14), (4,11)) # NOTE
check_minmax(df_changes, df_tp, (4,27), (4,11)) # NOTE
check_minmax(df_changes, df_tp, (4,28), (4,12))

# CONCLUSION timezone change automatically (?)
# In any case check since there were so many small deviations

# ROUND OFF
# recalculate n
df_changes.reset_index(inplace=True)
df_changes['n'] = df_changes.groupby('RIDER').cumcount()
df_changes.set_index(['RIDER', 'n'], inplace=True)
n_max = df_changes.groupby('RIDER')['timezone'].count()

# round off (because the real time zones are also in whole hours)
for n in range(7,13):
	df_changes.loc[(4,n), 'timezone'] = df_changes.loc[(4,n), 'timezone'].round('h')


# ------------------------- rider 5

# probably typed in the wrong time on his iphone, as the time was only wrong for an hour or so
# (so in the course of one hour, changed between +9 to +10 to +9 again)
df_changes = delete_rows(df_changes, 5, [7,8])

# CHECK if timestamp of timezone changes are correct
for n in range(7):
	check_minmax(df_changes, df_tp, (5,n), (5,n))
check_minmax(df_changes, df_tp, (5,9), (5,7)) # NOTE
check_minmax(df_changes, df_tp, (5,10), (5,7)) # NOTE
# CONCLUSION timezone change automatically (?)

# ------------------------- rider 6
# alternating between receiver and phone (receiver still at old time)
df_changes = delete_rows(df_changes, 6, range(2,42))
df_changes = keep_second(df_changes, 6, [0,1])

# alternating between receiver and phone (receiver still at old time)
df_changes = delete_rows(df_changes, 6, range(44,48))
df_changes = keep_second(df_changes, 6, [42,43])

# CHECK if timezones are correct

# I think 6,2 of the TP data might be incorrect 
# (there is only one day where it switches from +11 to +10, so maybe unlikely?)

# 6,54: seems to have switched to the wrong timezone (1h forward instead of backwards)
# As the timezone from TP is obtained from the location, which I assume to be correct
# we have to change the timezone for this week in dexcom
df_changes.loc[(6,54), 'timezone'] = pd.to_timedelta('0 days 02:00:00')

# CHECK if timestamp of timezone changes are correct
check_minmax(df_changes, df_tp, (6,0), (6,0))
check_minmax(df_changes, df_tp, (6,42), (6,1)) # NOTE
check_minmax(df_changes, df_tp, (6,48), (6,1)) # NOTE
for n in range(48,59):
	check_minmax(df_changes, df_tp, (6,n), (6,n-45))	
# CONCLUSION timezone change manually (?)
# Note: changes manually from 6,49 onwards

# ROUND OFF
# recalculate n
df_changes.reset_index(inplace=True)
df_changes['n'] = df_changes.groupby('RIDER').cumcount()
df_changes.set_index(['RIDER', 'n'], inplace=True)
n_max = df_changes.groupby('RIDER')['timezone'].count()

# the rider reset it here with an extra couple of min, but the actual timezone change is of course just 1 h
for n in df_changes.loc[6].index:
	df_changes.loc[(6,n), 'timezone'] = df_changes.loc[(6,n), 'timezone'].round('h')


# ------------------------- rider 10
# ROUND OFF
# people with iPhones also have to do manual resets (according to Simon)
for n in df_changes.loc[10].index:
	df_changes.loc[(10,n), 'timezone'] = df_changes.loc[(10,n), 'timezone'].round('h')

# CHECK if timezones are correct

# Censoring between 01-12-2018 and 11-12-2018

# 10,37-10,39 It seems like he went on holiday here
# However, I think the +2 -2 change for two hours was incorrect, so change it back
df_changes = delete_rows(df_changes, 10, [38,39])

# CHECK if timestamp of timezone changes are correct
check_minmax(df_changes, df_tp, (10,0), (10,1))
check_minmax(df_changes, df_tp, (10,1), (10,2))
check_minmax(df_changes, df_tp, (10,2), (10,3)) # NOTE
check_minmax(df_changes, df_tp, (10,3), (10,3)) # NOTE
check_minmax(df_changes, df_tp, (10,4), (10,3)) # NOTE
check_minmax(df_changes, df_tp, (10,5), (10,4))
check_minmax(df_changes, df_tp, (10,6), (10,5)) # NOTE
check_minmax(df_changes, df_tp, (10,7), (10,5)) # NOTE
check_minmax(df_changes, df_tp, (10,8), (10,6))
check_minmax(df_changes, df_tp, (10,9), (10,7))
check_minmax(df_changes, df_tp, (10,10), (10,8))
check_minmax(df_changes, df_tp, (10,11), (10,9))
check_minmax(df_changes, df_tp, (10,12), (10,10))
check_minmax(df_changes, df_tp, (10,13), (10,11))
check_minmax(df_changes, df_tp, (10,14), (10,12))
check_minmax(df_changes, df_tp, (10,15), (10,13))
check_minmax(df_changes, df_tp, (10,16), (10,14))
check_minmax(df_changes, df_tp, (10,17), (10,15)) # NOTE
check_minmax(df_changes, df_tp, (10,18), (10,15)) # NOTE
check_minmax(df_changes, df_tp, (10,19), (10,16)) # NOTE
check_minmax(df_changes, df_tp, (10,20), (10,16)) # NOTE
check_minmax(df_changes, df_tp, (10,21), (10,17))
check_minmax(df_changes, df_tp, (10,22), (10,18))
check_minmax(df_changes, df_tp, (10,23), (10,19))
check_minmax(df_changes, df_tp, (10,24), (10,20))
check_minmax(df_changes, df_tp, (10,25), (10,21))
check_minmax(df_changes, df_tp, (10,26), (10,22))
check_minmax(df_changes, df_tp, (10,27), (10,23))
check_minmax(df_changes, df_tp, (10,28), (10,24))
check_minmax(df_changes, df_tp, (10,29), (10,25))
check_minmax(df_changes, df_tp, (10,30), (10,26))
check_minmax(df_changes, df_tp, (10,31), (10,27))
check_minmax(df_changes, df_tp, (10,32), (10,28))
check_minmax(df_changes, df_tp, (10,33), (10,29)) # NOTE
check_minmax(df_changes, df_tp, (10,34), (10,29)) # NOTE
check_minmax(df_changes, df_tp, (10,35), (10,30))
check_minmax(df_changes, df_tp, (10,36), (10,31))

# CONCLUSION timezone change automatically
# also because travel is often split up

# TODO: find out actual timezone change 
# 14-12 and 15-13
df.loc[424417:424427]
# censoring after the actual travel, that's why only the second date is later

# 23-19 and 24-20
df.loc[435536:435546]
# censoring after the actual travel, that's why only the second date is later

# ------------------------- rider 12
# According to the rider, everything went automatic, but that does not seem to be true.
# There are only a few changes in Dexcom, therefore we have to find out everything manually
"""
    index          timezone       timestamp_min       timestamp_max 
n                                                                   
0  457544   0 days 08:00:00 2018-11-30 16:00:23 2018-12-17 20:14:46 
1  461843   0 days 01:00:00 2018-12-17 20:19:46 2019-11-06 21:42:18 
2  483795 -1 days +18:00:00 2019-11-06 21:47:18 2019-11-19 07:46:49 
3  487106   0 days 01:00:00 2019-11-19 13:26:48 2019-11-30 22:56:23 
"""
# Holiday Costa Rica: 2018-10-15 to 2018-10-24
# It seems that is why he had his dexcom still on old costa rica time until 2018-12-17
# Therefore timezone from 2018-11-30 was simply +01:00 UTC
df_changes = keep_second(df_changes, 12, [0,1])

df_changes = replace_row(df_changes, df_tp, 12, 0, [0,1,2,3,4,5,6,7,8,9,10])

df_changes = delete_rows(df_changes, 12, [4,5,6,7,8,9])

# change to winter time that was not included in the data
df_changes.loc[(12,10), 'timezone'] -= pd.to_timedelta('1h') 

# ------------------------- rider 13
# Information from rider: 
# "I did not change the time zone always, many time I forget, sorry. 
# But I can not tell you exactly wich race were wrong."

# the rider reset it here with an extra couple of min, but the actual timezone change is of course just 1 h
for n in range(10):
	df_changes.loc[(13,n), 'timezone'] = df_changes.loc[(13,n), 'timezone'].round('h')

# CHECK if timezones are correct
df_changes = keep_second(df_changes, 13, [5,6])

# Insert some rows for which the timezone changes were not found
df_changes = replace_row(df_changes, df_tp, 13, 8, [7,8,9,10])

# Missing travelling
df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-27 00:00:00') & (df.timestamp <= '2019-10-03 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# Note that this is because of censoring between:
# - 2019-04-15 and 2019-05-09
# - 2019-05-15 and 2019-09-04

# CHECK if timestamp of timezone changes are correct
for n in range(6):
	check_minmax(df_changes, df_tp, (13,n))
check_minmax(df_changes, df_tp, (13,7), (13,6))
check_minmax(df_changes, df_tp, (13,8), (13,7))
check_minmax(df_changes, df_tp, (13,11), (13,10))
check_minmax(df_changes, df_tp, (13,12), (13,11))

# CONCLUSION timezone change manually, but NOTE always changed right away it seems
# Therefore, we don't have to find out the timezones (for the first half), 
# but we just check with TrainingPeaks if it seems to be correct
# For the second half, we do need to figure out the timezones, as he did not change them.

# ------------------------- rider 14
# Information from rider:
# "With dexcom receiver I almost never change the date/time unfortunately. 
# At the time it was solely to track my bg values. Sometimes I changed it if we did a big travel 
# (Europe to US for example) but not always. Summer/winter time I never touched that."

df_changes = replace_row(df_changes, df_tp, 14, 0, np.arange(10))
df_changes = replace_row(df_changes, df_tp, 14, 10, np.arange(10, 20))

df_changes = delete_rows(df_changes, 14, [5,6,7])
df_changes = delete_rows(df_changes, 14, [15,16])

# CONCLUSION timezone change manually

# ------------------------- rider 15

for n in range(8):
	df_changes.loc[(15,n), 'timezone'] = df_changes.loc[(15,n), 'timezone'].round('h')

# CHECK if timezones are correct
# missing travelling between 2019-01-27 and 2019-07-21 due to censoring
df.loc[(df.RIDER == 15) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-27 00:00:00') & (df.timestamp <= '2019-07-21 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]

# I think the last ones from 15,5 to 15,7 are incorrect
# check if we did not miss any change when he changed his transmitter

# He set the date 1 day back. 
# I hope it doesn't mean that the day was wrong in the first place, because then that would screw up many things
# TODO: figure this out!!!

# If he changed it on purpose (because he noticed the date was wrong before 31 October 2019)
# (and this would have meant that we actually started with timezone +1 days 01:00:00)
# then all dexcom UTC timestamps for this rider should be shifted back 1 day.
# If he changed it on accident, then we can assume everything before was correct,
# and we only have to manually edit the timezone in the dexcom_changes_manual file.

# For now, assume it is the last one, and that only the days after are wrong
# (then we don't have to change anything in the dexcom data)

# Change timezone 6 and 7 to 0 days instead of -1 days manually
# Merge row 5 and 6 manually

# In addition, we have no data on cycling during that period
# So now simply assume he travelled when he reset the device (which is most likely not entirely correct)

# TODO: if the timezone turns out to be incorrect, make sure to change it also in the glucose data

df_changes = delete_rows(df_changes, 15, [6])
df_changes.loc[(15,7), 'timezone'] += pd.to_timedelta('1days')

check_minmax(df_changes, df_tp, (15,0), (15,0))
check_minmax(df_changes, df_tp, (15,1), (15,7))
check_minmax(df_changes, df_tp, (15,2), (15,8))
check_minmax(df_changes, df_tp, (15,3), (15,9))
check_minmax(df_changes, df_tp, (15,4), (15,8))

# CONCLUSION timezone change manually


df_changes = df_changes.sort_index()

df_changes.to_csv(path+'timezone/dexcom_changes2.csv')
# Note: This file is used for the MANUAL addition of actual timezone changes underneath


# ------------------------- Find out actual changes manually -------------------------

# ------------------------- rider 1
# CONCLUSION timezone change manually

# CHECK find out actual changes by looking at gaps

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-02-13 00:00:00') & (df.timestamp <= '2019-02-13 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
          local_timestamp           timestamp       censoring        timezone     timestamp_shift
19065 2019-02-13 15:34:33 2019-02-13 14:34:33 0 days 04:20:00 0 days 01:00:00 2019-02-13 10:14:33
19070 2019-02-14 00:14:49 2019-02-13 23:14:49 0 days 08:20:17 0 days 01:00:00 2019-02-13 14:54:32
"""
# flight to rwanda is ~8h, maybe he had a stop in between?
df.loc[19069:19070, 'timestamp']
"""
19069   2019-02-13 14:54:32
19070   2019-02-13 23:14:49
"""

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-04 00:00:00') & (df.timestamp <= '2019-03-06 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
          local_timestamp           timestamp       censoring        timezone     timestamp_shift
23204 2019-03-04 15:38:38 2019-03-04 13:38:39 0 days 01:00:00 0 days 02:00:00 2019-03-04 12:38:39
23322 2019-03-05 10:38:36 2019-03-05 08:38:37 0 days 09:15:01 0 days 02:00:00 2019-03-04 23:23:36
23327 2019-03-05 12:48:36 2019-03-05 10:48:37 0 days 01:50:00 0 days 02:00:00 2019-03-05 08:58:37
23344 2019-03-05 18:53:36 2019-03-05 16:53:37 0 days 04:45:00 0 days 02:00:00 2019-03-05 12:08:37
23684 2019-03-07 01:18:31 2019-03-06 23:18:32 0 days 02:09:59 0 days 02:00:00 2019-03-06 21:08:33
"""
df.loc[23321:23322, 'timestamp']
"""
23321   2019-03-04 23:23:36
23322   2019-03-05 08:38:37
"""

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-22 00:00:00') & (df.timestamp <= '2019-05-25 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
empty
"""
# It says 2019-05-23 in TrainingPeaks online, therefore take (local ts 00:00:00):
# 2019-05-23 21:59:39
# 2019-05-23 22:04:39

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-31 00:00:00') & (df.timestamp <= '2019-06-03 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
          local_timestamp           timestamp       censoring        timezone     timestamp_shift
46778 2019-05-31 18:29:16 2019-05-31 10:29:17 0 days 23:14:56 0 days 08:00:00 2019-05-30 11:14:21
"""
df.loc[46777:46778, 'timestamp']
"""
46777   2019-05-30 11:14:21
46778   2019-05-31 10:29:17
"""

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-03 00:00:00') & (df.timestamp <= '2019-09-06 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
          local_timestamp           timestamp       censoring        timezone     timestamp_shift
72295 2019-09-03 20:36:57 2019-09-03 18:36:58 0 days 00:55:01 0 days 02:00:00 2019-09-03 17:41:57
72349 2019-09-04 01:21:55 2019-09-03 23:21:56 0 days 00:20:01 0 days 02:00:00 2019-09-03 23:01:55
72882 2019-09-05 22:41:52 2019-09-05 20:41:53 0 days 01:00:01 0 days 02:00:00 2019-09-05 19:41:52
"""
# online it seems like the travel day is 2019-09-05 as there is no cycling
# note that the gap is not long enough for the travelling, but maybe this is the time he went
# through security or something?
df.loc[72881:72882, 'timestamp']
"""
72881   2019-09-05 19:41:52
72882   2019-09-05 20:41:53
"""

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-22 00:00:00') & (df.timestamp <= '2019-09-24 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
          local_timestamp           timestamp       censoring        timezone     timestamp_shift
77213 2019-09-22 09:06:11 2019-09-22 01:06:12 0 days 01:25:00 0 days 08:00:00 2019-09-21 23:41:12
77274 2019-09-22 15:01:11 2019-09-22 07:01:12 0 days 00:55:00 0 days 08:00:00 2019-09-22 06:06:12
77319 2019-09-22 19:01:10 2019-09-22 11:01:11 0 days 00:20:01 0 days 08:00:00 2019-09-22 10:41:10
77531 2019-09-23 13:21:08 2019-09-23 05:21:09 0 days 00:45:00 0 days 08:00:00 2019-09-23 04:36:09
77572 2019-09-23 17:56:08 2019-09-23 09:56:09 0 days 01:14:58 0 days 08:00:00 2019-09-23 08:41:11
77614 2019-09-23 23:31:07 2019-09-23 15:31:08 0 days 02:09:59 0 days 08:00:00 2019-09-23 13:21:09
77677 2019-09-24 07:01:06 2019-09-23 23:01:07 0 days 02:19:58 0 days 08:00:00 2019-09-23 20:41:09
77679 2019-09-24 07:21:07 2019-09-23 23:21:08 0 days 00:15:00 0 days 08:00:00 2019-09-23 23:06:08
77709 2019-09-24 11:26:05 2019-09-24 03:26:06 0 days 01:39:58 0 days 08:00:00 2019-09-24 01:46:08
77713 2019-09-24 12:16:08 2019-09-24 04:16:09 0 days 00:35:03 0 days 08:00:00 2019-09-24 03:41:06
77724 2019-09-25 01:01:05 2019-09-24 17:01:06 0 days 11:54:59 0 days 08:00:00 2019-09-24 05:06:07
"""
df.loc[77723:77724, 'timestamp']
"""
77723   2019-09-24 05:06:07
77724   2019-09-24 17:01:06
"""

# ------------------------- rider 4

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-07 00:00:00') & (df.timestamp <= '2018-12-09 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
193734 2018-12-07 11:18:17 2018-12-07 10:18:17 0 days 02:19:59 0 days 01:00:00 2018-12-07 07:58:18
193752 2018-12-07 13:43:17 2018-12-07 12:43:17 0 days 01:00:00 0 days 01:00:00 2018-12-07 11:43:17
193802 2018-12-07 20:03:16 2018-12-07 19:03:16 0 days 02:15:00 0 days 01:00:00 2018-12-07 16:48:16
193819 2018-12-08 05:08:15 2018-12-08 04:08:15 0 days 07:44:59 0 days 01:00:00 2018-12-07 20:23:16
193824 2018-12-08 10:13:14 2018-12-08 09:13:14 0 days 04:44:58 0 days 01:00:00 2018-12-08 04:28:16
193829 2018-12-08 12:03:14 2018-12-08 11:03:14 0 days 01:30:00 0 days 01:00:00 2018-12-08 09:33:14
193928 2018-12-09 17:08:13 2018-12-09 16:08:11 0 days 20:54:58 0 days 01:00:00 2018-12-08 19:13:13
"""
df.loc[193927:193928, 'timestamp']
"""
193927   2018-12-08 19:13:13
193928   2018-12-09 16:08:11
"""
# note that he changed the dexcom earlier than when he actually travelled

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-16 00:00:00') & (df.timestamp <= '2019-01-18 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp        censoring        timezone     timestamp_shift
199179 2019-01-17 11:42:11 2019-01-17 00:42:09 17 days 18:19:00 0 days 11:00:00 2018-12-30 06:23:09
199232 2019-01-17 20:41:57 2019-01-17 09:41:55  0 days 04:39:51 0 days 11:00:00 2019-01-17 05:02:04
199238 2019-01-18 07:02:17 2019-01-18 06:01:54  0 days 19:54:59 0 days 01:00:00 2019-01-17 10:06:55
199256 2019-01-18 09:27:16 2019-01-18 08:26:53  0 days 01:00:00 0 days 01:00:00 2019-01-18 07:26:53
199295 2019-01-18 15:42:15 2019-01-18 14:41:52  0 days 03:05:00 0 days 01:00:00 2019-01-18 11:36:52
"""
df.loc[199237:199238, 'timestamp'] # note that this is the change we also found in dexcom
"""
199237   2019-01-17 10:06:55
199238   2019-01-18 06:01:54
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-02-19 00:00:00') & (df.timestamp <= '2019-02-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
204398 2019-02-19 11:15:47 2019-02-19 10:15:24 0 days 01:29:59 0 days 01:00:00 2019-02-19 08:45:25
204688 2019-02-20 13:10:45 2019-02-20 12:10:22 0 days 01:50:00 0 days 01:00:00 2019-02-20 10:20:22
204783 2019-02-20 22:00:44 2019-02-20 21:00:21 0 days 01:00:02 0 days 01:00:00 2019-02-20 20:00:19
204923 2019-02-21 21:10:41 2019-02-21 20:10:18 0 days 11:34:59 0 days 01:00:00 2019-02-21 08:35:19
205041 2019-02-22 11:10:39 2019-02-22 07:10:17 0 days 01:15:00 0 days 04:00:00 2019-02-22 05:55:17
205181 2019-02-22 23:25:37 2019-02-22 19:25:15 0 days 00:39:59 0 days 04:00:00 2019-02-22 18:45:16
"""
df.loc[204922:204923, 'timestamp']
"""
204922   2019-02-21 08:35:19
204923   2019-02-21 20:10:18
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-02 00:00:00') & (df.timestamp <= '2019-03-05 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
206391 2019-03-02 12:25:18 2019-03-02 08:24:56 0 days 09:30:00 0 days 04:00:00 2019-03-01 22:54:56
206392 2019-03-02 15:35:17 2019-03-02 11:34:55 0 days 03:09:59 0 days 04:00:00 2019-03-02 08:24:56
206461 2019-03-02 22:15:16 2019-03-02 18:14:54 0 days 01:00:00 0 days 04:00:00 2019-03-02 17:14:54
206490 2019-03-03 01:35:14 2019-03-02 21:34:52 0 days 00:59:59 0 days 04:00:00 2019-03-02 20:34:53
206541 2019-03-03 06:40:16 2019-03-03 02:39:54 0 days 00:55:01 0 days 04:00:00 2019-03-03 01:44:53
206558 2019-03-03 16:35:14 2019-03-03 12:34:52 0 days 08:34:59 0 days 04:00:00 2019-03-03 03:59:53
206598 2019-03-03 21:15:14 2019-03-03 17:14:52 0 days 01:25:02 0 days 04:00:00 2019-03-03 15:49:50
206926 2019-03-05 07:00:09 2019-03-05 05:59:47 0 days 09:29:57 0 days 01:00:00 2019-03-04 20:29:50
"""
# Online it says travel was on 2019-03-03
df.loc[206557:206558, 'timestamp']
"""
206557   2019-03-03 03:59:53
206558   2019-03-03 12:34:52
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-22 00:00:00') & (df.timestamp <= '2019-03-25 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
210172 2019-03-22 14:24:20 2019-03-22 13:23:58 0 days 04:24:58 0 days 01:00:00 2019-03-22 08:59:00
210264 2019-03-23 00:45:10 2019-03-22 23:44:48 0 days 02:45:50 0 days 01:00:00 2019-03-22 20:58:58
210364 2019-03-23 16:35:09 2019-03-23 15:34:47 0 days 07:34:59 0 days 01:00:00 2019-03-23 07:59:48
210393 2019-03-24 09:15:04 2019-03-24 08:14:42 0 days 14:19:56 0 days 01:00:00 2019-03-23 17:54:46
210398 2019-03-24 20:20:04 2019-03-24 19:19:42 0 days 10:44:59 0 days 01:00:00 2019-03-24 08:34:43
210415 2019-03-24 23:30:04 2019-03-24 22:29:42 0 days 01:50:00 0 days 01:00:00 2019-03-24 20:39:42
"""
# online it says 2019-03-24  8pm (land)
df.loc[210397:210398, 'timestamp']
"""
210397   2019-03-24 08:34:43
210398   2019-03-24 19:19:42
"""

# winter time to summer time (first sunday in April in Australia)
# change from +11 to +10 around 2019-04-06
# We did not have to do the step underneath, but note that it is funny that there is also some censoring
# exactly on the same day that there is a timezone change.
df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-04-06 00:00:00') & (df.timestamp <= '2019-04-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
213504 2019-04-07 08:14:31 2019-04-06 22:14:09 0 days 22:44:57 0 days 10:00:00 2019-04-05 23:29:12
"""
df.loc[213503:213504, 'timestamp']
"""
213503   2019-04-05 23:29:12
213504   2019-04-06 22:14:09
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-03 00:00:00') & (df.timestamp <= '2019-09-04 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp         censoring        timezone     timestamp_shift
217165 2019-09-04 06:38:47 2019-09-04 04:38:25 112 days 01:56:00 0 days 02:00:00 2019-05-15 02:42:25
217203 2019-09-04 15:53:02 2019-09-04 07:53:23   0 days 00:10:00 0 days 08:00:00 2019-09-04 07:43:23
217240 2019-09-04 20:28:02 2019-09-04 12:28:23   0 days 01:35:00 0 days 08:00:00 2019-09-04 10:53:23
"""
# online: 2019-09-04
# it seems he started using dexcom again after arriving in China
# there is no exact flight date
df.loc[217164:217165, 'timestamp']
"""
217164   2019-05-15 02:42:25
217165   2019-09-04 04:38:25
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-22 00:00:00') & (df.timestamp <= '2019-09-24 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
222222 2019-09-25 01:22:31 2019-09-24 17:22:52 0 days 01:50:01 0 days 08:00:00 2019-09-24 15:32:51
222230 2019-09-25 03:22:29 2019-09-24 19:22:50 0 days 01:24:59 0 days 08:00:00 2019-09-24 17:57:51
222241 2019-09-25 04:47:27 2019-09-24 20:47:48 0 days 00:35:00 0 days 08:00:00 2019-09-24 20:12:48
"""
# online: 2019-09-23 and 2019-09-24
# No big enough gap found, so we take 2019-09-24 00:00:00 local time
df.loc[(df.RIDER == 4) & (df.timestamp <= '2019-09-23 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-09-23 15:57:52
df.loc[(df.RIDER == 4) & (df.timestamp >= '2019-09-23 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-09-23 16:02:52

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-05 00:00:00') & (df.timestamp <= '2019-10-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
223349 2019-10-06 05:13:12 2019-10-05 18:12:33 0 days 01:20:00 0 days 11:01:00 2019-10-05 16:52:33
223571 2019-10-07 03:48:08 2019-10-06 16:47:29 0 days 04:09:59 0 days 11:01:00 2019-10-06 12:37:30
223656 2019-10-07 11:23:10 2019-10-07 00:22:31 0 days 00:35:00 0 days 11:01:00 2019-10-06 23:47:31
"""
df.loc[223348:223349, 'timestamp']
"""
223348   2019-10-05 16:52:33
223349   2019-10-05 18:12:33
"""
df.loc[223570:223571, 'timestamp']
"""
223570   2019-10-06 12:37:30
223571   2019-10-06 16:47:29
"""
# Note: in dexcom the timezone change is split up
# This is not observed in trainingpeaks

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-28 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
227455 2019-10-20 21:27:50 2019-10-20 12:27:12 0 days 00:49:58 0 days 09:01:00 2019-10-20 11:37:14
227756 2019-10-23 11:37:45 2019-10-23 02:37:07 1 days 13:09:57 0 days 09:01:00 2019-10-21 13:27:10
227851 2019-10-24 05:22:46 2019-10-23 18:22:08 0 days 07:54:59 0 days 11:01:00 2019-10-23 10:27:09
228605 2019-10-26 20:42:41 2019-10-26 09:42:03 0 days 00:35:01 0 days 11:01:00 2019-10-26 09:07:02
228612 2019-10-26 21:37:41 2019-10-26 10:37:03 0 days 00:25:00 0 days 11:01:00 2019-10-26 10:12:03
229199 2019-10-29 03:47:37 2019-10-28 16:46:59 0 days 05:19:59 0 days 11:01:00 2019-10-28 11:27:00
"""
# there is nothing in the calendar online
# the travel is again split up
df.loc[227454:227455, 'timestamp']
"""
227454   2019-10-20 11:37:14
227455   2019-10-20 12:27:12
"""
df.loc[227755:227756, 'timestamp']
"""
227755   2019-10-21 13:27:10
227756   2019-10-23 02:37:07
"""

# ------------------------- rider 5

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-06 00:00:00') & (df.timestamp <= '2019-01-09 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
242183 2019-01-08 10:11:28 2019-01-08 09:11:28 0 days 08:29:59 0 days 01:00:00 2019-01-08 00:41:29
"""
df.loc[242182:242183, 'timestamp']
"""
242182   2019-01-08 00:41:29
242183   2019-01-08 09:11:28
"""

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-15 00:00:00') & (df.timestamp <= '2019-01-19 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
244130 2019-01-15 05:26:11 2019-01-15 05:26:11 0 days 01:00:01 0 days 00:00:00 2019-01-15 04:26:10
244276 2019-01-15 17:56:10 2019-01-15 17:56:10 0 days 00:25:01 0 days 00:00:00 2019-01-15 17:31:09
244557 2019-01-16 18:06:07 2019-01-16 18:06:07 0 days 00:50:00 0 days 00:00:00 2019-01-16 17:16:07
245002 2019-01-19 10:11:35 2019-01-19 09:11:34 1 days 02:05:32 0 days 01:00:00 2019-01-18 07:06:02
"""
# online: 2019-01-16 was the (planned) travel day
# the timezone was changed on the 17th in the morning, which means that they most likely travelled before that
# therefore, instead of taking the 19th as the travel day, just keep it to 16th
df.loc[244556:244557, 'timestamp']
"""
244556   2019-01-16 17:16:07
244557   2019-01-16 18:06:07
"""

# on 31-03-2019 there was a change from summer time to winter time that happened overnight (so not manually)
df.loc[(df.RIDER == 5) & (df.timestamp <= '2019-03-31 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-03-31 00:59:39
df.loc[(df.RIDER == 5) & (df.timestamp >= '2019-03-31 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-03-31 01:04:39

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-21 00:00:00') & (df.timestamp <= '2019-05-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
276422 2019-05-21 03:50:24 2019-05-21 01:50:23 0 days 02:10:00 0 days 02:00:00 2019-05-20 23:40:23
276427 2019-05-21 05:10:24 2019-05-21 03:10:23 0 days 01:00:00 0 days 02:00:00 2019-05-21 02:10:23
276697 2019-05-22 04:35:21 2019-05-22 02:35:20 0 days 00:59:59 0 days 02:00:00 2019-05-22 01:35:21
276907 2019-05-22 23:40:20 2019-05-22 20:40:19 0 days 00:50:01 0 days 03:00:00 2019-05-22 19:50:18
276922 2019-05-23 02:10:19 2019-05-22 23:10:18 0 days 01:19:59 0 days 03:00:00 2019-05-22 21:50:19
276928 2019-05-23 02:30:19 2019-05-22 23:30:18 0 days 00:10:00 0 days 03:00:00 2019-05-22 23:20:18
"""
# I think the one at 2019-05-23 is too late, as the race was already then
# It probably happened before 2019-05-22 13:40 (local time) when he manually reset the phone
# I think all the other gaps happened at times when there are no airplanes going (unless he went by train??)
# Therefore we just take 2019-05-22 00:00:00 (local time) as the change (which is 2019-05-21 22:00:00 UTC)
df.loc[(df.RIDER == 5) & (df.timestamp <= '2019-05-21 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-05-21 21:55:21
df.loc[(df.RIDER == 5) & (df.timestamp >= '2019-05-21 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-05-21 22:00:21

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-25 00:00:00') & (df.timestamp <= '2019-05-26 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
277558 2019-05-25 11:55:13 2019-05-25 08:55:12 0 days 00:20:01 0 days 03:00:00 2019-05-25 08:35:11
277579 2019-05-25 15:35:12 2019-05-25 12:35:11 0 days 02:00:00 0 days 03:00:00 2019-05-25 10:35:11
277718 2019-05-26 03:55:10 2019-05-26 00:55:09 0 days 00:49:59 0 days 03:00:00 2019-05-26 00:05:10
277786 2019-05-26 09:40:10 2019-05-26 06:40:09 0 days 00:09:59 0 days 03:00:00 2019-05-26 06:30:10
277787 2019-05-26 08:50:11 2019-05-26 06:50:09 0 days 00:10:00 0 days 02:00:00 2019-05-26 06:40:09
277892 2019-05-26 17:45:09 2019-05-26 15:45:07 0 days 00:15:00 0 days 02:00:00 2019-05-26 15:30:07
"""
# online it looks like 2019-05-27, but 2019-05-26 also possible
# manual change happened on 2019-05-26 in the morning (so most likely travelled before that)
df.loc[277578:277579, 'timestamp']
"""
277578   2019-05-25 10:35:11
277579   2019-05-25 12:35:11
"""

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-16 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
312964 2019-10-16 03:51:17 2019-10-16 01:51:15 0 days 01:15:01 0 days 02:00:00 2019-10-16 00:36:14
"""
# this gap is not enough to travel to Japan
# online: 2019-10-16
# manual: 2019-10-16 evening
# fligera.net: Amsterdam (AMS / EHAM) 14:40 CEST	- Tokyo (NRT / RJAA) 08:40 JST
# previously mostly taking arrival time: 2019-10-16 23:40 UTC (note this is very similar to assumption
# which would have been 2019-10-16 22:00:00 UTC (so let's just take the assumption here as well))
df.loc[(df.RIDER == 5) & (df.timestamp <= '2019-10-16 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 21:56:14
df.loc[(df.RIDER == 5) & (df.timestamp >= '2019-10-16 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-16 22:01:14

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-20 00:00:00') & (df.timestamp <= '2019-11-11 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
314282 2019-10-21 04:06:04 2019-10-20 19:06:02 0 days 00:55:00 0 days 09:00:00 2019-10-20 18:11:02
314285 2019-10-21 05:01:05 2019-10-20 20:01:03 0 days 00:45:00 0 days 09:00:00 2019-10-20 19:16:03
314297 2019-10-21 06:46:05 2019-10-20 21:46:03 0 days 00:50:00 0 days 09:00:00 2019-10-20 20:56:03
314549 2019-10-22 04:36:01 2019-10-21 19:35:59 0 days 00:55:00 0 days 09:00:00 2019-10-21 18:40:59
314570 2019-10-22 09:36:00 2019-10-22 00:35:58 0 days 03:19:59 0 days 09:00:00 2019-10-21 21:15:59
315694 2019-10-26 00:20:49 2019-10-25 22:20:47 0 days 00:09:59 0 days 02:00:00 2019-10-25 22:10:48
315978 2019-11-01 01:20:32 2019-11-01 00:20:28 5 days 02:24:44 0 days 01:00:00 2019-10-26 21:55:44
316298 2019-11-02 05:15:30 2019-11-02 04:15:26 0 days 01:19:59 0 days 01:00:00 2019-11-02 02:55:27
316806 2019-11-04 00:25:26 2019-11-03 23:25:22 0 days 00:55:00 0 days 01:00:00 2019-11-03 22:30:22
316869 2019-11-04 06:05:25 2019-11-04 05:05:21 0 days 00:30:00 0 days 01:00:00 2019-11-04 04:35:21
317025 2019-11-04 19:35:24 2019-11-04 18:35:20 0 days 00:35:00 0 days 01:00:00 2019-11-04 18:00:20
317091 2019-11-05 02:15:22 2019-11-05 01:15:18 0 days 01:14:58 0 days 01:00:00 2019-11-05 00:00:20
317104 2019-11-05 03:35:23 2019-11-05 02:35:19 0 days 00:20:01 0 days 01:00:00 2019-11-05 02:15:18
317154 2019-11-05 08:40:22 2019-11-05 07:40:18 0 days 01:00:00 0 days 01:00:00 2019-11-05 06:40:18
317850 2019-11-07 19:00:15 2019-11-07 18:00:11 0 days 00:25:00 0 days 01:00:00 2019-11-07 17:35:11
318008 2019-11-08 08:50:14 2019-11-08 07:50:10 0 days 00:44:59 0 days 01:00:00 2019-11-08 07:05:11
318026 2019-11-08 10:50:14 2019-11-08 09:50:10 0 days 00:35:01 0 days 01:00:00 2019-11-08 09:15:09
318037 2019-11-08 11:55:13 2019-11-08 10:55:09 0 days 00:15:00 0 days 01:00:00 2019-11-08 10:40:09
318204 2019-11-09 02:00:12 2019-11-09 01:00:08 0 days 00:14:59 0 days 01:00:00 2019-11-09 00:45:09
318205 2019-11-09 03:10:11 2019-11-09 02:10:07 0 days 01:09:59 0 days 01:00:00 2019-11-09 01:00:08
318255 2019-11-09 08:00:12 2019-11-09 07:00:08 0 days 00:45:00 0 days 01:00:00 2019-11-09 06:15:08
318369 2019-11-09 17:55:10 2019-11-09 16:55:06 0 days 00:30:00 0 days 01:00:00 2019-11-09 16:25:06
318476 2019-11-10 03:20:10 2019-11-10 02:20:06 0 days 00:35:00 0 days 01:00:00 2019-11-10 01:45:06
318498 2019-11-10 06:00:09 2019-11-10 05:00:05 0 days 00:54:59 0 days 01:00:00 2019-11-10 04:05:06
318518 2019-11-10 08:15:08 2019-11-10 07:15:04 0 days 00:40:00 0 days 01:00:00 2019-11-10 06:35:04
318525 2019-11-10 09:15:08 2019-11-10 08:15:04 0 days 00:30:00 0 days 01:00:00 2019-11-10 07:45:04
318691 2019-11-11 01:20:07 2019-11-11 00:20:03 0 days 02:20:01 0 days 01:00:00 2019-11-10 22:00:02
"""
# online: 2019-10-22
# manual: 2019-10-22 afternoon
# there is no gap that is big enough again
# fligera.net: Tokyo (NRT / RJAA) 10:25 JST	- Amsterdam (AMS / EHAM) 15:05 CEST
# arrival at 2019-10-22 13:00 UTC
# keep manual timezone change (as it is very close to actual change)
# 2019-10-22 12:50:56
# 2019-10-22 12:55:56

# the last change is from summer time to winter time
df.loc[(df.RIDER == 5) & (df.timestamp <= '2019-10-27 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-26 21:55:44
df.loc[(df.RIDER == 5) & (df.timestamp >= '2019-10-27 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-11-01 00:20:28

# ------------------------- rider 6

# NOTE: we are checking for gaps here. The gaps usually happen when a rider puts his phone in 
# airplane mode and the bluetooth connection is lost. However, as 6 is not using his phone but
# a receiver, this problem does not occur, and it is a bit useless to look at gaps.

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-07 00:00:00') & (df.timestamp <= '2018-12-09 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# maybe the change was here also automatically for the first three entries with the phone
# and then afterwards with the receiver of course manually
# so far the first three (automatic?) changes check out

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-16 00:00:00') & (df.timestamp <= '2019-01-19 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
336451 2019-01-16 11:24:55 2019-01-16 00:21:41 0 days 02:20:01 0 days 11:03:00 2019-01-15 22:01:40
"""
# Gap is not big enough for travel from australia to spain
# trainingpeaks online: 2019-01-18
# maybe this change also still happened with the android

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-22 00:00:00') & (df.timestamp <= '2019-02-02 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
338206 2019-01-22 04:33:31 2019-01-22 03:26:20 0 days 01:04:59 0 days 01:07:00 2019-01-22 02:21:21
338560 2019-01-23 12:23:26 2019-01-23 11:16:15 0 days 02:29:59 0 days 01:07:00 2019-01-23 08:46:16
338993 2019-01-25 00:28:20 2019-01-24 23:21:09 0 days 00:34:58 0 days 01:07:00 2019-01-24 22:46:11
339086 2019-01-25 08:53:20 2019-01-25 07:46:09 0 days 00:45:00 0 days 01:07:00 2019-01-25 07:01:09
339264 2019-01-26 08:43:17 2019-01-26 07:36:06 0 days 09:19:59 0 days 01:07:00 2019-01-25 22:16:07
340260 2019-01-29 20:03:05 2019-01-29 18:55:54 0 days 00:30:00 0 days 01:07:00 2019-01-29 18:25:54
340264 2019-01-29 20:33:05 2019-01-29 19:25:54 0 days 00:15:01 0 days 01:07:00 2019-01-29 19:10:53
340947 2019-02-01 16:02:31 2019-02-01 04:55:43 0 days 00:39:58 0 days 11:07:00 2019-02-01 04:15:45
341328 2019-02-03 00:32:27 2019-02-02 13:25:39 0 days 00:50:01 0 days 11:07:00 2019-02-02 12:35:38
341331 2019-02-03 01:12:28 2019-02-02 14:05:40 0 days 00:30:02 0 days 11:07:00 2019-02-02 13:35:38
341350 2019-02-03 03:42:27 2019-02-02 16:35:39 0 days 01:00:00 0 days 11:07:00 2019-02-02 15:35:39
341419 2019-02-03 10:32:25 2019-02-02 23:25:37 0 days 01:10:00 0 days 11:07:00 2019-02-02 22:15:37
"""
# he got sick in this period during the training camp, there is nothing in the calendar
# I can't see when he travelled back..
# So now just assume it was on the day that he filled it out

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-19 00:00:00') & (df.timestamp <= '2019-03-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
353691 2019-03-21 21:24:45 2019-03-21 10:17:57 0 days 00:10:00 0 days 11:07:00 2019-03-21 10:07:57
353695 2019-03-21 21:49:44 2019-03-21 10:42:56 0 days 00:10:00 0 days 11:07:00 2019-03-21 10:32:56
353730 2019-03-22 00:54:45 2019-03-21 13:47:57 0 days 00:15:00 0 days 11:07:00 2019-03-21 13:32:57
354113 2019-03-22 23:44:31 2019-03-22 22:37:51 0 days 01:00:00 0 days 01:07:00 2019-03-22 21:37:51
"""
# not big enough of a gap again
# trainingpeaks online: 2019-03-20 as travel, and 2019-03-21 no cycling
# assume arrival on 22-03-2019 00:00:00 australia time, which is 21-03-2019 13:00:00 UTC
df.loc[(df.RIDER == 6) & (df.timestamp <= '2019-03-21 13:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-03-21 12:57:58
df.loc[(df.RIDER == 6) & (df.timestamp >= '2019-03-21 13:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-03-21 13:02:57

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-30 00:00:00') & (df.timestamp <= '2019-03-31 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
356558 2019-03-31 23:13:03 2019-03-31 22:06:23 0 days 00:34:59 0 days 01:07:00 2019-03-31 21:31:24
"""
# not related to travel again
# Note: he stayed in Spain here. The change was due to summer time/winter time.
# The change observed in Dexcom was on 2019-04-06, meaning that he changed it manually, but just a week later.
# This makes sense as he is using the receiver, which is not connected to the internet.
# So actual change:
df.loc[(df.RIDER == 6) & (df.timestamp <= '2019-03-31 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-03-31 00:56:25
df.loc[(df.RIDER == 6) & (df.timestamp >= '2019-03-31 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-03-31 01:01:24

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-21 00:00:00') & (df.timestamp <= '2019-05-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
361083 2019-05-21 19:40:35 2019-05-21 17:34:14 0 days 00:20:00 0 days 02:06:00 2019-05-21 17:14:14
361196 2019-05-22 05:15:33 2019-05-22 03:09:12 0 days 00:15:00 0 days 02:06:00 2019-05-22 02:54:12
361206 2019-05-22 07:35:33 2019-05-22 05:29:12 0 days 01:35:02 0 days 02:06:00 2019-05-22 03:54:10
"""
# again: gaps not useful
# trainingpeaks online: flight from belgium to estonia on 2019-05-22
# he changed it on 22.05.19 16:22 UTC, so he must have travelled before that
# so assume he now just changed it right after travelling himself
# 2019-05-22 16:19:11
# 2019-05-22 16:24:11

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-25 00:00:00') & (df.timestamp <= '2019-05-27 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
362009 2019-05-25 11:44:59 2019-05-25 08:39:05 0 days 00:30:01 0 days 03:06:00 2019-05-25 08:09:04
362011 2019-05-25 12:14:58 2019-05-25 09:09:04 0 days 00:24:58 0 days 03:06:00 2019-05-25 08:44:06
362015 2019-05-25 13:39:57 2019-05-25 10:34:03 0 days 01:10:00 0 days 03:06:00 2019-05-25 09:24:03
362084 2019-05-25 19:54:58 2019-05-25 16:49:04 0 days 00:36:00 0 days 03:06:00 2019-05-25 16:13:04
362094 2019-05-25 20:54:57 2019-05-25 17:49:03 0 days 00:15:00 0 days 03:06:00 2019-05-25 17:34:03
362118 2019-05-25 23:24:57 2019-05-25 20:19:03 0 days 00:35:00 0 days 03:06:00 2019-05-25 19:44:03
362321 2019-05-26 16:34:56 2019-05-26 13:29:02 0 days 00:30:01 0 days 03:06:00 2019-05-26 12:59:01
"""
# no info in gaps
# trainingpeaks online: it seems like the travelling happened on 2019-05-26 (probably from estonia to barcelona)
# manually changed on: 2019-05-30 (this was the day he travelled back from barcelona to australia)
# so not immediately changed and not other clues
# so assume he arrived at 2019-05-27 00:00 old local time (UTC+3)
df.loc[(df.RIDER == 6) & (df.timestamp <= '2019-05-26 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-05-26 20:59:00
df.loc[(df.RIDER == 6) & (df.timestamp >= '2019-05-26 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-05-26 21:04:00

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-28 00:00:00') & (df.timestamp <= '2019-05-31 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
363299 2019-05-30 03:19:46 2019-05-30 00:13:52 0 days 01:24:59 0 days 03:06:00 2019-05-29 22:48:53
363306 2019-05-30 04:59:46 2019-05-30 01:53:52 0 days 01:09:59 0 days 03:06:00 2019-05-30 00:43:53
363389 2019-05-30 13:38:44 2019-05-30 09:33:51 0 days 00:50:00 0 days 04:05:00 2019-05-30 08:43:51
363390 2019-05-30 14:43:43 2019-05-30 10:38:50 0 days 01:04:59 0 days 04:05:00 2019-05-30 09:33:51
363399 2019-05-30 15:38:44 2019-05-30 11:33:51 0 days 00:14:59 0 days 04:05:00 2019-05-30 11:18:52
363415 2019-05-30 18:18:43 2019-05-30 14:13:50 0 days 01:24:59 0 days 04:05:00 2019-05-30 12:48:51
363422 2019-05-30 19:03:45 2019-05-30 14:58:52 0 days 00:15:01 0 days 04:05:00 2019-05-30 14:43:51
363673 2019-05-31 22:51:53 2019-05-31 12:43:52 0 days 00:55:00 0 days 10:08:00 2019-05-31 11:48:52
363674 2019-05-31 23:46:53 2019-05-31 13:38:52 0 days 00:55:00 0 days 10:08:00 2019-05-31 12:43:52
"""
# trainingpeaks online: from 2019-05-29 to 2019-05-31
# manually changed on: 2019-05-30
# assume the manual change is correct
# 2019-05-30 19:38:50
# 2019-05-30 19:43:50

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-07-05 00:00:00') & (df.timestamp <= '2019-07-08 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# censoring between 2019-06-27 and 2019-08-12
df.loc[(df.RIDER == 6) & (df.timestamp <= '2019-06-28 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-06-27 05:17:43
df.loc[(df.RIDER == 6) & (df.timestamp >= '2019-08-11 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-12 09:23:08

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-28 00:00:00') & (df.timestamp <= '2019-09-30 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
383748 2019-09-29 16:55:11 2019-09-29 14:47:10 0 days 00:40:01 0 days 02:08:00 2019-09-29 14:07:09
"""
# manually changed on 2019-10-03 (too late)
# trainingpeaks online: 2019-09-28 to 2019-09-30
# assume he arrived on 2019-09-30 00:00 old local time (UTC+2)
df.loc[(df.RIDER == 6) & (df.timestamp <= '2019-09-29 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-09-29 21:57:08
df.loc[(df.RIDER == 6) & (df.timestamp >= '2019-09-29 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-09-29 22:02:08

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-05 00:00:00') & (df.timestamp <= '2019-10-19 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# australia DST 2019 is from april 7 until october 6
# manual change happened on 2019-10-14
# precise change: 2019-10-06 02:00 UTC+10 which is 2019-10-05 16:00 UTC
df.loc[(df.RIDER == 6) & (df.timestamp <= '2019-10-05 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-05 15:56:49
df.loc[(df.RIDER == 6) & (df.timestamp >= '2019-10-05 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-05 16:01:49


# ------------------------- rider 12

# Travel to Rwanda on 2019-02-13 to +02:00 (window day before and day after)
df.loc[(df.RIDER == 12) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-02-12 00:00:00') & (df.timestamp <= '2019-02-14 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# censoring between 2019-01-27 and 2019-02-17 
df.loc[471873:471874, 'timestamp']
"""
471873   2019-01-27 15:18:43
471874   2019-02-17 06:48:02
"""

# Travel back to Spain on 2019-03-05 to +01:00 (window day before and day after)
df.loc[(df.RIDER == 12) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-04 00:00:00') & (df.timestamp <= '2019-03-06 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# assume arrive on 2019-03-06 00:00:00 old local time (2019-03-05 22:00:00)
df.loc[(df.RIDER == 12) & (df.timestamp <= '2019-03-05 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-03-05 21:57:03
df.loc[(df.RIDER == 12) & (df.timestamp >= '2019-03-05 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-03-05 22:02:03

# Winter/Summer time in Spain on 2019-03-31 00:01:00 (?)
df.loc[(df.RIDER == 12) & (df.timestamp <= '2019-03-31 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-03-31 00:55:35
df.loc[(df.RIDER == 12) & (df.timestamp >= '2019-03-31 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-03-31 01:00:33

# Travel to US somewhere between 2019-05-06 and 2019-05-08
df.loc[(df.RIDER == 12) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-06 00:00:00') & (df.timestamp <= '2019-05-08 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty

# Travel back to Spain somewhere between 2019-05-18 and 2019-05-19
# Travel to China on 2019-05-23 or 2019-05-24
# Travel back to Spain on 2019-06-01
# Travel to China on 2019-10-06
# Travel to Japan on 2019-10-16
# Travel back to Spain on 2019-10-21 (?)
# Summer-Winter time in Italy on 2019-10-27 01:00?

# censoring between 2019
df.loc[481177:481178, 'timestamp']
"""
481177   2019-03-31 10:45:32
481178   2019-10-27 21:52:39
"""

# Holidays on 2019-11-06 to 2019-11-17
# assume arrival on 2019-11-07 00:00:00 old local time -> 2019-11-06 23:00:00
df.loc[(df.RIDER == 12) & (df.timestamp <= '2019-11-06 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-11-06 22:57:16
df.loc[(df.RIDER == 12) & (df.timestamp >= '2019-11-06 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-11-06 23:02:17
# assume arrival on 2019-11-17 00:00:00 old local time -> 2019-11-16 18:00:00
df.loc[(df.RIDER == 12) & (df.timestamp <= '2019-11-16 18:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-11-16 17:56:56
df.loc[(df.RIDER == 12) & (df.timestamp >= '2019-11-16 18:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-11-16 18:01:56


# ------------------------- rider 13

# TODO: find out actual changes
# Just assume it was at 00:00 old local time if we cannot find out the time that they arrived

# Note there was something wrong with the date 27.05.19 04:55

df.loc[(df.RIDER == 13) & (df.timestamp <= '2019-09-27 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-09-27 21:57:33
df.loc[(df.RIDER == 13) & (df.timestamp >= '2019-09-27 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-09-27 22:02:33

df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-30 00:00:00') & (df.timestamp <= '2019-10-02 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
570134 2019-09-30 11:33:20 2019-09-30 09:32:27 0 days 01:55:00 0 days 02:01:00 2019-09-30 07:37:27
570141 2019-09-30 12:28:20 2019-09-30 10:27:27 0 days 00:25:00 0 days 02:01:00 2019-09-30 10:02:27
570170 2019-09-30 15:38:20 2019-09-30 13:37:27 0 days 00:50:00 0 days 02:01:00 2019-09-30 12:47:27
570174 2019-09-30 20:28:20 2019-09-30 18:27:27 0 days 04:35:00 0 days 02:01:00 2019-09-30 13:52:27
"""
# we have no idea about the date
# we now just take the one where there was 4 hours censored
df.loc[570173:570174, 'timestamp']
"""
570173   2019-09-30 13:52:27
570174   2019-09-30 18:27:27
"""

df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# trainingpeaks online: seems to be on 2019-10-16
# assume now change at 2019-10-16 22:00 (UTC)
df.loc[(df.RIDER == 13) & (df.timestamp <= '2019-10-16 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 21:55:12
df.loc[(df.RIDER == 13)	& (df.timestamp >= '2019-10-16 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-16 22:00:12

df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-20 00:00:00') & (df.timestamp <= '2019-11-01 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
574854 2019-10-24 10:25:35 2019-10-24 08:24:42 0 days 00:50:00 0 days 02:01:00 2019-10-24 07:34:42
574862 2019-10-27 22:50:24 2019-10-27 20:49:31 3 days 11:49:48 0 days 02:01:00 2019-10-24 08:59:43
"""
# Note: we don't know when he flied back
# There is a 10-day time window
# But usually, they fly back right after the race, so now we can assume the same.
df.loc[(df.RIDER == 13) & (df.timestamp <= '2019-10-21 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-21 14:59:53
df.loc[(df.RIDER == 13) & (df.timestamp >= '2019-10-21 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-21 15:04:54

# ------------------------- rider 14

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-20 00:00:00') & (df.timestamp <= '2018-12-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
587675 2018-12-20 04:08:07 2018-12-20 01:08:07 0 days 00:49:59 0 days 03:00:00 2018-12-20 00:18:08
587711 2018-12-20 07:43:07 2018-12-20 04:43:07 0 days 00:40:00 0 days 03:00:00 2018-12-20 04:03:07
587807 2018-12-20 16:18:06 2018-12-20 13:18:06 0 days 00:40:01 0 days 03:00:00 2018-12-20 12:38:05
587885 2018-12-21 00:53:04 2018-12-20 21:53:04 0 days 02:10:01 0 days 03:00:00 2018-12-20 19:43:03
"""
# Travel from Spain to Finland
# I'm not sure if any of the above gaps correspond to travelling
# TrainingPeaks online: 2018-12-21
# Assume arrive at 2018-12-21 23:00:00
df.loc[(df.RIDER == 14) & (df.timestamp <= '2018-12-21 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2018-12-21 22:58:01
df.loc[(df.RIDER == 14) & (df.timestamp >= '2018-12-21 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2018-12-21 23:03:01

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-24 00:00:00') & (df.timestamp <= '2018-12-26 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# travel from finland to spain canary islands (therefore ttimezone UTC+00)
# trainingpeaks online: 2018-12-25
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2018-12-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2018-12-25 21:57:45
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2018-12-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2018-12-25 22:02:46

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-05 00:00:00') & (df.timestamp <= '2019-01-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# travel within spain (probably from canary/faroe islands to mainland)
# trainingpeaks online: cannot find anything
# assume arrival 2019-01-07 00:00:00 UTC
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-01-07 00:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-01-06 23:57:00
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-01-07 00:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-01-07 00:01:59

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-02-12 00:00:00') & (df.timestamp <= '2019-02-14 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
602158 2019-02-12 13:37:25 2019-02-12 10:37:25 0 days 03:30:01 0 days 03:00:00 2019-02-12 07:07:24
602170 2019-02-12 15:02:25 2019-02-12 12:02:25 0 days 00:30:00 0 days 03:00:00 2019-02-12 11:32:25
"""
# travel from Spain to Rwanda
# TrainingPeaks online: probably 2019-02-13
# assume arrival at 2019-02-13 23:00:00 UTC
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-02-13 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-02-13 22:57:17
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-02-13 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-02-13 23:02:18

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-04 00:00:00') & (df.timestamp <= '2019-05-31 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# censored between 2019-02-23 and 2019-05-31
"""
           local_timestamp           timestamp        censoring        timezone     timestamp_shift
605299 2019-05-31 11:30:24 2019-05-31 08:30:24 96 days 14:33:43 0 days 03:00:00 2019-02-23 17:56:41
605300 2019-05-31 12:50:24 2019-05-31 09:50:24  0 days 01:20:00 0 days 03:00:00 2019-05-31 08:30:24
605302 2019-05-31 18:15:21 2019-05-31 15:15:21  0 days 05:19:57 0 days 03:00:00 2019-05-31 09:55:24
605303 2019-05-31 18:30:22 2019-05-31 15:30:22  0 days 00:15:01 0 days 03:00:00 2019-05-31 15:15:21
605304 2019-05-31 19:00:22 2019-05-31 16:00:22  0 days 00:30:00 0 days 03:00:00 2019-05-31 15:30:22
605305 2019-05-31 19:20:21 2019-05-31 16:20:21  0 days 00:19:59 0 days 03:00:00 2019-05-31 16:00:22
605306 2019-05-31 19:30:22 2019-05-31 16:30:22  0 days 00:10:01 0 days 03:00:00 2019-05-31 16:20:21
605307 2019-05-31 19:40:21 2019-05-31 16:40:21  0 days 00:09:59 0 days 03:00:00 2019-05-31 16:30:22
605308 2019-05-31 19:50:21 2019-05-31 16:50:21  0 days 00:10:00 0 days 03:00:00 2019-05-31 16:40:21
605310 2019-05-31 21:00:23 2019-05-31 18:00:23  0 days 01:05:02 0 days 03:00:00 2019-05-31 16:55:21
"""
df.loc[605298:605299, 'timestamp']
# 2019-02-23 17:51:41
# 2019-05-31 08:30:24

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-06-17 00:00:00') & (df.timestamp <= '2019-06-18 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
605501 2019-06-17 05:24:00 2019-06-17 02:24:00 0 days 08:09:59 0 days 03:00:00 2019-06-16 18:14:01
605503 2019-06-17 08:33:59 2019-06-17 05:33:59 0 days 03:04:59 0 days 03:00:00 2019-06-17 02:29:00
605504 2019-06-17 11:08:58 2019-06-17 08:08:58 0 days 02:34:59 0 days 03:00:00 2019-06-17 05:33:59
605510 2019-06-17 11:53:59 2019-06-17 08:53:59 0 days 00:20:01 0 days 03:00:00 2019-06-17 08:33:58
605515 2019-06-18 14:53:53 2019-06-18 11:53:53 1 days 02:39:54 0 days 03:00:00 2019-06-17 09:13:59
"""
# from finland to slovenia
# no info in trainingpeaks online
# assume he arrived on 2019-06-18 00:00 old local time
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-06-17 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-06-17 09:13:59
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-06-17 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-06-18 11:53:53
# note that this is also during the last censoring

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-06-23 00:00:00') & (df.timestamp <= '2019-06-27 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
605527 2019-06-23 08:28:29 2019-06-23 05:28:29 0 days 10:44:58 0 days 03:00:00 2019-06-22 18:43:31
605529 2019-06-23 15:18:27 2019-06-23 12:18:27 0 days 06:44:58 0 days 03:00:00 2019-06-23 05:33:29
605530 2019-06-23 15:38:28 2019-06-23 12:38:28 0 days 00:20:01 0 days 03:00:00 2019-06-23 12:18:27
605531 2019-06-24 13:23:23 2019-06-24 10:23:23 0 days 21:44:55 0 days 03:00:00 2019-06-23 12:38:28
605532 2019-06-24 15:48:23 2019-06-24 12:48:23 0 days 02:25:00 0 days 03:00:00 2019-06-24 10:23:23
605534 2019-06-24 18:33:21 2019-06-24 15:33:21 0 days 02:39:58 0 days 03:00:00 2019-06-24 12:53:23
605537 2019-06-25 03:43:21 2019-06-25 00:43:21 0 days 09:00:00 0 days 03:00:00 2019-06-24 15:43:21
605538 2019-06-25 08:28:19 2019-06-25 05:28:19 0 days 04:44:58 0 days 03:00:00 2019-06-25 00:43:21
605540 2019-06-25 18:33:17 2019-06-25 15:33:17 0 days 09:59:58 0 days 03:00:00 2019-06-25 05:33:19
605544 2019-06-25 18:58:18 2019-06-25 15:58:18 0 days 00:10:01 0 days 03:00:00 2019-06-25 15:48:17
605562 2019-06-26 02:58:16 2019-06-25 23:58:16 0 days 06:34:58 0 days 03:00:00 2019-06-25 17:23:18
605563 2019-06-26 06:13:16 2019-06-26 03:13:16 0 days 03:15:00 0 days 03:00:00 2019-06-25 23:58:16
605565 2019-06-26 06:33:15 2019-06-26 03:33:15 0 days 00:14:59 0 days 03:00:00 2019-06-26 03:18:16
605568 2019-06-26 18:58:11 2019-06-26 15:58:11 0 days 12:14:56 0 days 03:00:00 2019-06-26 03:43:15
605570 2019-06-26 19:23:13 2019-06-26 16:23:13 0 days 00:20:02 0 days 03:00:00 2019-06-26 16:03:11
605574 2019-06-26 20:03:12 2019-06-26 17:03:12 0 days 00:24:59 0 days 03:00:00 2019-06-26 16:38:13
605575 2019-06-27 06:03:09 2019-06-27 03:03:09 0 days 09:59:57 0 days 03:00:00 2019-06-26 17:03:12
605577 2019-06-27 10:18:08 2019-06-27 07:18:08 0 days 04:09:59 0 days 03:00:00 2019-06-27 03:08:09
605578 2019-06-27 14:23:09 2019-06-27 11:23:09 0 days 04:05:01 0 days 03:00:00 2019-06-27 07:18:08
605579 2019-06-27 15:48:07 2019-06-27 12:48:07 0 days 01:24:58 0 days 03:00:00 2019-06-27 11:23:09
605583 2019-06-27 16:13:08 2019-06-27 13:13:08 0 days 00:10:02 0 days 03:00:00 2019-06-27 13:03:06
605585 2019-06-27 16:33:07 2019-06-27 13:33:07 0 days 00:14:59 0 days 03:00:00 2019-06-27 13:18:08
"""
# it seems like there are not too many glucose measurements during these days anyway
# trainingpeaks online: probably travel took place between 2019-06-24 and 2019-06-26
# assume arrival 2019-06-26 00:00 old local time
df.loc[(df.RIDER == 14) & (df.timestamp <= '2019-06-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-06-25 17:23:18
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-06-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-06-25 23:58:16

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-01 00:00:00') & (df.timestamp <= '2019-08-02 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# censoring between 2019-07-12 and 2019-08-03
df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-07-12 00:00:00') & (df.timestamp <= '2019-08-03 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp        censoring        timezone     timestamp_shift
608544 2019-08-03 13:42:21 2019-08-03 09:35:39 22 days 04:23:45 0 days 04:07:00 2019-07-12 05:11:54
"""
df.loc[608543:608544, 'timestamp']
"""
608543   2019-07-12 05:11:54
608544   2019-08-03 09:35:39
"""
# use the censored min and max as change dates

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-09 00:00:00') & (df.timestamp <= '2019-08-11 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
610760 2019-08-11 08:36:39 2019-08-11 04:29:57 0 days 00:59:58 0 days 04:07:00 2019-08-11 03:29:59
610768 2019-08-12 00:07:55 2019-08-11 20:01:13 0 days 14:56:16 0 days 04:07:00 2019-08-11 05:04:57
"""
# from poland to finland
# trainingpeaks online: probably on 2019-08-10
# assume arrival on 2019-08-11 00:00 old local time
df.loc[(df.RIDER == 14) & (df.timestamp <= '2019-08-10 21:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-08-10 21:54:59
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-08-10 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-10 22:00:00

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-12 00:00:00') & (df.timestamp <= '2019-08-14 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
610821 2019-08-12 05:02:53 2019-08-12 00:56:11 0 days 00:35:01 0 days 04:07:00 2019-08-12 00:21:10
"""
# from finland to spain
# trainingpeaks online: probably on 2019-08-13
# assume arrival on 2019-08-14 00:00 old local time
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-08-13 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-08-13 20:56:05
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-08-13 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-13 21:01:04

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-25 00:00:00') & (df.timestamp <= '2019-08-27 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# travel from Denmark to Finland
# trainingpeaks online: most likely on 2019-08-26
# assume arrival on 2019-08-27 00:00 old local time
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-08-26 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-08-25 13:15:19
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-08-26 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-30 21:00:19
# Note: some censoring here

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
614579 2019-10-15 09:03:53 2019-10-15 04:57:11 0 days 11:55:00 0 days 04:07:00 2019-10-14 17:02:11
614580 2019-10-15 09:13:52 2019-10-15 05:07:10 0 days 00:09:59 0 days 04:07:00 2019-10-15 04:57:11
614581 2019-10-15 09:38:53 2019-10-15 05:32:11 0 days 00:25:01 0 days 04:07:00 2019-10-15 05:07:10
614582 2019-10-15 11:48:52 2019-10-15 07:42:10 0 days 02:09:59 0 days 04:07:00 2019-10-15 05:32:11
614590 2019-10-15 12:33:50 2019-10-15 08:27:08 0 days 00:09:58 0 days 04:07:00 2019-10-15 08:17:10
614591 2019-10-15 15:13:50 2019-10-15 11:07:08 0 days 02:40:00 0 days 04:07:00 2019-10-15 08:27:08
614592 2019-10-15 21:53:49 2019-10-15 17:47:07 0 days 06:39:59 0 days 04:07:00 2019-10-15 11:07:08
614593 2019-10-16 00:18:49 2019-10-15 20:12:07 0 days 02:25:00 0 days 04:07:00 2019-10-15 17:47:07
614596 2019-10-16 10:33:50 2019-10-16 06:27:08 0 days 10:05:01 0 days 04:07:00 2019-10-15 20:22:07
614598 2019-10-16 10:53:47 2019-10-16 06:47:05 0 days 00:14:57 0 days 04:07:00 2019-10-16 06:32:08
614599 2019-10-16 12:23:47 2019-10-16 08:17:05 0 days 01:30:00 0 days 04:07:00 2019-10-16 06:47:05
614600 2019-10-16 12:43:48 2019-10-16 08:37:06 0 days 00:20:01 0 days 04:07:00 2019-10-16 08:17:05
614601 2019-10-16 12:58:48 2019-10-16 08:52:06 0 days 00:15:00 0 days 04:07:00 2019-10-16 08:37:06
"""
# travel from finland to japan
# assume arrival on 2019-10-17 00:00 old local time
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-10-16 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 08:57:06
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-10-16 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-18 01:12:00

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-20 00:00:00') & (df.timestamp <= '2019-10-23 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
614633 2019-10-20 04:13:37 2019-10-20 00:06:55 0 days 00:15:02 0 days 04:07:00 2019-10-19 23:51:53
614634 2019-10-20 04:43:35 2019-10-20 00:36:53 0 days 00:29:58 0 days 04:07:00 2019-10-20 00:06:55
614635 2019-10-20 05:23:34 2019-10-20 01:16:52 0 days 00:39:59 0 days 04:07:00 2019-10-20 00:36:53
614636 2019-10-20 05:43:33 2019-10-20 01:36:51 0 days 00:19:59 0 days 04:07:00 2019-10-20 01:16:52
614637 2019-10-20 05:53:33 2019-10-20 01:46:51 0 days 00:10:00 0 days 04:07:00 2019-10-20 01:36:51
614638 2019-10-20 06:08:35 2019-10-20 02:01:53 0 days 00:15:02 0 days 04:07:00 2019-10-20 01:46:51
614642 2019-10-20 06:38:34 2019-10-20 02:31:52 0 days 00:14:59 0 days 04:07:00 2019-10-20 02:16:53
614643 2019-10-21 02:58:30 2019-10-20 22:51:48 0 days 20:19:56 0 days 04:07:00 2019-10-20 02:31:52
614645 2019-10-21 03:13:30 2019-10-20 23:06:48 0 days 00:10:00 0 days 04:07:00 2019-10-20 22:56:48
614649 2019-10-22 18:08:25 2019-10-22 14:01:43 1 days 14:39:55 0 days 04:07:00 2019-10-20 23:21:48
614672 2019-10-22 21:33:24 2019-10-22 17:26:42 0 days 01:34:59 0 days 04:07:00 2019-10-22 15:51:43
614674 2019-10-23 06:43:23 2019-10-23 02:36:41 0 days 09:04:59 0 days 04:07:00 2019-10-22 17:31:42
614675 2019-10-23 08:33:22 2019-10-23 04:26:40 0 days 01:49:59 0 days 04:07:00 2019-10-23 02:36:41
614676 2019-10-23 11:28:21 2019-10-23 07:21:39 0 days 02:54:59 0 days 04:07:00 2019-10-23 04:26:40
614677 2019-10-23 13:38:21 2019-10-23 09:31:39 0 days 02:10:00 0 days 04:07:00 2019-10-23 07:21:39
614678 2019-10-23 16:03:20 2019-10-23 11:56:38 0 days 02:24:59 0 days 04:07:00 2019-10-23 09:31:39
614679 2019-10-23 16:18:20 2019-10-23 12:11:38 0 days 00:15:00 0 days 04:07:00 2019-10-23 11:56:38
614681 2019-10-23 16:33:21 2019-10-23 12:26:39 0 days 00:10:01 0 days 04:07:00 2019-10-23 12:16:38
614682 2019-10-23 16:53:20 2019-10-23 12:46:38 0 days 00:19:59 0 days 04:07:00 2019-10-23 12:26:39
614683 2019-10-23 17:13:20 2019-10-23 13:06:38 0 days 00:20:00 0 days 04:07:00 2019-10-23 12:46:38
614685 2019-10-23 17:33:20 2019-10-23 13:26:38 0 days 00:15:00 0 days 04:07:00 2019-10-23 13:11:38
614686 2019-10-23 17:48:20 2019-10-23 13:41:38 0 days 00:15:00 0 days 04:07:00 2019-10-23 13:26:38
614704 2019-10-23 20:23:21 2019-10-23 16:16:39 0 days 01:10:01 0 days 04:07:00 2019-10-23 15:06:38
"""
# assume arrival on 2019-10-23 00:00 old local time
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-10-22 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-22 14:56:43
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-10-22 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-22 15:01:43

# last change is from summer time to winter time
# this change happens at 01:00 UTC
df.loc[(df.RIDER == 14)	& (df.timestamp <= '2019-10-27 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-26 23:21:26
df.loc[(df.RIDER == 14)	& (df.timestamp >= '2019-10-27 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-27 07:31:25

# ------------------------- rider 15
df.loc[(df.RIDER == 15) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-05 00:00:00') & (df.timestamp <= '2019-10-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
632415 2019-10-07 10:39:59 2019-10-07 08:39:59 3 days 23:54:43 0 days 02:00:00 2019-10-03 08:45:16
632512 2019-10-07 18:54:58 2019-10-07 16:54:58 0 days 00:14:59 0 days 02:00:00 2019-10-07 16:39:59
"""
# travel from France to China
# reset on 2019-10-07 20:59
# maybe just assume he travelled exactly in the 3 days gap
df.loc[632414:632415, 'timestamp']
"""
632414   2019-10-03 08:45:16
632415   2019-10-07 08:39:59
"""

df.loc[(df.RIDER == 15) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
635004 2019-10-17 03:18:26 2019-10-16 19:19:25 0 days 00:45:01 0 days 07:59:00 2019-10-16 18:34:24
"""
# travel from China to Japan
# assume arrival on 2019-10-17 00:00 old local time
df.loc[(df.RIDER == 15)	& (df.timestamp <= '2019-10-16 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 15:59:24
df.loc[(df.RIDER == 15)	& (df.timestamp >= '2019-10-16 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-16 16:04:24

# Note: for the last changes, we do not have any data any more. We don't have any data from cycling
# neither any data from his calendar. It seems he just went on holiday here. So now leave it as it is, and then see.

del df, df_tp, df_changes ; gc.collect()

# ------------------ final file compilation

# note that this file was called df_changes before (TODO: maybe clean up code and rename)
df_dc = pd.read_csv(path+'timezone/dexcom_changes_manual.csv', index_col=[0,1])
df_dc.drop(['index', 'Source Device ID', 'Transmitter ID', 'device_change', 'transmitter_change', 
	'trust_time'], axis=1, inplace=True)

df_dc.drop(['timestamp_min', 'timestamp_max', 'local_timestamp_min', 'local_timestamp_max'], axis=1, inplace=True)
df_dc.rename(columns={'actual_min':'timestamp_min', 'actual_max':'timestamp_max'}, inplace=True)

df_dc.timezone = pd.to_timedelta(df_dc.timezone)
df_dc.timestamp_min = pd.to_datetime(df_dc.timestamp_min)
df_dc.timestamp_max = pd.to_datetime(df_dc.timestamp_max)

df_dc['local_timestamp_min'] = df_dc.timestamp_min + df_dc.timezone
df_dc['local_timestamp_max'] = df_dc.timestamp_max + df_dc.timezone

df_dc.to_csv(path+'timezone/dexcom_changes_final.csv')

# ------------------ create check file for Kristina
df_changes = pd.read_csv(path+'timezone/trainingpeaks_dexcom_changes_manual.csv', index_col=[0,1])

df_changes.timestamp_min = pd.to_datetime(df_changes.timestamp_min)
df_changes.timestamp_max = pd.to_datetime(df_changes.timestamp_max)

df_tp = pd.read_csv(path_tp+'timezone_list_final.csv', index_col=0)
df_tp.timestamp_min = pd.to_datetime(df_tp.timestamp_min)
df_tp.timestamp_max = pd.to_datetime(df_tp.timestamp_max)
df_tp.timezone = pd.to_timedelta(df_tp.timezone)
df_tp.date = pd.to_datetime(df_tp.date)

df_changes['country_min'], df_changes['country_max'] = np.nan, np.nan
for idx, ((i,_), (ts_min, ts_max)) in enumerate(df_changes[['timestamp_min', 'timestamp_max']].iterrows()):
	try:
		df_changes['country_min'].iloc[idx] = df_tp.loc[(df_tp.RIDER == i) & (df_tp['timestamp_min'] == ts_min), 'location'].values[0]
	except IndexError:
		pass
	try:
		df_changes['country_max'].iloc[idx] = df_tp.loc[(df_tp.RIDER == i) & (df_tp['timestamp_max'] == ts_max), 'location'].values[0]
	except IndexError:
		pass

df_changes.drop(['timestamp_min', 'timestamp_max', 'local_timestamp_min', 'local_timestamp_max'], axis=1, inplace=True)

df_changes.rename(columns={'date_min':'date_trainingpeaks_min',
						   'date_max':'date_trainingpeaks_max'}, inplace=True)

# convert to datetime
df_changes.replace({'CENSORED':np.nan, }, inplace=True)
df_changes.actual_min = pd.to_datetime(df_changes.actual_min)
df_changes.actual_max = pd.to_datetime(df_changes.actual_max)
df_changes.timezone = pd.to_timedelta(df_changes.timezone)

# calculate actual local time
df_changes['local_timestamp_min'] = df_changes['actual_min'] + df_changes['timezone']
df_changes['local_timestamp_max'] = df_changes['actual_max'] + df_changes['timezone']
df_changes.drop(['actual_min', 'actual_max'], axis=1, inplace=True)

df_changes.reset_index(inplace=True)
df_changes.drop('n', axis=1, inplace=True)

rider_mapping_inv = {v:k for k, v in rider_mapping.items()}
df_changes.RIDER = df_changes.RIDER.map(rider_mapping_inv)

df_changes = df_changes[['RIDER', 'date_trainingpeaks_min', 'date_trainingpeaks_max', 'local_timestamp_min',
						'local_timestamp_max', 'certainty', 'timezone', 'country_min', 'country_max']]

df_changes.to_csv(path+'timezone/travel_check_Kristina.csv')