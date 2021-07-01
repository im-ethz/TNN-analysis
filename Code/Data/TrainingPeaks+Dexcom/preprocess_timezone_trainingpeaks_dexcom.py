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

# Figure out actual timezone changes

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
	if pd.isnull(df_changes.loc[(i,n_max[i]-1), 'timestamp_max']):
		df_changes.loc[(i,n_max[i]-1), 'timestamp_max'] = df.loc[(df.RIDER == i) & (df['Event Type'] == 'EGV'), 'timestamp'].iloc[-1]

	df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max'] = df.loc[(df.RIDER == i), 'local_timestamp'].iloc[-1]
	if pd.isnull(df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max']):
		df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max'] = df.loc[(df.RIDER == i) & (df['Event Type'] == 'EGV'), 'local_timestamp'].iloc[-1]

	print(i, df_changes.loc[(i,n_max[i]-1), 'timestamp_max'], df_changes.loc[(i,n_max[i]-1), 'local_timestamp_max'])

df_changes.to_csv(path+'timezone/dexcom_changes.csv')

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

df_changes.reset_index(inplace=True)
df_changes['n'] = df_changes.groupby('RIDER').cumcount()
df_changes.set_index(['RIDER', 'n'], inplace=True)
n_max = df_changes.groupby('RIDER')['timezone'].count()

# CHECK if timestamp of timezone changes are correct
for n in range(n_max[1]):
	check_minmax(df_changes, df_tp, (1,n))
# later happens quite a few times

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
72061 2019-09-03 20:36:57 2019-09-03 18:36:58 0 days 00:55:01 0 days 02:00:00 2019-09-03 17:41:57
72115 2019-09-04 01:21:55 2019-09-03 23:21:56 0 days 00:20:01 0 days 02:00:00 2019-09-03 23:01:55
72648 2019-09-05 22:41:52 2019-09-05 20:41:53 0 days 01:00:01 0 days 02:00:00 2019-09-05 19:41:52
"""
# online it seems like the travel day is 2019-09-05 as there is no cycling
# note that the gap is not long enough for the travelling, but maybe this is the time he went
# through security or something?
df.loc[72647:72648, 'timestamp']
"""
72647   2019-09-05 19:41:52
72648   2019-09-05 20:41:53
"""

df.loc[(df.RIDER == 1) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-22 00:00:00') & (df.timestamp <= '2019-09-24 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
          local_timestamp           timestamp       censoring        timezone     timestamp_shift
76979 2019-09-22 09:06:11 2019-09-22 01:06:12 0 days 01:25:00 0 days 08:00:00 2019-09-21 23:41:12
77040 2019-09-22 15:01:11 2019-09-22 07:01:12 0 days 00:55:00 0 days 08:00:00 2019-09-22 06:06:12
77085 2019-09-22 19:01:10 2019-09-22 11:01:11 0 days 00:20:01 0 days 08:00:00 2019-09-22 10:41:10
77297 2019-09-23 13:21:08 2019-09-23 05:21:09 0 days 00:45:00 0 days 08:00:00 2019-09-23 04:36:09
77338 2019-09-23 17:56:08 2019-09-23 09:56:09 0 days 01:14:58 0 days 08:00:00 2019-09-23 08:41:11
77380 2019-09-23 23:31:07 2019-09-23 15:31:08 0 days 02:09:59 0 days 08:00:00 2019-09-23 13:21:09
77443 2019-09-24 07:01:06 2019-09-23 23:01:07 0 days 02:19:58 0 days 08:00:00 2019-09-23 20:41:09
77445 2019-09-24 07:21:07 2019-09-23 23:21:08 0 days 00:15:00 0 days 08:00:00 2019-09-23 23:06:08
77475 2019-09-24 11:26:05 2019-09-24 03:26:06 0 days 01:39:58 0 days 08:00:00 2019-09-24 01:46:08
77479 2019-09-24 12:16:08 2019-09-24 04:16:09 0 days 00:35:03 0 days 08:00:00 2019-09-24 03:41:06
77490 2019-09-25 01:01:05 2019-09-24 17:01:06 0 days 11:54:59 0 days 08:00:00 2019-09-24 05:06:07
"""
df.loc[77489:77490, 'timestamp']
"""
77489   2019-09-24 05:06:07
77490   2019-09-24 17:01:06
"""

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
df.loc[144509:144510, 'timestamp']
# Note that this is because of censoring between 2018-12-12 and 2019-01-14

# Missing travelling from +2 to +8 and back between 2019-05-21 and 2019-06-02
df.loc[(df.RIDER == 3) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-31 00:00:00') & (df.timestamp <= '2019-06-18 00:00:00')]
# Note that this is because of censoring between 2019-04-04 and 2019-06-11

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

# round off
df_changes.loc[(4,11), 'timezone'] = df_changes.loc[(4,11), 'timezone'].round('h')

# CHECK if timezones are correct

# Missing travelling
df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-04-06 00:00:00') & (df.timestamp <= '2019-09-04 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
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

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-07 00:00:00') & (df.timestamp <= '2018-12-09 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
193500 2018-12-07 11:18:17 2018-12-07 10:18:17 0 days 02:19:59 0 days 01:00:00 2018-12-07 07:58:18
193518 2018-12-07 13:43:17 2018-12-07 12:43:17 0 days 01:00:00 0 days 01:00:00 2018-12-07 11:43:17
193568 2018-12-07 20:03:16 2018-12-07 19:03:16 0 days 02:15:00 0 days 01:00:00 2018-12-07 16:48:16
193585 2018-12-08 05:08:15 2018-12-08 04:08:15 0 days 07:44:59 0 days 01:00:00 2018-12-07 20:23:16
193590 2018-12-08 10:13:14 2018-12-08 09:13:14 0 days 04:44:58 0 days 01:00:00 2018-12-08 04:28:16
193595 2018-12-08 12:03:14 2018-12-08 11:03:14 0 days 01:30:00 0 days 01:00:00 2018-12-08 09:33:14
193694 2018-12-09 17:08:13 2018-12-09 16:08:11 0 days 20:54:58 0 days 01:00:00 2018-12-08 19:13:13
"""
df.loc[193693:193694, 'timestamp']
"""
193693   2018-12-08 19:13:13
193694   2018-12-09 16:08:11
"""
# note that he changed the dexcom earlier than when he actually travelled

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-16 00:00:00') & (df.timestamp <= '2019-01-18 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp        censoring        timezone     timestamp_shift
198945 2019-01-17 11:42:11 2019-01-17 00:42:09 17 days 18:19:00 0 days 11:00:00 2018-12-30 06:23:09
198998 2019-01-17 20:41:57 2019-01-17 09:41:55  0 days 04:39:51 0 days 11:00:00 2019-01-17 05:02:04
199004 2019-01-18 07:02:17 2019-01-18 06:01:54  0 days 19:54:59 0 days 01:00:00 2019-01-17 10:06:55
199022 2019-01-18 09:27:16 2019-01-18 08:26:53  0 days 01:00:00 0 days 01:00:00 2019-01-18 07:26:53
199061 2019-01-18 15:42:15 2019-01-18 14:41:52  0 days 03:05:00 0 days 01:00:00 2019-01-18 11:36:52
"""
df.loc[199003:199004, 'timestamp'] # note that this is the change we also found in dexcom
"""
199003   2019-01-17 10:06:55
199004   2019-01-18 06:01:54
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-02-19 00:00:00') & (df.timestamp <= '2019-02-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
204164 2019-02-19 11:15:47 2019-02-19 10:15:24 0 days 01:29:59 0 days 01:00:00 2019-02-19 08:45:25
204454 2019-02-20 13:10:45 2019-02-20 12:10:22 0 days 01:50:00 0 days 01:00:00 2019-02-20 10:20:22
204549 2019-02-20 22:00:44 2019-02-20 21:00:21 0 days 01:00:02 0 days 01:00:00 2019-02-20 20:00:19
204689 2019-02-21 21:10:41 2019-02-21 20:10:18 0 days 11:34:59 0 days 01:00:00 2019-02-21 08:35:19
204807 2019-02-22 11:10:39 2019-02-22 07:10:17 0 days 01:15:00 0 days 04:00:00 2019-02-22 05:55:17
204947 2019-02-22 23:25:37 2019-02-22 19:25:15 0 days 00:39:59 0 days 04:00:00 2019-02-22 18:45:16
"""
df.loc[204688:204689, 'timestamp']
"""
204688   2019-02-21 08:35:19
204689   2019-02-21 20:10:18
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-02 00:00:00') & (df.timestamp <= '2019-03-05 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
206157 2019-03-02 12:25:18 2019-03-02 08:24:56 0 days 09:30:00 0 days 04:00:00 2019-03-01 22:54:56
206158 2019-03-02 15:35:17 2019-03-02 11:34:55 0 days 03:09:59 0 days 04:00:00 2019-03-02 08:24:56
206227 2019-03-02 22:15:16 2019-03-02 18:14:54 0 days 01:00:00 0 days 04:00:00 2019-03-02 17:14:54
206256 2019-03-03 01:35:14 2019-03-02 21:34:52 0 days 00:59:59 0 days 04:00:00 2019-03-02 20:34:53
206307 2019-03-03 06:40:16 2019-03-03 02:39:54 0 days 00:55:01 0 days 04:00:00 2019-03-03 01:44:53
206324 2019-03-03 16:35:14 2019-03-03 12:34:52 0 days 08:34:59 0 days 04:00:00 2019-03-03 03:59:53
206364 2019-03-03 21:15:14 2019-03-03 17:14:52 0 days 01:25:02 0 days 04:00:00 2019-03-03 15:49:50
206692 2019-03-05 07:00:09 2019-03-05 05:59:47 0 days 09:29:57 0 days 01:00:00 2019-03-04 20:29:50
"""
# Online it says travel was on 2019-03-03
df.loc[206323:206324, 'timestamp']
"""
206323   2019-03-03 03:59:53
206324   2019-03-03 12:34:52
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-22 00:00:00') & (df.timestamp <= '2019-03-25 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
209938 2019-03-22 14:24:20 2019-03-22 13:23:58 0 days 04:24:58 0 days 01:00:00 2019-03-22 08:59:00
210030 2019-03-23 00:45:10 2019-03-22 23:44:48 0 days 02:45:50 0 days 01:00:00 2019-03-22 20:58:58
210130 2019-03-23 16:35:09 2019-03-23 15:34:47 0 days 07:34:59 0 days 01:00:00 2019-03-23 07:59:48
210159 2019-03-24 09:15:04 2019-03-24 08:14:42 0 days 14:19:56 0 days 01:00:00 2019-03-23 17:54:46
210164 2019-03-24 20:20:04 2019-03-24 19:19:42 0 days 10:44:59 0 days 01:00:00 2019-03-24 08:34:43
210181 2019-03-24 23:30:04 2019-03-24 22:29:42 0 days 01:50:00 0 days 01:00:00 2019-03-24 20:39:42
"""
# online it says 2019-03-24  8pm (land)
df.loc[210163:210164, 'timestamp']
"""
210163   2019-03-24 08:34:43
210164   2019-03-24 19:19:42
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
213270 2019-04-07 08:14:31 2019-04-06 22:14:09 0 days 22:44:57 0 days 10:00:00 2019-04-05 23:29:12
"""
df.loc[213269:213270, 'timestamp']
"""
213269   2019-04-05 23:29:12
213270   2019-04-06 22:14:09
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-03 00:00:00') & (df.timestamp <= '2019-09-04 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp         censoring        timezone     timestamp_shift
216931 2019-09-04 06:38:47 2019-09-04 04:38:47 112 days 01:56:22 0 days 02:00:00 2019-05-15 02:42:25
216969 2019-09-04 15:53:02 2019-09-04 07:53:45   0 days 00:10:00 0 days 07:59:00 2019-09-04 07:43:45
217006 2019-09-04 20:28:02 2019-09-04 12:28:45   0 days 01:35:00 0 days 07:59:00 2019-09-04 10:53:45
"""
# online: 2019-09-04
# it seems he started using dexcom again after arriving in China
# there is no exact flight date
df.loc[216930:216931, 'timestamp']
"""
216930   2019-05-15 02:42:25
216931   2019-09-04 04:38:47
"""

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-22 00:00:00') & (df.timestamp <= '2019-09-24 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
221988 2019-09-25 01:22:31 2019-09-24 17:23:14 0 days 01:50:01 0 days 07:59:00 2019-09-24 15:33:13
221996 2019-09-25 03:22:29 2019-09-24 19:23:12 0 days 01:24:59 0 days 07:59:00 2019-09-24 17:58:13
222007 2019-09-25 04:47:27 2019-09-24 20:48:10 0 days 00:35:00 0 days 07:59:00 2019-09-24 20:13:10
"""
# online: 2019-09-23 and 2019-09-24
# No big enough gap found, so we take 2019-09-24 00:00:00 local time
# 2019-09-23 15:58:14
# 2019-09-23 16:03:14

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-05 00:00:00') & (df.timestamp <= '2019-10-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
223115 2019-10-06 05:13:12 2019-10-05 18:12:55 0 days 01:20:00 0 days 11:00:00 2019-10-05 16:52:55
223337 2019-10-07 03:48:08 2019-10-06 16:47:51 0 days 04:09:59 0 days 11:00:00 2019-10-06 12:37:52
223422 2019-10-07 11:23:10 2019-10-07 00:22:53 0 days 00:35:00 0 days 11:00:00 2019-10-06 23:47:53
"""
df.loc[223114:223115, 'timestamp']
"""
223114   2019-10-05 16:52:55
223115   2019-10-05 18:12:55
"""
df.loc[223336:223337, 'timestamp']
"""
223336   2019-10-06 12:37:52
223337   2019-10-06 16:47:51
"""
# Note: in dexcom the timezone change is split up
# This is not observed in trainingpeaks

df.loc[(df.RIDER == 4) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-28 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
227221 2019-10-20 21:27:50 2019-10-20 12:27:34 0 days 00:49:58 0 days 09:00:00 2019-10-20 11:37:36
227522 2019-10-23 11:37:45 2019-10-23 02:37:29 1 days 13:09:57 0 days 09:00:00 2019-10-21 13:27:32
227617 2019-10-24 05:22:46 2019-10-23 18:22:30 0 days 07:54:59 0 days 11:00:00 2019-10-23 10:27:31
228371 2019-10-26 20:42:41 2019-10-26 09:42:25 0 days 00:35:01 0 days 11:00:00 2019-10-26 09:07:24
228378 2019-10-26 21:37:41 2019-10-26 10:37:25 0 days 00:25:00 0 days 11:00:00 2019-10-26 10:12:25
228965 2019-10-29 03:47:37 2019-10-28 16:47:21 0 days 05:19:59 0 days 11:00:00 2019-10-28 11:27:22
"""
# there is nothing in the calendar online
# the travel is again split up
df.loc[227220:227221, 'timestamp']
"""
227220   2019-10-20 11:37:36
227221   2019-10-20 12:27:34
"""
df.loc[227521:227522, 'timestamp']
"""
227521   2019-10-21 13:27:32
227522   2019-10-23 02:37:29
"""

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

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-06 00:00:00') & (df.timestamp <= '2019-01-09 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
235867 2019-01-07 02:31:32 2019-01-07 01:31:32 0 days 00:40:01 0 days 01:00:00 2019-01-07 00:51:31
236146 2019-01-08 10:11:28 2019-01-08 09:11:28 0 days 08:29:59 0 days 01:00:00 2019-01-08 00:41:29
"""
df.loc[236145:236146, 'timestamp']
"""
236145   2019-01-08 00:41:29
236146   2019-01-08 09:11:28
"""

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-15 00:00:00') & (df.timestamp <= '2019-01-19 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
238093 2019-01-15 05:26:11 2019-01-15 05:26:11 0 days 01:00:01 0 days 00:00:00 2019-01-15 04:26:10
238239 2019-01-15 17:56:10 2019-01-15 17:56:10 0 days 00:25:01 0 days 00:00:00 2019-01-15 17:31:09
238520 2019-01-16 18:06:07 2019-01-16 18:06:07 0 days 00:50:00 0 days 00:00:00 2019-01-16 17:16:07
238965 2019-01-19 10:11:35 2019-01-19 09:11:34 1 days 02:05:32 0 days 01:00:00 2019-01-18 07:06:02
"""
# online: 2019-01-16 was the (planned) travel day
# the timezone was changed on the 17th in the morning, which means that they most likely travelled before that
# therefore, instead of taking the 19th as the travel day, just keep it to 16th
df.loc[238519:238520, 'timestamp']
"""
238519   2019-01-16 17:16:07
238520   2019-01-16 18:06:07
"""

# on 31-03-2019 there was a change from summer time to winter time that happened overnight (so not manually)
# 2019-03-31 00:59:39
# 2019-03-31 01:04:39

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-21 00:00:00') & (df.timestamp <= '2019-05-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
269994 2019-05-21 03:50:24 2019-05-21 01:50:23 0 days 02:10:00 0 days 02:00:00 2019-05-20 23:40:23
269999 2019-05-21 05:10:24 2019-05-21 03:10:23 0 days 01:00:00 0 days 02:00:00 2019-05-21 02:10:23
270269 2019-05-22 04:35:21 2019-05-22 02:35:20 0 days 00:59:59 0 days 02:00:00 2019-05-22 01:35:21
270479 2019-05-22 23:40:20 2019-05-22 20:40:19 0 days 00:50:01 0 days 03:00:00 2019-05-22 19:50:18
270494 2019-05-23 02:10:19 2019-05-22 23:10:18 0 days 01:19:59 0 days 03:00:00 2019-05-22 21:50:19
270500 2019-05-23 02:30:19 2019-05-22 23:30:18 0 days 00:10:00 0 days 03:00:00 2019-05-22 23:20:18
"""
# I think the one at 2019-05-23 is too late, as the race was already then
# It probably happened before 2019-05-22 13:40 (local time) when he manually reset the phone
# I think all the other gaps happened at times when there are no airplanes going (unless he went by train??)
# Therefore we just take 2019-05-22 00:00:00 (local time) as the change (which is 2019-05-21 22:00:00 UTC)
# 2019-05-21 21:55:21
# 2019-05-21 22:00:21

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-25 00:00:00') & (df.timestamp <= '2019-05-26 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
271130 2019-05-25 11:55:13 2019-05-25 08:55:12 0 days 00:20:01 0 days 03:00:00 2019-05-25 08:35:11
271151 2019-05-25 15:35:12 2019-05-25 12:35:11 0 days 02:00:00 0 days 03:00:00 2019-05-25 10:35:11
271290 2019-05-26 03:55:10 2019-05-26 00:55:09 0 days 00:49:59 0 days 03:00:00 2019-05-26 00:05:10
271358 2019-05-26 09:40:10 2019-05-26 06:40:09 0 days 00:09:59 0 days 03:00:00 2019-05-26 06:30:10
271359 2019-05-26 08:50:11 2019-05-26 06:50:09 0 days 00:10:00 0 days 02:00:00 2019-05-26 06:40:09
271464 2019-05-26 17:45:09 2019-05-26 15:45:07 0 days 00:15:00 0 days 02:00:00 2019-05-26 15:30:07
"""
# online it looks like 2019-05-27, but 2019-05-26 also possible
# manual change happened on 2019-05-26 in the morning (so most likely travelled before that)
df.loc[271150:271151, 'timestamp']
"""
271150   2019-05-25 10:35:11
271151   2019-05-25 12:35:11
"""

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-16 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
306536 2019-10-16 03:51:17 2019-10-16 01:51:15 0 days 01:15:01 0 days 02:00:00 2019-10-16 00:36:14
"""
# this gap is not enough to travel to Japan
# online: 2019-10-16
# manual: 2019-10-16 evening
# fligera.net: Amsterdam (AMS / EHAM) 14:40 CEST	- Tokyo (NRT / RJAA) 08:40 JST
# previously mostly taking arrival time: 2019-10-16 23:40 UTC (note this is very similar to assumption
# which would have been 2019-10-16 22:00:00 UTC (so let's just take the assumption here as well))
# 2019-10-16 21:56:14
# 2019-10-16 22:01:14

df.loc[(df.RIDER == 5) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-20 00:00:00') & (df.timestamp <= '2019-11-11 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
307854 2019-10-21 04:06:04 2019-10-20 19:06:02 0 days 00:55:00 0 days 09:00:00 2019-10-20 18:11:02
307857 2019-10-21 05:01:05 2019-10-20 20:01:03 0 days 00:45:00 0 days 09:00:00 2019-10-20 19:16:03
307869 2019-10-21 06:46:05 2019-10-20 21:46:03 0 days 00:50:00 0 days 09:00:00 2019-10-20 20:56:03
308121 2019-10-22 04:36:01 2019-10-21 19:35:59 0 days 00:55:00 0 days 09:00:00 2019-10-21 18:40:59
308142 2019-10-22 09:36:00 2019-10-22 00:35:58 0 days 03:19:59 0 days 09:00:00 2019-10-21 21:15:59
309266 2019-10-26 00:20:49 2019-10-25 22:20:47 0 days 00:09:59 0 days 02:00:00 2019-10-25 22:10:48
309550 2019-11-01 01:20:32 2019-11-01 00:20:28 5 days 02:24:44 0 days 01:00:00 2019-10-26 21:55:44
309870 2019-11-02 05:15:30 2019-11-02 04:15:26 0 days 01:19:59 0 days 01:00:00 2019-11-02 02:55:27
310378 2019-11-04 00:25:26 2019-11-03 23:25:22 0 days 00:55:00 0 days 01:00:00 2019-11-03 22:30:22
310441 2019-11-04 06:05:25 2019-11-04 05:05:21 0 days 00:30:00 0 days 01:00:00 2019-11-04 04:35:21
310597 2019-11-04 19:35:24 2019-11-04 18:35:20 0 days 00:35:00 0 days 01:00:00 2019-11-04 18:00:20
310663 2019-11-05 02:15:22 2019-11-05 01:15:18 0 days 01:14:58 0 days 01:00:00 2019-11-05 00:00:20
310676 2019-11-05 03:35:23 2019-11-05 02:35:19 0 days 00:20:01 0 days 01:00:00 2019-11-05 02:15:18
310726 2019-11-05 08:40:22 2019-11-05 07:40:18 0 days 01:00:00 0 days 01:00:00 2019-11-05 06:40:18
311422 2019-11-07 19:00:15 2019-11-07 18:00:11 0 days 00:25:00 0 days 01:00:00 2019-11-07 17:35:11
311580 2019-11-08 08:50:14 2019-11-08 07:50:10 0 days 00:44:59 0 days 01:00:00 2019-11-08 07:05:11
311598 2019-11-08 10:50:14 2019-11-08 09:50:10 0 days 00:35:01 0 days 01:00:00 2019-11-08 09:15:09
311609 2019-11-08 11:55:13 2019-11-08 10:55:09 0 days 00:15:00 0 days 01:00:00 2019-11-08 10:40:09
311776 2019-11-09 02:00:12 2019-11-09 01:00:08 0 days 00:14:59 0 days 01:00:00 2019-11-09 00:45:09
311777 2019-11-09 03:10:11 2019-11-09 02:10:07 0 days 01:09:59 0 days 01:00:00 2019-11-09 01:00:08
311827 2019-11-09 08:00:12 2019-11-09 07:00:08 0 days 00:45:00 0 days 01:00:00 2019-11-09 06:15:08
311941 2019-11-09 17:55:10 2019-11-09 16:55:06 0 days 00:30:00 0 days 01:00:00 2019-11-09 16:25:06
312048 2019-11-10 03:20:10 2019-11-10 02:20:06 0 days 00:35:00 0 days 01:00:00 2019-11-10 01:45:06
312070 2019-11-10 06:00:09 2019-11-10 05:00:05 0 days 00:54:59 0 days 01:00:00 2019-11-10 04:05:06
312090 2019-11-10 08:15:08 2019-11-10 07:15:04 0 days 00:40:00 0 days 01:00:00 2019-11-10 06:35:04
312097 2019-11-10 09:15:08 2019-11-10 08:15:04 0 days 00:30:00 0 days 01:00:00 2019-11-10 07:45:04
312263 2019-11-11 01:20:07 2019-11-11 00:20:03 0 days 02:20:01 0 days 01:00:00 2019-11-10 22:00:02
"""
# online: 2019-10-22
# manual: 2019-10-22 afternoon
# there is no gap that is big enough again
# fligera.net: Tokyo (NRT / RJAA) 10:25 JST	- Amsterdam (AMS / EHAM) 15:05 CEST
# arrival at 2019-10-22 13:00 UTC
# keep manual timezone change (as it is very close to actual change)
# 22.10.2019 12:50:56
# 22.10.2019 12:55:56

# the last change is from summer time to winter time
# 27.10.2019 00:59:48
# 27.10.2019 01:04:48

# ------------------------- rider 6

# alternating between receiver and phone (receiver still at old time)
df_changes = delete_rows(df_changes, 6, range(2,42))
df_changes = keep_second(df_changes, 6, [0,1])

# alternating between receiver and phone (receiver still at old time)
df_changes = delete_rows(df_changes, 6, range(44,48))
df_changes = keep_second(df_changes, 6, [42,43])

# the rider reset it here with an extra couple of min, but the actual timezone change is of course just 1 h
for n in range(49,59):
	df_changes.loc[(6,n), 'timezone'] = df_changes.loc[(6,n), 'timezone'].round('h')

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
330023 2019-01-16 11:24:55 2019-01-16 00:24:41 0 days 02:20:01 0 days 11:00:00 2019-01-15 22:04:40
"""
# Gap is not big enough for travel from australia to spain
# trainingpeaks online: 2019-01-18
# maybe this change also still happened with the android

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-22 00:00:00') & (df.timestamp <= '2019-02-02 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
331778 2019-01-22 04:33:31 2019-01-22 03:29:20 0 days 01:04:59 0 days 01:04:00 2019-01-22 02:24:21
332132 2019-01-23 12:23:26 2019-01-23 11:19:15 0 days 02:29:59 0 days 01:04:00 2019-01-23 08:49:16
332565 2019-01-25 00:28:20 2019-01-24 23:24:09 0 days 00:34:58 0 days 01:04:00 2019-01-24 22:49:11
332658 2019-01-25 08:53:20 2019-01-25 07:49:09 0 days 00:45:00 0 days 01:04:00 2019-01-25 07:04:09
332836 2019-01-26 08:43:17 2019-01-26 07:39:06 0 days 09:19:59 0 days 01:04:00 2019-01-25 22:19:07
333832 2019-01-29 20:03:05 2019-01-29 18:58:54 0 days 00:30:00 0 days 01:04:00 2019-01-29 18:28:54
333836 2019-01-29 20:33:05 2019-01-29 19:28:54 0 days 00:15:01 0 days 01:04:00 2019-01-29 19:13:53
334519 2019-02-01 16:02:31 2019-02-01 04:58:43 0 days 00:39:58 0 days 11:04:00 2019-02-01 04:18:45
334900 2019-02-03 00:32:27 2019-02-02 13:28:39 0 days 00:50:01 0 days 11:04:00 2019-02-02 12:38:38
334903 2019-02-03 01:12:28 2019-02-02 14:08:40 0 days 00:30:02 0 days 11:04:00 2019-02-02 13:38:38
334922 2019-02-03 03:42:27 2019-02-02 16:38:39 0 days 01:00:00 0 days 11:04:00 2019-02-02 15:38:39
334991 2019-02-03 10:32:25 2019-02-02 23:28:37 0 days 01:10:00 0 days 11:04:00 2019-02-02 22:18:37
"""
# he got sick in this period during the training camp, there is nothing in the calendar
# I can't see when he travelled back..
# So now just assume it was on the day that he filled it out

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-19 00:00:00') & (df.timestamp <= '2019-03-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
347263 2019-03-21 21:24:45 2019-03-21 10:20:57 0 days 00:10:00 0 days 11:04:00 2019-03-21 10:10:57
347267 2019-03-21 21:49:44 2019-03-21 10:45:56 0 days 00:10:00 0 days 11:04:00 2019-03-21 10:35:56
347302 2019-03-22 00:54:45 2019-03-21 13:50:57 0 days 00:15:00 0 days 11:04:00 2019-03-21 13:35:57
347685 2019-03-22 23:44:31 2019-03-22 22:40:51 0 days 01:00:00 0 days 01:04:00 2019-03-22 21:40:51
"""
# not big enough of a gap again
# trainingpeaks online: 2019-03-20 as travel, and 2019-03-21 no cycling
# assume arrival on 22-03-2019 00:00:00 australia time, which is 21-03-2019 13:00:00 UTC
# 2019-03-21 12:55:56
# 2019-03-21 13:00:58

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-30 00:00:00') & (df.timestamp <= '2019-03-31 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
350130 2019-03-31 23:13:03 2019-03-31 22:09:23 0 days 00:34:59 0 days 01:04:00 2019-03-31 21:34:24
"""
# not related to travel again
# Note: he stayed in Spain here. The change was due to summer time/winter time.
# The change observed in Dexcom was on 2019-04-06, meaning that he changed it manually, but just a week later.
# This makes sense as he is using the receiver, which is not connected to the internet.
# So actual change:
# 2019-03-31 00:59:25
# 2019-03-31 01:04:24

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-21 00:00:00') & (df.timestamp <= '2019-05-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
354655 2019-05-21 19:40:35 2019-05-21 17:37:14 0 days 00:20:00 0 days 02:03:00 2019-05-21 17:17:14
354768 2019-05-22 05:15:33 2019-05-22 03:12:12 0 days 00:15:00 0 days 02:03:00 2019-05-22 02:57:12
354778 2019-05-22 07:35:33 2019-05-22 05:32:12 0 days 01:35:02 0 days 02:03:00 2019-05-22 03:57:10
"""
# again: gaps not useful
# trainingpeaks online: flight from belgium to estonia on 2019-05-22
# he changed it on 22.05.19 16:22 UTC, so he must have travelled before that
# so assume he now just changed it right after travelling himself
# 22.05.19 16:22
# 22.05.19 16:27

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-25 00:00:00') & (df.timestamp <= '2019-05-27 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
355581 2019-05-25 11:44:59 2019-05-25 08:42:05 0 days 00:30:01 0 days 03:03:00 2019-05-25 08:12:04
355583 2019-05-25 12:14:58 2019-05-25 09:12:04 0 days 00:24:58 0 days 03:03:00 2019-05-25 08:47:06
355587 2019-05-25 13:39:57 2019-05-25 10:37:03 0 days 01:10:00 0 days 03:03:00 2019-05-25 09:27:03
355656 2019-05-25 19:54:58 2019-05-25 16:52:04 0 days 00:36:00 0 days 03:03:00 2019-05-25 16:16:04
355666 2019-05-25 20:54:57 2019-05-25 17:52:03 0 days 00:15:00 0 days 03:03:00 2019-05-25 17:37:03
355690 2019-05-25 23:24:57 2019-05-25 20:22:03 0 days 00:35:00 0 days 03:03:00 2019-05-25 19:47:03
355893 2019-05-26 16:34:56 2019-05-26 13:32:02 0 days 00:30:01 0 days 03:03:00 2019-05-26 13:02:01
"""
# no info in gaps
# trainingpeaks online: it seems like the travelling happened on 2019-05-26 (probably from estonia to barcelona)
# manually changed on: 2019-05-30 (this was the day he travelled back from barcelona to australia)
# so not immediately changed and not other clues
# so assume he arrived at 2019-05-27 00:00 old local time (UTC+3)
# 2019-05-26 20:57:00
# 2019-05-26 21:02:00

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-05-28 00:00:00') & (df.timestamp <= '2019-05-31 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
356871 2019-05-30 03:19:46 2019-05-30 00:16:52 0 days 01:24:59 0 days 03:03:00 2019-05-29 22:51:53
356878 2019-05-30 04:59:46 2019-05-30 01:56:52 0 days 01:09:59 0 days 03:03:00 2019-05-30 00:46:53
356961 2019-05-30 13:38:44 2019-05-30 09:36:51 0 days 00:50:00 0 days 04:02:00 2019-05-30 08:46:51
356962 2019-05-30 14:43:43 2019-05-30 10:41:50 0 days 01:04:59 0 days 04:02:00 2019-05-30 09:36:51
356971 2019-05-30 15:38:44 2019-05-30 11:36:51 0 days 00:14:59 0 days 04:02:00 2019-05-30 11:21:52
356987 2019-05-30 18:18:43 2019-05-30 14:16:50 0 days 01:24:59 0 days 04:02:00 2019-05-30 12:51:51
356994 2019-05-30 19:03:45 2019-05-30 15:01:52 0 days 00:15:01 0 days 04:02:00 2019-05-30 14:46:51
357245 2019-05-31 22:51:53 2019-05-31 12:46:52 0 days 00:55:00 0 days 10:05:00 2019-05-31 11:51:52
357246 2019-05-31 23:46:53 2019-05-31 13:41:52 0 days 00:55:00 0 days 10:05:00 2019-05-31 12:46:52
"""
# trainingpeaks online: from 2019-05-29 to 2019-05-31
# manually changed on: 2019-05-30
# assume the manual change is correct
# 30.05.2019 19:41:50
# 30.05.2019 19:46:50

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-07-05 00:00:00') & (df.timestamp <= '2019-07-08 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# censoring between 2019-06-27 and 2019-08-12
# 27.06.2019 05:20:43
# 12.08.2019 09:31:09

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-28 00:00:00') & (df.timestamp <= '2019-09-30 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
377320 2019-09-29 16:55:11 2019-09-29 14:55:11 0 days 00:40:01 0 days 02:00:00 2019-09-29 14:15:10
"""
# manually changed on 2019-10-03 (too late)
# trainingpeaks online: 2019-09-28 to 2019-09-30
# assume he arrived on 2019-09-30 00:00 old local time (UTC+2)
# 2019-09-29 21:55:09
# 2019-09-29 22:00:10

df.loc[(df.RIDER == 6) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-05 00:00:00') & (df.timestamp <= '2019-10-19 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# australia DST 2019 is from april 7 until october 6
# manual change happened on 2019-10-14
# precise change: 2019-10-06 02:00 UTC+10 which is 2019-10-05 16:00 UTC
# 2019-10-05 15:59:50
# 2019-10-05 16:04:50

# ------------------------- rider 10

# people with iPhones also have to do manual resets (according to Simon)
for n in range(40):
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

# TODO: find out actual changes
# Just assume it was at 00:00 old local time if we cannot find out the time that they arrived

# Note there was something wrong with the date 27.05.19 04:55

df.loc[(df.RIDER == 13)\
	& (df.timestamp <= '2019-09-27 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-09-27 21:57:33
df.loc[(df.RIDER == 13)\
	& (df.timestamp >= '2019-09-27 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-09-27 22:02:33

df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-30 00:00:00') & (df.timestamp <= '2019-10-02 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
563192 2019-09-30 11:33:20 2019-09-30 09:32:27 0 days 01:55:00 0 days 02:01:00 2019-09-30 07:37:27
563199 2019-09-30 12:28:20 2019-09-30 10:27:27 0 days 00:25:00 0 days 02:01:00 2019-09-30 10:02:27
563228 2019-09-30 15:38:20 2019-09-30 13:37:27 0 days 00:50:00 0 days 02:01:00 2019-09-30 12:47:27
563232 2019-09-30 20:28:20 2019-09-30 18:27:27 0 days 04:35:00 0 days 02:01:00 2019-09-30 13:52:27
"""
# we have no idea about the date
# we now just take the one where there was 4 hours censored
df.loc[563231:563232, 'timestamp']
"""
563231   2019-09-30 13:52:27
563232   2019-09-30 18:27:27
"""

df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# trainingpeaks online: seems to be on 2019-10-16
# assume now change at 2019-10-16 22:00 (UTC)
df.loc[(df.RIDER == 13)\
	& (df.timestamp <= '2019-10-16 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 21:55:12
df.loc[(df.RIDER == 13)\
	& (df.timestamp >= '2019-10-16 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-16 22:00:12

df.loc[(df.RIDER == 13) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-20 00:00:00') & (df.timestamp <= '2019-11-01 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
567912 2019-10-24 10:25:35 2019-10-24 08:24:42 0 days 00:50:00 0 days 02:01:00 2019-10-24 07:34:42
567920 2019-10-27 22:50:24 2019-10-27 20:49:31 3 days 11:49:48 0 days 02:01:00 2019-10-24 08:59:43
"""
# Note: we don't know when he flied back
# There is a 10-day time window
# But usually, they fly back right after the race, so now we can assume the same.
df.loc[(df.RIDER == 13)\
	& (df.timestamp <= '2019-10-21 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-21 14:59:53
df.loc[(df.RIDER == 13)\
	& (df.timestamp >= '2019-10-21 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-21 15:04:54


# ------------------------- rider 14
# Information from rider:
# "With dexcom receiver I almost never change the date/time unfortunately. 
# At the time it was solely to track my bg values. Sometimes I changed it if we did a big travel 
# (Europe to US for example) but not always. Summer/winter time I never touched that."

df_changes = replace_row(df_changes, df_tp, 14, 0, np.arange(10))
df_changes = replace_row(df_changes, df_tp, 14, 10, np.arange(10, 20))

# CONCLUSION timezone change manually

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-20 00:00:00') & (df.timestamp <= '2018-12-22 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
580733 2018-12-20 04:08:07 2018-12-20 02:08:07 0 days 00:49:59 0 days 02:00:00 2018-12-20 01:18:08
580769 2018-12-20 07:43:07 2018-12-20 05:43:07 0 days 00:40:00 0 days 02:00:00 2018-12-20 05:03:07
580865 2018-12-20 16:18:06 2018-12-20 14:18:06 0 days 00:40:01 0 days 02:00:00 2018-12-20 13:38:05
580943 2018-12-21 00:53:04 2018-12-20 22:53:04 0 days 02:10:01 0 days 02:00:00 2018-12-20 20:43:03
"""
# Travel from Spain to Finland
# I'm not sure if any of the above gaps correspond to travelling
# TrainingPeaks online: 2018-12-21
# Assume arrive at 2018-12-21 23:00:00
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2018-12-21 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2018-12-21 22:58:01
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2018-12-21 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2018-12-21 23:02:59

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2018-12-24 00:00:00') & (df.timestamp <= '2018-12-26 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# travel from finland to spain canary islands (therefore ttimezone UTC+00)
# trainingpeaks online: 2018-12-25
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2018-12-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2018-12-25 21:57:45
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2018-12-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2018-12-25 22:02:45

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-01-05 00:00:00') & (df.timestamp <= '2019-01-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# travel within spain (probably from canary/faroe islands to mainland)
# trainingpeaks online: cannot find anything
# assume arrival 2019-01-07 00:00:00 UTC
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-01-07 00:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-01-06 23:56:59
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-01-07 00:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-01-07 00:01:59

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-02-12 00:00:00') & (df.timestamp <= '2019-02-14 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
595216 2019-02-12 13:37:25 2019-02-12 11:37:25 0 days 03:30:01 0 days 02:00:00 2019-02-12 08:07:24
595228 2019-02-12 15:02:25 2019-02-12 13:02:25 0 days 00:30:00 0 days 02:00:00 2019-02-12 12:32:25
"""
# travel from Spain to Rwanda
# TrainingPeaks online: probably 2019-02-13
# assume arrival at 2019-02-13 23:00:00 UTC
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-02-13 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-02-13 22:57:19
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-02-13 23:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-02-13 23:02:17

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-03-04 00:00:00') & (df.timestamp <= '2019-03-06 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# censored between 2019-02-23 and 2019-05-31
# also deleted rows 5 6 and 7 manually, and for the rows 3 and 8, used the first and last glucose date
df.loc[598356]
df.loc[598355]

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-06-17 00:00:00') & (df.timestamp <= '2019-06-18 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
598558 2019-06-17 05:24:00 2019-06-17 03:24:00 0 days 08:09:59 0 days 02:00:00 2019-06-16 19:14:01
598560 2019-06-17 08:33:59 2019-06-17 06:33:59 0 days 03:04:59 0 days 02:00:00 2019-06-17 03:29:00
598561 2019-06-17 11:08:58 2019-06-17 09:08:58 0 days 02:34:59 0 days 02:00:00 2019-06-17 06:33:59
598567 2019-06-17 11:53:59 2019-06-17 09:53:59 0 days 00:20:01 0 days 02:00:00 2019-06-17 09:33:58
598572 2019-06-18 14:53:53 2019-06-18 12:53:53 1 days 02:39:54 0 days 02:00:00 2019-06-17 10:13:59
"""
# from finland to slovenia
# no info in trainingpeaks online
# assume he arrived on 2019-06-18 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-06-17 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-06-17 10:13:59
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-06-17 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-06-18 12:53:53
# note that this is also during the last censoring

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-06-23 00:00:00') & (df.timestamp <= '2019-06-27 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
598584 2019-06-23 08:28:29 2019-06-23 06:28:29 0 days 10:44:58 0 days 02:00:00 2019-06-22 19:43:31
598586 2019-06-23 15:18:27 2019-06-23 13:18:27 0 days 06:44:58 0 days 02:00:00 2019-06-23 06:33:29
598587 2019-06-23 15:38:28 2019-06-23 13:38:28 0 days 00:20:01 0 days 02:00:00 2019-06-23 13:18:27
598588 2019-06-24 13:23:23 2019-06-24 11:23:23 0 days 21:44:55 0 days 02:00:00 2019-06-23 13:38:28
598589 2019-06-24 15:48:23 2019-06-24 13:48:23 0 days 02:25:00 0 days 02:00:00 2019-06-24 11:23:23
598591 2019-06-24 18:33:21 2019-06-24 16:33:21 0 days 02:39:58 0 days 02:00:00 2019-06-24 13:53:23
598594 2019-06-25 03:43:21 2019-06-25 01:43:21 0 days 09:00:00 0 days 02:00:00 2019-06-24 16:43:21
598595 2019-06-25 08:28:19 2019-06-25 06:28:19 0 days 04:44:58 0 days 02:00:00 2019-06-25 01:43:21
598597 2019-06-25 18:33:17 2019-06-25 16:33:17 0 days 09:59:58 0 days 02:00:00 2019-06-25 06:33:19
598601 2019-06-25 18:58:18 2019-06-25 16:58:18 0 days 00:10:01 0 days 02:00:00 2019-06-25 16:48:17
598619 2019-06-26 02:58:16 2019-06-26 00:58:16 0 days 06:34:58 0 days 02:00:00 2019-06-25 18:23:18
598620 2019-06-26 06:13:16 2019-06-26 04:13:16 0 days 03:15:00 0 days 02:00:00 2019-06-26 00:58:16
598622 2019-06-26 06:33:15 2019-06-26 04:33:15 0 days 00:14:59 0 days 02:00:00 2019-06-26 04:18:16
598625 2019-06-26 18:58:11 2019-06-26 16:58:11 0 days 12:14:56 0 days 02:00:00 2019-06-26 04:43:15
598627 2019-06-26 19:23:13 2019-06-26 17:23:13 0 days 00:20:02 0 days 02:00:00 2019-06-26 17:03:11
598631 2019-06-26 20:03:12 2019-06-26 18:03:12 0 days 00:24:59 0 days 02:00:00 2019-06-26 17:38:13
598632 2019-06-27 06:03:09 2019-06-27 04:03:09 0 days 09:59:57 0 days 02:00:00 2019-06-26 18:03:12
598634 2019-06-27 10:18:08 2019-06-27 08:18:08 0 days 04:09:59 0 days 02:00:00 2019-06-27 04:08:09
598635 2019-06-27 14:23:09 2019-06-27 12:23:09 0 days 04:05:01 0 days 02:00:00 2019-06-27 08:18:08
598636 2019-06-27 15:48:07 2019-06-27 13:48:07 0 days 01:24:58 0 days 02:00:00 2019-06-27 12:23:09
598640 2019-06-27 16:13:08 2019-06-27 14:13:08 0 days 00:10:02 0 days 02:00:00 2019-06-27 14:03:06
598642 2019-06-27 16:33:07 2019-06-27 14:33:07 0 days 00:14:59 0 days 02:00:00 2019-06-27 14:18:08
"""
# it seems like there are not too many glucose measurements during these days anyway
# trainingpeaks online: probably travel took place between 2019-06-24 and 2019-06-26
# assume arrival 2019-06-26 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-06-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-06-25 18:23:18
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-06-25 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-06-26 00:58:16

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-01 00:00:00') & (df.timestamp <= '2019-08-02 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# censoring between 2019-07-12 and 2019-08-03
df.loc[601600:601601, 'timestamp']
"""
601600   2019-07-12 06:11:54
601601   2019-08-03 10:35:39
"""
# use the censored min and max as change dates

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-09 00:00:00') & (df.timestamp <= '2019-08-11 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
603817 2019-08-11 08:36:39 2019-08-11 05:29:57 0 days 00:59:58 0 days 03:07:00 2019-08-11 04:29:59
603825 2019-08-12 00:07:55 2019-08-11 21:01:13 0 days 14:56:16 0 days 03:07:00 2019-08-11 06:04:57
"""
# from poland to finland
# trainingpeaks online: probably on 2019-08-10
# assume arrival on 2019-08-11 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-08-10 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-08-10 21:59:59
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-08-10 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-10 22:04:59

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-12 00:00:00') & (df.timestamp <= '2019-08-14 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
603878 2019-08-12 05:02:53 2019-08-12 01:56:11 0 days 00:35:01 0 days 03:07:00 2019-08-12 01:21:10
"""
# from finland to spain
# trainingpeaks online: probably on 2019-08-13
# assume arrival on 2019-08-14 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-08-13 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-08-13 20:56:05
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-08-13 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-13 21:01:03

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-08-25 00:00:00') & (df.timestamp <= '2019-08-27 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
# empty
# travel from Denmark to Finland
# trainingpeaks online: most likely on 2019-08-26
# assume arrival on 2019-08-27 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-08-26 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-08-25 14:15:19
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-08-26 22:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-08-30 22:00:19
# Note: some censoring here

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-09-05 00:00:00') & (df.timestamp <= '2019-09-06 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
df.loc[607595:607605, 'timestamp']
# censoring 
# - between 2019-08-31 and 2019-09-10
# - between 2019-09-10 and 2019-10-11
# remove row 15 manually
# merge row 14 and 16 manually
df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
607636 2019-10-15 09:03:53 2019-10-15 05:57:11 0 days 11:55:00 0 days 03:07:00 2019-10-14 18:02:11
607637 2019-10-15 09:13:52 2019-10-15 06:07:10 0 days 00:09:59 0 days 03:07:00 2019-10-15 05:57:11
607638 2019-10-15 09:38:53 2019-10-15 06:32:11 0 days 00:25:01 0 days 03:07:00 2019-10-15 06:07:10
607639 2019-10-15 11:48:52 2019-10-15 08:42:10 0 days 02:09:59 0 days 03:07:00 2019-10-15 06:32:11
607647 2019-10-15 12:33:50 2019-10-15 09:27:08 0 days 00:09:58 0 days 03:07:00 2019-10-15 09:17:10
607648 2019-10-15 15:13:50 2019-10-15 12:07:08 0 days 02:40:00 0 days 03:07:00 2019-10-15 09:27:08
607649 2019-10-15 21:53:49 2019-10-15 18:47:07 0 days 06:39:59 0 days 03:07:00 2019-10-15 12:07:08
607650 2019-10-16 00:18:49 2019-10-15 21:12:07 0 days 02:25:00 0 days 03:07:00 2019-10-15 18:47:07
607653 2019-10-16 10:33:50 2019-10-16 07:27:08 0 days 10:05:01 0 days 03:07:00 2019-10-15 21:22:07
607655 2019-10-16 10:53:47 2019-10-16 07:47:05 0 days 00:14:57 0 days 03:07:00 2019-10-16 07:32:08
607656 2019-10-16 12:23:47 2019-10-16 09:17:05 0 days 01:30:00 0 days 03:07:00 2019-10-16 07:47:05
607657 2019-10-16 12:43:48 2019-10-16 09:37:06 0 days 00:20:01 0 days 03:07:00 2019-10-16 09:17:05
607658 2019-10-16 12:58:48 2019-10-16 09:52:06 0 days 00:15:00 0 days 03:07:00 2019-10-16 09:37:06
"""
# travel from finland to japan
# assume arrival on 2019-10-17 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-10-16 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 09:57:06
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-10-16 21:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-18 02:12:00

df.loc[(df.RIDER == 14) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-20 00:00:00') & (df.timestamp <= '2019-10-23 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
607684 2019-10-20 03:13:34 2019-10-20 00:06:52 0 days 00:15:01 0 days 03:07:00 2019-10-19 23:51:51
607685 2019-10-20 03:23:34 2019-10-20 00:16:52 0 days 00:10:00 0 days 03:07:00 2019-10-20 00:06:52
607686 2019-10-20 03:38:34 2019-10-20 00:31:52 0 days 00:15:00 0 days 03:07:00 2019-10-20 00:16:52
607689 2019-10-20 03:58:35 2019-10-20 00:51:53 0 days 00:10:01 0 days 03:07:00 2019-10-20 00:41:52
607690 2019-10-20 04:13:37 2019-10-20 01:06:55 0 days 00:15:02 0 days 03:07:00 2019-10-20 00:51:53
607691 2019-10-20 04:43:35 2019-10-20 01:36:53 0 days 00:29:58 0 days 03:07:00 2019-10-20 01:06:55
607692 2019-10-20 05:23:34 2019-10-20 02:16:52 0 days 00:39:59 0 days 03:07:00 2019-10-20 01:36:53
607693 2019-10-20 05:43:33 2019-10-20 02:36:51 0 days 00:19:59 0 days 03:07:00 2019-10-20 02:16:52
607694 2019-10-20 05:53:33 2019-10-20 02:46:51 0 days 00:10:00 0 days 03:07:00 2019-10-20 02:36:51
607695 2019-10-20 06:08:35 2019-10-20 03:01:53 0 days 00:15:02 0 days 03:07:00 2019-10-20 02:46:51
607699 2019-10-20 06:38:34 2019-10-20 03:31:52 0 days 00:14:59 0 days 03:07:00 2019-10-20 03:16:53
607700 2019-10-21 02:58:30 2019-10-20 23:51:48 0 days 20:19:56 0 days 03:07:00 2019-10-20 03:31:52
607702 2019-10-21 03:13:30 2019-10-21 00:06:48 0 days 00:10:00 0 days 03:07:00 2019-10-20 23:56:48
607706 2019-10-22 18:08:25 2019-10-22 15:01:43 1 days 14:39:55 0 days 03:07:00 2019-10-21 00:21:48
607729 2019-10-22 21:33:24 2019-10-22 18:26:42 0 days 01:34:59 0 days 03:07:00 2019-10-22 16:51:43
607731 2019-10-23 06:43:23 2019-10-23 03:36:41 0 days 09:04:59 0 days 03:07:00 2019-10-22 18:31:42
607732 2019-10-23 08:33:22 2019-10-23 05:26:40 0 days 01:49:59 0 days 03:07:00 2019-10-23 03:36:41
607733 2019-10-23 11:28:21 2019-10-23 08:21:39 0 days 02:54:59 0 days 03:07:00 2019-10-23 05:26:40
607734 2019-10-23 13:38:21 2019-10-23 10:31:39 0 days 02:10:00 0 days 03:07:00 2019-10-23 08:21:39
607735 2019-10-23 16:03:20 2019-10-23 12:56:38 0 days 02:24:59 0 days 03:07:00 2019-10-23 10:31:39
607736 2019-10-23 16:18:20 2019-10-23 13:11:38 0 days 00:15:00 0 days 03:07:00 2019-10-23 12:56:38
607738 2019-10-23 16:33:21 2019-10-23 13:26:39 0 days 00:10:01 0 days 03:07:00 2019-10-23 13:16:38
607739 2019-10-23 16:53:20 2019-10-23 13:46:38 0 days 00:19:59 0 days 03:07:00 2019-10-23 13:26:39
607740 2019-10-23 17:13:20 2019-10-23 14:06:38 0 days 00:20:00 0 days 03:07:00 2019-10-23 13:46:38
607742 2019-10-23 17:33:20 2019-10-23 14:26:38 0 days 00:15:00 0 days 03:07:00 2019-10-23 14:11:38
607743 2019-10-23 17:48:20 2019-10-23 14:41:38 0 days 00:15:00 0 days 03:07:00 2019-10-23 14:26:38
607761 2019-10-23 20:23:21 2019-10-23 17:16:39 0 days 01:10:01 0 days 03:07:00 2019-10-23 16:06:38
"""
# assume arrival on 2019-10-23 00:00 old local time
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-10-22 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-21 00:21:48
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-10-22 15:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-22 15:01:43

# last change is from summer time to winter time
# this change happens at 01:00 UTC
df.loc[(df.RIDER == 14)\
	& (df.timestamp <= '2019-10-27 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-27 00:21:26
df.loc[(df.RIDER == 14)\
	& (df.timestamp >= '2019-10-27 01:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-27 08:31:25

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

check_minmax(df_changes, df_tp, (15,0), (15,0))
check_minmax(df_changes, df_tp, (15,1), (15,7))
check_minmax(df_changes, df_tp, (15,2), (15,8))
check_minmax(df_changes, df_tp, (15,3), (15,9))
check_minmax(df_changes, df_tp, (15,4), (15,8))

# CONCLUSION timezone change manually

df.loc[624861:624862, 'timestamp']

df.loc[(df.RIDER == 15) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-05 00:00:00') & (df.timestamp <= '2019-10-07 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
625472 2019-10-07 10:39:59 2019-10-07 08:39:59 3 days 23:54:43 0 days 02:00:00 2019-10-03 08:45:16
625569 2019-10-07 18:54:58 2019-10-07 16:54:58 0 days 00:14:59 0 days 02:00:00 2019-10-07 16:39:59
"""
# travel from France to China
# reset on 2019-10-07 20:59
# maybe just assume he travelled exactly in the 3 days gap
df.loc[625471:625472, 'timestamp']
"""
625471   2019-10-03 08:45:16
625472   2019-10-07 08:39:59
"""

df.loc[(df.RIDER == 15) & (df['censoring'] > '6min') \
	& (df.timestamp >= '2019-10-15 00:00:00') & (df.timestamp <= '2019-10-17 23:59:59'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']]
"""
           local_timestamp           timestamp       censoring        timezone     timestamp_shift
628061 2019-10-17 03:18:26 2019-10-16 19:19:25 0 days 00:45:01 0 days 07:59:00 2019-10-16 18:34:24
"""
# travel from China to Japan
# assume arrival on 2019-10-17 00:00 old local time
df.loc[(df.RIDER == 15)\
	& (df.timestamp <= '2019-10-16 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[-1]['timestamp']
# 2019-10-16 15:59:24
df.loc[(df.RIDER == 15)\
	& (df.timestamp >= '2019-10-16 16:00:00'),
	['local_timestamp', 'timestamp', 'censoring', 'timezone', 'timestamp_shift']].iloc[0]['timestamp']
# 2019-10-16 16:04:24

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

del df, df_tp, df_changes ; gc.collect()


# note that this file was called df_changes before (TODO: maybe clean up code and rename)
df_dc = pd.read_csv(path+'timezone/dexcom_changes_manual.csv', index_col=[0,1])
df_dc.drop(['index', 'Source Device ID', 'Transmitter ID', 'device_change', 'transmitter_change', 
	'trust_time', 'Unnamed: 15'], axis=1, inplace=True)

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