# mask timezone changes in the trainingpeaks data for which we do not have glucose
df_tz['glucose'] = False
for (i, n), (ts_min, ts_max) in df_tz[['local_timestamp_min', 'local_timestamp_max']].iterrows():
	if not df_egv.loc[(df_egv.RIDER == i) & (df_egv.local_timestamp >= ts_min) & (df_egv.local_timestamp <= ts_max)].empty:
		df_tz.loc[(i,n), 'glucose'] = True




# fill up more accurate timezone changes
tz_extra_window = datetime.timedelta(days=3)

notfound = pd.DataFrame(columns=['RIDER', 't_min', 't_max', 'tz_diff'])
found = pd.DataFrame(columns=['RIDER', 't_min', 't_max', 'tz_diff'])

df_tz['local_timestamp_final_min'] = np.nan ; df_tz['local_timestamp_final_min'] = df_tz['local_timestamp_final_min'].astype(object)
df_tz['local_timestamp_final_max'] = np.nan ; df_tz['local_timestamp_final_max'] = df_tz['local_timestamp_final_max'].astype(object)
df_tz['local_timestamp_final_diff'] = np.nan ; df_tz['local_timestamp_final_diff'] = df_tz['local_timestamp_final_diff'].astype(object)

for idx in df_tz.index:
	df_tz.at[idx, 'local_timestamp_final_min'] = []
	df_tz.at[idx, 'local_timestamp_final_max'] = []
	df_tz.at[idx, 'local_timestamp_final_diff'] = []

for i in df_egv.loc[df_egv['change'], 'RIDER'].unique():
	mask = (df_egv['change']) & (df_egv.RIDER == i)
	for idx, (t_min, tz_diff) in df_egv.loc[mask, ['local_timestamp', 'timediff']].iterrows():
		count_found = 0
		t_max = df_egv.loc[idx-1, 'local_timestamp']

		for n in df_tz.loc[i].index[:-1]:
			if (df_tz.loc[i].loc[n, 'local_timestamp_max'] <= t_max) & (df_tz.loc[i].loc[n+1, 'local_timestamp_min']+tz_extra_window >= t_min):
				df_tz.at[(i,n+1), 'local_timestamp_final_min'].append(str(t_min))
				df_tz.at[(i,n), 'local_timestamp_final_max'].append(str(t_max))
				df_tz.at[(i,n), 'local_timestamp_final_diff'].append(str(tz_diff))#.round('h')
				found.loc[idx] = [i, t_min, t_max, tz_diff]
				count_found += 1
		if count_found == 0:
			print(idx, i,  t_min, t_max, tz_diff)
			notfound.loc[idx] = [i, t_min, t_max, tz_diff]
		elif count_found > 1:
			print("Found more than once: ", idx, i,  t_min, t_max, tz_diff)

df_min = df_tz['local_timestamp_final_min'].apply(pd.Series).add_prefix('local_timestamp_final_min_')
df_max = df_tz['local_timestamp_final_max'].apply(pd.Series).add_prefix('local_timestamp_final_max_')
df_diff = df_tz['local_timestamp_final_diff'].apply(pd.Series).add_prefix('local_timestamp_final_diff_')

df_tz = df_tz.join([df_min, df_max, df_diff])[df_tz.columns.tolist() + \
	['local_timestamp_final_%s_%s'%(j,i) for i in range(df_min.shape[1]) for j in ('min', 'max', 'diff')]]
df_tz.drop(['local_timestamp_final_min', 'local_timestamp_final_max', 'local_timestamp_final_diff'], axis=1, inplace=True)
df_tz.to_csv(path+'timezone/check_tz.csv')

df_tz_backup = df_tz.copy()

def insert_rows(df, idx_insert, notfound, idx):
	"""
	Insert rows of time changes that were not identified from training peaks
	e.g. can happen if they have an intermediate stop in their travels
	
	Arguments:
		df 			dataframe with timezone changes (usually df_tz)
		idx_insert	index in (RIDER, n) where to insert in df
		notfound	list of timezone changes that are not found
		idx 		index in notfound of specific timezone change we want to insert

	Returns
		dataframe with missing timezone change inserted
	"""
	i = df.index.get_loc(idx_insert)
	if isinstance(i, slice):
		i = i.start

	df_A = df.iloc[:i]

	rows = pd.DataFrame.from_records(data=(len(idx)+1)*[df.iloc[i]], 
		index=pd.MultiIndex.from_tuples((len(idx)+1)*[idx_insert], names=['RIDER', 'n']))

	for j, k in enumerate(idx):
		rows.loc[:, 'date_max'].iloc[j] = notfound.loc[k, 't_max'].date()
		rows.loc[:, 'date_min'].iloc[j+1] = notfound.loc[k, 't_min'].date()

		rows.loc[:, 'local_timestamp_max'].iloc[j] = notfound.loc[k, 't_max']
		rows.loc[:, 'local_timestamp_min'].iloc[j+1] = notfound.loc[k, 't_min']

		rows.loc[:, 'local_timestamp_final_max_0'].iloc[j] = notfound.loc[k, 't_max']
		rows.loc[:, 'local_timestamp_final_min_0'].iloc[j+1] = notfound.loc[k, 't_min']

		rows.loc[:, 'local_timestamp_final_diff_0'].iloc[j] = notfound.loc[k, 'tz_diff']#.round('h')

		rows.loc[:, 'timezone'].iloc[j+1] = rows.loc[:, 'timezone'].iloc[j] + notfound.loc[k, 'tz_diff']#.round('h')

		rows.loc[:, 'timestamp_max'].iloc[j] = notfound.loc[k, 't_max'] - rows.loc[:, 'timezone'].iloc[j]
		rows.loc[:, 'timestamp_min'].iloc[j+1] = notfound.loc[k, 't_min'] - rows.loc[:, 'timezone'].iloc[j+1]

	df_B = df.iloc[i+1:]

	return df_A.append(rows).append(df_B)

def substitute_rows(df, idx_subs, notfound, idx):
	i0 = df.index.get_loc(idx_subs[0]).start
	i1 = df.index.get_loc(idx_subs[1]).start

	df_A = df.iloc[:i0]

	rows = pd.DataFrame.from_records(data=len(idx)*[df.iloc[i0]]+[df.iloc[i1]],
		index=pd.MultiIndex.from_tuples(len(idx)*[idx_subs[0]]+[idx_subs[1]], names=['RIDER', 'n']))

	for j, k in enumerate(idx):
		rows.loc[:, 'date_max'].iloc[j] = notfound.loc[k, 't_max'].date()
		rows.loc[:, 'date_min'].iloc[j+1] = notfound.loc[k, 't_min'].date()

		rows.loc[:, 'local_timestamp_max'].iloc[j] = notfound.loc[k, 't_max']
		rows.loc[:, 'local_timestamp_min'].iloc[j+1] = notfound.loc[k, 't_min']

		rows.loc[:, 'local_timestamp_final_max_0'].iloc[j] = notfound.loc[k, 't_max']
		rows.loc[:, 'local_timestamp_final_min_0'].iloc[j+1] = notfound.loc[k, 't_min']

		rows.loc[:, 'local_timestamp_final_diff_0'].iloc[j] = notfound.loc[k, 'tz_diff']#.round('h')

		rows.loc[:, 'timezone'].iloc[j+1] = rows.loc[:, 'timezone'].iloc[j] + notfound.loc[k, 'tz_diff']#.round('h')

		rows.loc[:, 'timestamp_max'].iloc[j] = notfound.loc[k, 't_max'] - rows.loc[:, 'timezone'].iloc[j]
		rows.loc[:, 'timestamp_min'].iloc[j+1] = notfound.loc[k, 't_min'] - rows.loc[:, 'timezone'].iloc[j+1]

	df_B = df.iloc[i1+1:]

	return df_A.append(rows).append(df_B)

def print_row(text, row):
	print(text, row.name, *tuple(row.to_list()))

def get_timezone_entry(rider, loc):
	return [rider, df_egv.loc[loc, 'local_timestamp'],
				   df_egv.loc[loc-1, 'local_timestamp'],
				   df_egv.loc[loc, 'timediff']]

def get_mints_after_censoring(i,n):
	"""
	After censoring
	"""
	return df_egv.loc[(df_egv.RIDER == i) \
		& (df_egv.local_timestamp >= df_tz.loc[(i,n-1), 'local_timestamp_max']), 'local_timestamp'].iloc[0]

def get_maxts_before_censoring(i, n):
	"""
	Before censoring
	"""
	return df_egv.loc[(df_egv.RIDER == i) \
		& (df_egv.local_timestamp <= df_tz.loc[(i,n+1), 'local_timestamp_min']), 'local_timestamp'].iloc[-1]


def merge_rows(df_tz, x, y, keep_tz='first'):
	"""
	When timezone change is not found
	"""
	df_tz.loc[x, 'date_max'] = df_tz.loc[y, 'date_max']
	df_tz.loc[x, 'timestamp_max'] = df_tz.loc[y, 'timestamp_max']
	df_tz.loc[x, 'local_timestamp_max'] = df_tz.loc[y, 'local_timestamp_max']
	df_tz.loc[x, 'local_timestamp_final_max_0'] = df_tz.loc[y, 'local_timestamp_final_max_0']
	df_tz.loc[x, 'local_timestamp_final_diff_0'] = df_tz.loc[y, 'local_timestamp_final_diff_0']

	if keep_tz == 'last':
		df_tz.loc[x, 'timezone'] = df_tz.loc[y, 'timezone']

	df_tz.drop(y, inplace=True)
	return df_tz

"""
# NOTE: every time there's a new transmitter we kind of have to re-evaluate, as we do not take these time changes into account
# So if they reset the time or were travelling, WHILE they changed their transmitter,
# We might not catch this time reset or travelling
        RIDER     local_timestamp local_timestamp_diff
0           1 2018-12-01 00:00:47                  NaT
784         1 2018-12-03 20:05:51      0 days 02:50:11
12419       1 2019-01-18 23:25:45      0 days 02:16:37
38606       1 2019-05-01 12:00:44      0 days 02:09:52
68078       1 2019-08-19 23:12:32      0 days 23:47:06
95479       2 2018-12-18 22:27:42  -348 days +22:29:22
109250      2 2019-03-30 01:41:44      2 days 08:12:00
128806      2 2019-07-08 01:48:49     14 days 19:29:22
133098      2 2019-08-04 11:12:07      3 days 12:49:29
138738      2 2019-10-06 01:11:41     12 days 03:11:46
141574      3 2018-12-01 05:41:40  -319 days +00:35:21
144414      3 2019-01-14 13:15:27     32 days 19:59:20
144578      3 2019-01-18 23:51:05      1 days 06:25:49
149837      3 2019-06-11 16:49:01     67 days 06:36:33
165245      3 2019-08-12 12:23:07      2 days 18:01:28
189577      3 2019-11-21 20:17:56      2 days 02:19:26
192192      4 2018-12-01 00:03:27  -365 days +00:05:42
194772      4 2018-12-15 15:13:54      0 days 20:50:48
209886      4 2019-03-23 00:45:10      0 days 02:45:50
216787      4 2019-09-04 06:38:47    111 days 17:56:00
233976      5 2019-01-01 00:01:48  -334 days +00:06:06
238815      5 2019-01-19 10:11:35      1 days 02:05:32
264556      5 2019-04-30 00:51:21      0 days 02:42:31
288042      5 2019-08-08 00:19:28      0 days 07:07:38
314924      5 2019-11-21 01:06:52      0 days 02:17:13
317626      6 2018-12-01 08:01:41  -365 days +09:00:19
323067      6 2018-12-21 23:36:24      0 days 12:11:36
348603      6 2019-03-27 09:58:15      0 days 04:33:59
363876      6 2019-08-12 11:31:09     45 days 20:05:25
375875      6 2019-09-25 09:50:25      0 days 12:51:09
394064     10 2018-12-11 17:04:32  -355 days +17:07:13
403959     10 2019-01-18 23:56:04      0 days 02:57:38
422499     10 2019-04-28 08:55:37      1 days 08:27:22
437865     10 2019-08-07 12:14:33      0 days 13:41:23
450279     12 2018-12-01 00:00:23  -343 days +19:25:40
455460     12 2018-12-21 21:09:37      0 days 03:34:59
464567     12 2019-02-17 07:48:03     20 days 15:29:19
473859     12 2019-10-27 22:52:40    210 days 11:07:07
482923     13 2018-12-01 00:01:51  -365 days +00:05:27
496844     13 2019-01-19 00:20:39      0 days 02:16:16
521859     13 2019-04-19 19:27:14      0 days 03:04:04
544785     13 2019-07-23 23:11:12      1 days 01:28:23
562323     13 2019-10-14 23:36:11      3 days 00:38:19
574664     14 2018-12-01 10:14:19  -365 days +10:17:10
586062     14 2019-01-19 00:48:56      0 days 02:12:41
595437     14 2019-05-31 11:30:24     96 days 14:38:43
600906     14 2019-08-12 00:07:55      0 days 14:56:16
605433     14 2019-11-08 02:05:04      0 days 15:32:40
611621     15 2018-12-01 00:03:13  -365 days +00:04:46
619675     15 2019-01-19 00:19:46      0 days 02:13:44
621934     15 2019-07-21 14:27:36    174 days 22:33:13
622183     15 2019-08-21 08:17:53      0 days 18:26:41
"""

print("--------- Fix timezones - RIDER 1")

# 20457 1 2019-02-21 21:14:09 2019-02-21 19:14:10 0 days 00:59:59
print_row("FILL\n", notfound.loc[20457])
df_tz.loc[(1,0), 'local_timestamp_final_max_0'] = notfound.loc[20457, 't_max'] # '2019-02-21 21:14:09'
df_tz.loc[(1,1), 'local_timestamp_final_min_0'] = notfound.loc[20457, 't_min'] # '2019-02-21 19:14:10'
df_tz.loc[(1,0), 'local_timestamp_final_diff_0'] = notfound.loc[20457, 'tz_diff']#.round('h') #'0 days 01:59:59'
notfound.drop(20457, inplace=True)

# 25489 1 2019-03-13 18:58:11 2019-03-13 19:53:11 -1 days +23:00:00
# Clock of the trainingpeaks devices was reset 15 days later than of the dexcom
print_row("FILL\n", notfound.loc[25489])
df_tz.loc[(1,1), 'local_timestamp_final_max_0'] = notfound.loc[25489, 't_max'] # '2019-03-13 19:53:11'
df_tz.loc[(1,2), 'local_timestamp_final_min_0'] = notfound.loc[25489, 't_min'] # '2019-03-13 18:58:11'
df_tz.loc[(1,1), 'local_timestamp_final_diff_0'] = notfound.loc[25489, 'tz_diff']#.round('h') #'-1 days +23:05:00'
notfound.drop(25489, inplace=True)

# 48947 1 2019-06-08 07:58:55 2019-06-08 13:53:55 -1 days +18:00:00
# Dexcom is 5 days too late
print_row("FILL\n", notfound.loc[48947])
df_tz.loc[(1,4), 'local_timestamp_final_max_0'] = notfound.loc[48947, 't_max'] #'2019-06-08 13:53:55'
df_tz.loc[(1,5), 'local_timestamp_final_min_0'] = notfound.loc[48947, 't_min'] #'2019-06-08 07:58:55'
df_tz.loc[(1,4), 'local_timestamp_final_diff_0'] = notfound.loc[48947, 'tz_diff']#.round('h') #'-1 days +18:00:00'
notfound.drop(48947, inplace=True)

# 73830 1 2019-09-10 21:21:36 2019-09-10 15:16:37 0 days 05:59:59
# Dexcom is 4 days too late
print_row("FILL\n", notfound.loc[73830])
df_tz.loc[(1,5), 'local_timestamp_final_max_0'] = notfound.loc[73830, 't_max'] #'2019-09-10 15:16:37'
df_tz.loc[(1,6), 'local_timestamp_final_min_0'] = notfound.loc[73830, 't_min'] #'2019-09-10 21:21:36'
df_tz.loc[(1,5), 'local_timestamp_final_diff_0'] = notfound.loc[73830, 'tz_diff']#.round('h') #'0 days 06:00:00'
notfound.drop(73830, inplace=True)

# Note: here it switches constantly between devices with different timezones
# (so one of them is probably still in the old timezone)
# Note: we should still change the time of these measurements
print("INSERT\n", notfound.loc[[74791, 74792, 74797, 74805, 75503, 75514, 76081, 76085,76655, 76658]])

# 74791 1 2019-09-14 03:10:59 2019-09-14 09:11:31 -1 days +17:59:16
# 74792 1 2019-09-14 09:16:32 2019-09-14 03:10:59 0 days 06:00:45
# 74797 1 2019-09-14 03:40:48 2019-09-14 09:36:31 -1 days +17:59:17
# 74805 1 2019-09-14 10:21:30 2019-09-14 04:15:46 0 days 06:00:44
# 75503 1 2019-09-17 03:35:36 2019-09-17 09:31:24 -1 days +17:59:12
# 75514 1 2019-09-17 10:31:24 2019-09-17 04:25:36 0 days 06:00:48
# 76081 1 2019-09-19 03:45:28 2019-09-19 09:41:19 -1 days +17:59:09
# 76085 1 2019-09-19 10:06:19 2019-09-19 04:00:28 0 days 06:00:51
# 76655 1 2019-09-21 03:35:19 2019-09-21 09:31:14 -1 days +17:59:05
# 76658 1 2019-09-21 09:51:13 2019-09-21 03:45:19 0 days 06:00:54
df_tz = insert_rows(df_tz, (1,6), notfound, 
	[74791, 74792, 74797, 74805, 75503, 75514, 76081, 76085,76655, 76658])
notfound.drop([74791, 74792, 74797, 74805, 75503, 75514, 76081, 76085,76655, 76658], inplace=True)


print("--------- Fix timezones - RIDER 2")

# 105308 2 2019-03-10 01:00:27 2019-03-02 09:15:46 -1 days +21:00:00
print_row("FILL\n", notfound.loc[105308])
df_tz.loc[(2,1), 'local_timestamp_final_max_0'] = notfound.loc[105308, 't_max'] #'2019-03-10 01:00:27'
df_tz.loc[(2,2), 'local_timestamp_final_min_0'] = notfound.loc[105308, 't_min'] #'2019-03-02 09:15:46'
df_tz.loc[(2,1), 'local_timestamp_final_diff_0'] = notfound.loc[105308, 'tz_diff']#.round('h') #'-1 days +21:00:00'
notfound.drop(105308, inplace=True)

# Found more than once:  122780 2 2019-05-22 14:55:17 2019-05-22 13:50:17 0 days 01:00:00
print_row("REMOVE duplicate\n", found.loc[122780])
df_tz.loc[(2,4), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(2,5), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(2,4), 'local_timestamp_final_diff_1'] = np.nan

# Found more than once:  123724 2 2019-05-26 05:05:11 2019-05-26 05:55:12 -1 days +22:59:59
print_row("REMOVE duplicate\n", found.loc[123724])
df_tz.loc[(2,5), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(2,6), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(2,5), 'local_timestamp_final_diff_1'] = np.nan


print("--------- Fix timezones - RIDER 3")

# Found more than once:  181887 3 2019-10-22 18:19:49 2019-10-23 01:14:49 -1 days +17:00:00
print_row("REMOVE duplicate\n", found.loc[181887])
df_tz.loc[(3,11), 'local_timestamp_final_max_0'] = df_tz.loc[(3,11), 'local_timestamp_final_max_1']
df_tz.loc[(3,11), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(3,12), 'local_timestamp_final_min_0'] = df_tz.loc[(3,12), 'local_timestamp_final_min_1']
df_tz.loc[(3,12), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(3,11), 'local_timestamp_final_diff_0'] = df_tz.loc[(3,11), 'local_timestamp_final_diff_1']
df_tz.loc[(3,11), 'local_timestamp_final_diff_1'] = np.nan


print("--------- Fix timezones - RIDER 4")

# Double: insert behind
# 4 2018-12-09 05:13:14 2018-12-08 19:08:15 (+10h)

# 4 2018-12-09 07:32:28 2018-12-09 05:28:12 0 days 02:00:00
notfound.loc[193552] = get_timezone_entry(4, 193552)

# 4 2018-12-09 17:08:13 2018-12-09 08:12:28 -1 days +12:00:00
notfound.loc[193561] = get_timezone_entry(4, 193561)

# 4 2018-12-10 21:18:07 2018-12-10 11:13:07 0 days 10:00:00
notfound.loc[193779] = get_timezone_entry(4, 193779)

df_tz.loc[(4,0), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(4,1), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(4,0), 'local_timestamp_final_diff_1'] = np.nan

df_tz.loc[(4,0), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(4,1), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(4,0), 'local_timestamp_final_diff_2'] = np.nan

df_tz.loc[(4,0), 'local_timestamp_final_max_3'] = np.nan
df_tz.loc[(4,1), 'local_timestamp_final_min_3'] = np.nan
df_tz.loc[(4,0), 'local_timestamp_final_diff_3'] = np.nan

print("INSERT\n", notfound.loc[[193552, 193561, 193779]])

df_tz = insert_rows(df_tz, (4,1), notfound, 
	[193552, 193561, 193779])
notfound.drop([193552, 193561, 193779], inplace=True)

# 213126 4 2019-04-07 08:14:31 2019-04-06 10:29:34 -1 days +23:00:00
print_row("FILL\n", notfound.loc[213126])
df_tz.loc[(4,5), 'local_timestamp_final_max_0'] = notfound.loc[213126, 't_max'] #'2019-04-07 08:14:31'
df_tz.loc[(4,6), 'local_timestamp_final_min_0'] = notfound.loc[213126, 't_min'] #'2019-04-06 10:29:34'
df_tz.loc[(4,5), 'local_timestamp_final_diff_0'] = notfound.loc[213126, 'tz_diff']#.round('h') #'-1 days +23:00:00'
notfound.drop(213126, inplace=True)

# 222955 4 2019-10-06 03:03:11 2019-10-06 01:58:11 0 days 01:00:00
# Seems like there is some intermediate travelling here
# 223714 4 2019-10-08 21:13:07 2019-10-09 00:08:07 -1 days +21:00:00
notfound.loc[223714] = get_timezone_entry(4, 223714)

print("SUBSTITUTE\n", notfound.loc[[222955, 223714]])
df_tz = substitute_rows(df_tz, [(4,10), (4,11)], notfound, [222955, 223714])
notfound.drop([222955, 223714], inplace=True)

# Combine two travels (do this before the next one, otherwise difficult)
# 4 2019-10-20 20:12:50 2019-10-20 19:07:51 0 days 01:00:00
df_tz.loc[(4,11), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(4,12), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(4,11), 'local_timestamp_final_diff_1'] = np.nan

# 4 2019-10-23 20:37:45 2019-10-23 18:32:46 0 days 02:00:00
notfound.loc[227457] = get_timezone_entry(4, 227457)

# 4 2019-10-20 20:12:50 2019-10-20 19:07:51 0 days 01:00:00
notfound.loc[227066] = get_timezone_entry(4, 227066)

print("SUBSTITUTE\n", notfound.loc[[227457, 227066]])
df_tz = substitute_rows(df_tz, [(4,11), (4,12)], notfound, [227066, 227457])
notfound.drop([227066, 227457], inplace=True)

# It seems it switches back and forth between the recording device and the phone
# They are both in different timezones, therefore it constantly switches
df_tz.loc[(4,10), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(4,10), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(4,11), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(4,11), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(4,10), 'local_timestamp_final_diff_1'] = np.nan
df_tz.loc[(4,10), 'local_timestamp_final_diff_2'] = np.nan

# 4 2019-10-09 22:00:06 2019-10-10 21:58:03 -2 days +23:57:03
notfound.loc[224283] = get_timezone_entry(4, 224283)

# 4 2019-10-10 10:15:07 2019-10-11 10:13:04 -2 days +23:57:03
notfound.loc[224410] = get_timezone_entry(4, 224410)

print("INSERT\n", notfound.loc[[224283, 224341, 224410, 224417, 224646, 
	224655, 224934, 224951, 225217, 225233, 225499, 225514]])
#224283	 4 2019-10-09 22:00:06 2019-10-10 21:58:03 -1 days +00:02:03 # TODO
# 224341 4 2019-10-11 04:18:04 2019-10-10 04:10:05 1 days 00:02:59
#224410	 4 2019-10-10 10:15:07 2019-10-11 10:13:04 -1 days +00:02:03 # TODO
# 224417 4 2019-10-11 11:13:03 2019-10-10 10:45:05 1 days 00:02:58
# 224646 4 2019-10-11 09:30:03 2019-10-12 09:28:02 -2 days +23:57:01
# 224655 4 2019-10-12 10:18:03 2019-10-11 10:10:05 1 days 00:02:58
# 224934 4 2019-10-12 09:30:02 2019-10-13 09:28:01 -2 days +23:57:01
# 224951 4 2019-10-13 10:58:00 2019-10-12 10:50:02 1 days 00:02:58
# 225217 4 2019-10-13 09:05:01 2019-10-14 09:03:00 -2 days +23:57:01
# 225233 4 2019-10-14 10:27:59 2019-10-13 10:20:01 1 days 00:02:58
# 225499 4 2019-10-14 08:35:01 2019-10-15 08:32:58 -2 days +23:57:03
# 225514 4 2019-10-15 09:52:57 2019-10-14 09:45:00 1 days 00:02:57
df_tz = insert_rows(df_tz, (4,11), notfound, 
	[224283, 224341, 224410, 224417, 224646, 224655, 224934, 224951, 
	225217, 225233, 225499, 225514])
notfound.drop([224283, 224341, 224410, 224417, 224646, 224655, 224934, 
	224951, 225217, 225233, 225499, 225514], inplace=True)

print("--------- Fix timezones - RIDER 5")

# note: have to do this one first
# 308126 5 2019-10-22 14:55:58 2019-10-22 21:50:58 -1 days +17:00:00
notfound.loc[308126] = get_timezone_entry(5, 308126)

# 309386 5 2019-11-01 01:20:32 2019-10-26 23:55:46 -1 days +23:00:00
notfound.loc[309386] = get_timezone_entry(5, 309386)

df_tz.loc[(5,6), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(5,7), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(5,6), 'local_timestamp_final_diff_1'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[308126, 309386]])
df_tz = substitute_rows(df_tz, [(5,6), (5,7)], notfound, [308126, 309386])
notfound.drop([308126, 309386], inplace=True)

# 5 2019-10-17 07:16:15 2019-10-17 06:11:15 0 days 01:00:00
notfound.loc[306605] = get_timezone_entry(5, 306605)

# 5 2019-10-17 08:01:14 2019-10-17 08:56:13 -1 days +23:00:01
notfound.loc[306626] = get_timezone_entry(5, 306626)

df_tz.loc[(5,5), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(5,5), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(5,6), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(5,6), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(5,5), 'local_timestamp_final_diff_1'] = np.nan
df_tz.loc[(5,5), 'local_timestamp_final_diff_2'] = np.nan
print("REMOVE duplicate\n", notfound.loc[[306605, 306626]])

print("INSERT\n", notfound.loc[[306605, 306626]])
df_tz = insert_rows(df_tz, (5,6), notfound, [306605, 306626])
notfound.drop([306605, 306626], inplace=True)

print("--------- Fix timezones - RIDER 6")

# Note: do this one first, otherwise difficult
# 319713	 6 2018-12-09 11:09:18 2018-12-09 03:04:18   0 days 08:00:00
notfound.loc[319713] = get_timezone_entry(6, 319713)

print_row("FILL\n", notfound.loc[319713])
df_tz.loc[(6,0), 'local_timestamp_final_max_0'] = notfound.loc[319713, 't_max'] #'2018-12-09 11:09:18'
df_tz.loc[(6,1), 'local_timestamp_final_min_0'] = notfound.loc[319713, 't_min'] #'2018-12-09 03:04:18'
df_tz.loc[(6,0), 'local_timestamp_final_diff_0'] = notfound.loc[319713, 'tz_diff']#.round('h') #'0 days 08:00:00'
notfound.drop(319713, inplace=True)

# The receiver is 3 min behind, therefore adjust
df_tz.loc[(6,0), 'timezone'] = pd.to_timedelta('0 days 00:57:00')

# 317641     6 2018-12-01 00:04:38 2018-12-01 17:26:39 -1 days +00:03:00
# 317729     6 2018-12-02 09:06:37 2018-12-01 08:14:39   0 days 23:56:58
# 317751     6 2018-12-01 16:34:39 2018-12-02 15:46:36 -1 days +00:03:04
# 317937     6 2018-12-03 10:11:34 2018-12-02 07:59:37   0 days 23:56:57
# 317939     6 2018-12-02 12:44:36 2018-12-03 10:16:34 -1 days +00:03:02
# 318128     6 2018-12-04 10:16:30 2018-12-03 08:04:34   0 days 23:56:56
# 318150     6 2018-12-03 12:59:35 2018-12-04 12:51:29 -1 days +00:03:06
# 318167     6 2018-12-04 14:21:29 2018-12-03 14:19:33   0 days 23:56:56
# 318205     6 2018-12-03 17:34:34 2018-12-04 17:26:29 -1 days +00:03:05
# 318363     6 2018-12-05 06:36:32 2018-12-04 06:39:32   0 days 23:56:55
# 318364     6 2018-12-04 06:44:32 2018-12-05 06:36:32 -1 days +00:03:05
# 318400     6 2018-12-05 09:41:27 2018-12-04 09:39:30   0 days 23:56:57
# 318438     6 2018-12-04 12:54:30 2018-12-05 12:46:27 -1 days +00:03:03
# 318443     6 2018-12-05 13:16:26 2018-12-04 13:14:31   0 days 23:56:55
# 318464     6 2018-12-04 15:04:31 2018-12-05 14:56:26 -1 days +00:03:05
# 318626     6 2018-12-06 04:31:23 2018-12-05 04:29:28   0 days 23:56:55
# 318636     6 2018-12-05 05:24:29 2018-12-06 05:16:24 -1 days +00:03:05
# 318680     6 2018-12-06 11:01:37 2018-12-05 11:04:29   0 days 23:56:55
# 318681     6 2018-12-05 11:09:28 2018-12-06 11:01:37 -1 days +00:03:04
# 318706     6 2018-12-06 13:11:24 2018-12-05 13:09:28   0 days 23:56:56
# 318734     6 2018-12-05 15:34:28 2018-12-06 15:26:22 -1 days +00:03:06
# 318750     6 2018-12-06 16:51:23 2018-12-05 16:49:28   0 days 23:56:55
# 318761     6 2018-12-05 17:49:28 2018-12-06 17:41:22 -1 days +00:03:06
# 318916     6 2018-12-07 06:41:21 2018-12-06 06:39:26   0 days 23:56:55
# 318943     6 2018-12-06 08:59:27 2018-12-07 08:51:20 -1 days +00:03:07
# 318961     6 2018-12-07 10:26:20 2018-12-06 10:24:25   0 days 23:56:55
# 318972     6 2018-12-06 11:24:25 2018-12-07 11:16:20 -1 days +00:03:05
# 318977     6 2018-12-07 11:46:21 2018-12-06 11:44:26   0 days 23:56:55
# 319009     6 2018-12-06 14:29:24 2018-12-07 14:21:19 -1 days +00:03:05
# 319015     6 2018-12-07 14:56:19 2018-12-06 14:54:25   0 days 23:56:54
# 319025     6 2018-12-06 15:49:24 2018-12-07 15:41:19 -1 days +00:03:05
# 319030     6 2018-12-07 16:11:19 2018-12-06 16:09:24   0 days 23:56:55
# 319048     6 2018-12-06 17:44:25 2018-12-07 17:36:18 -1 days +00:03:07
# 319064     6 2018-12-07 19:01:19 2018-12-06 18:59:25   0 days 23:56:54
# 319091     6 2018-12-06 21:19:24 2018-12-07 21:11:18 -1 days +00:03:06
# 319233     6 2018-12-08 09:06:17 2018-12-07 09:04:24   0 days 23:56:53
# 319259     6 2018-12-07 11:19:23 2018-12-08 11:11:16 -1 days +00:03:07
# 319276     6 2018-12-08 12:41:16 2018-12-07 12:39:23   0 days 23:56:53
# 319292     6 2018-12-07 14:04:24 2018-12-08 13:56:16 -1 days +00:03:08
# 319342     6 2018-12-08 18:11:16 2018-12-07 18:09:21   0 days 23:56:55
# 319364     6 2018-12-07 20:04:22 2018-12-08 19:56:15 -1 days +00:03:07
# 319381     6 2018-12-08 21:26:15 2018-12-07 21:24:21   0 days 23:56:54
# 319435     6 2018-12-08 03:59:21 2018-12-09 01:51:14 -1 days +02:03:07
# 319476     6 2018-12-09 05:21:14 2018-12-08 07:19:21   0 days 21:56:53
# 319493     6 2018-12-08 08:49:21 2018-12-09 06:41:14 -1 days +02:03:07
# 319498     6 2018-12-09 07:11:14 2018-12-08 09:09:20   0 days 21:56:54
# 319652     6 2018-12-08 22:04:19 2018-12-09 19:56:12 -1 days +02:03:07

for n in np.arange(1,16):
	df_tz.loc[(6,0), 'local_timestamp_final_max_%s'%n] = np.nan
	df_tz.loc[(6,0), 'local_timestamp_final_diff_%s'%n] = np.nan
	df_tz.loc[(6,1), 'local_timestamp_final_min_%s'%n] = np.nan

for i in [319009, 319025, 319048, 319091, 319259, 319276, 319292, 319342,\
		  319364, 319381, 319435, 319476, 319493, 319498, 319652]:
	notfound.loc[i] = get_timezone_entry(6, i)
notfound = notfound.sort_index()

# Note: I think the data of athlete 6 already starts with the recorder that is 1 day behind, 
# therefore first change that before insertion
print("INSERT\n", notfound.loc[[317641, 317729, 317751, 317937, 317939, 318128, 318150, 318167,
								318205, 318363, 318364, 318400, 318438, 318443, 318464, 318626,
								318636, 318680, 318681, 318706, 318734, 318750, 318761, 318916,
								318943, 318961, 318972, 318977, 319009, 319015, 319025, 319030,
								319048, 319064, 319091, 319233, 319259, 319276, 319292, 319342,
								319364, 319381, 319435, 319476, 319493, 319498, 319652]])
df_tz.loc[(6,0), 'timezone'] += pd.to_timedelta('1 days')
df_tz = insert_rows(df_tz, (6,0), notfound, 
	[317641, 317729, 317751, 317937, 317939, 318128, 318150, 318167,
	318205, 318363, 318364, 318400, 318438, 318443, 318464, 318626,
	318636, 318680, 318681, 318706, 318734, 318750, 318761, 318916,
	318943, 318961, 318972, 318977, 319009, 319015, 319025, 319030,
	319048, 319064, 319091, 319233, 319259, 319276, 319292, 319342,
	319364, 319381, 319435, 319476, 319493, 319498, 319652])
notfound.drop([317641, 317729, 317751, 317937, 317939, 318128, 318150, 318167,
			   318205, 318363, 318364, 318400, 318438, 318443, 318464, 318626,
			   318636, 318680, 318681, 318706, 318734, 318750, 318761, 318916,
			   318943, 318961, 318972, 318977, 319009, 319015, 319025, 319030,
			   319048, 319064, 319091, 319233, 319259, 319276, 319292, 319342,
			   319364, 319381, 319435, 319476, 319493, 319498, 319652], inplace=True)

# 351241 6 2019-04-06 12:07:29 2019-04-06 11:02:48 0 days 00:59:41
print_row("FILL\n", notfound.loc[351241])
df_tz.loc[(6,6), 'local_timestamp_final_max_0'] = notfound.loc[351241, 't_max'] #'2019-04-06 11:02:48'
df_tz.loc[(6,7), 'local_timestamp_final_min_0'] = notfound.loc[351241, 't_min'] #'2019-04-06 12:07:29'
df_tz.loc[(6,6), 'local_timestamp_final_diff_0'] = notfound.loc[351241, 'tz_diff']#.round('h') #'0 days 01:00:00'
notfound.drop(351241, inplace=True)

# 356817 6 2019-05-31 05:51:51 2019-05-30 23:43:43 0 days 06:00:00
# Found more than once:  356668 6 2019-05-30 07:58:45 2019-05-30 06:54:46 0 days 00:58:59
df_tz.loc[(6,9), 'local_timestamp_final_max_0'] = df_tz.loc[(6,9), 'local_timestamp_final_max_1']
df_tz.loc[(6,9), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(6,10), 'local_timestamp_final_min_0'] = df_tz.loc[(6,10), 'local_timestamp_final_min_1']
df_tz.loc[(6,10), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(6,9), 'local_timestamp_final_diff_0'] = df_tz.loc[(6,9), 'local_timestamp_final_diff_1']
df_tz.loc[(6,9), 'local_timestamp_final_diff_1'] = np.nan
print_row("REMOVE duplicate\n", found.loc[356668])

# 378143 6 2019-10-03 18:24:29 2019-10-03 10:19:58 0 days 07:59:31
print_row("FILL\n", notfound.loc[378143])
df_tz.loc[(6,11), 'local_timestamp_final_max_0'] = notfound.loc[378143, 't_max'] #'2019-10-03 10:19:58'
df_tz.loc[(6,12), 'local_timestamp_final_min_0'] = notfound.loc[378143, 't_min'] #'2019-10-03 18:24:29'
df_tz.loc[(6,11), 'local_timestamp_final_diff_0'] = notfound.loc[378143, 'tz_diff']#.round('h') #'0 days 08:00:00'
notfound.drop(378143, inplace=True)


print("--------- Fix timezones - RIDER 10")
# 405474 10 2019-01-31 07:30:41 2019-01-27 09:24:38 -1 days +23:01:10
print_row("FILL\n", notfound.loc[405474])
df_tz.loc[(10,2), 'local_timestamp_final_max_0'] = notfound.loc[405474, 't_max'] #'2019-01-31 07:30:41'
df_tz.loc[(10,3), 'local_timestamp_final_min_0'] = notfound.loc[405474, 't_min'] #'2019-01-27 09:24:38'
df_tz.loc[(10,2), 'local_timestamp_final_diff_0'] = notfound.loc[405474, 'tz_diff']#.round('h') #'-1 days +23:01:10'
notfound.drop(405474, inplace=True)

# COMBINE travels
# total time diff should be four hours so it seems the travel is split up
# 409993 10 2019-02-20 15:50:03 2019-02-20 14:45:03 0 days 01:00:00 (1h)
notfound.loc[409993] = get_timezone_entry(10, 409993)

# 410248 10 2019-02-21 15:50:01 2019-02-21 13:45:01 0 days 02:00:00 (2h)
notfound.loc[410248] = get_timezone_entry(10, 410248)

# 410261 10 2019-02-21 20:05:01 2019-02-21 18:20:02 0 days 01:00:00 (1h)
notfound.loc[410261] = get_timezone_entry(10, 410261)

df_tz.loc[(10,3), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,4), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,3), 'local_timestamp_final_diff_1'] = np.nan

df_tz.loc[(10,3), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(10,4), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(10,3), 'local_timestamp_final_diff_2'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[409993, 410248, 410261]])
df_tz = substitute_rows(df_tz, [(10,3), (10,4)], notfound, [409993, 410248, 410261])
notfound.drop([409993, 410248, 410261], inplace=True)

# COMBINE travels
# 412469 10 2019-03-03 13:29:45 2019-03-03 16:24:44 -1 days +21:00:00
notfound.loc[412469] = get_timezone_entry(10, 412469)

# 412518 10 2019-03-06 06:19:38 2019-03-03 17:49:45 -1 days +23:00:00
notfound.loc[412518] = get_timezone_entry(10, 412518)

df_tz.loc[(10,4), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,5), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,4), 'local_timestamp_final_diff_1'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[412469, 412518]])
df_tz = substitute_rows(df_tz, [(10,4), (10,5)], notfound, [412469, 412518])
notfound.drop([412469, 412518], inplace=True)

# Found more than once:  419124 10 2019-04-07 16:28:39 2019-04-07 17:23:39 -1 days +23:00:00
print_row("REMOVE duplicate\n", found.loc[419124])
df_tz.loc[(10,8), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,9), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,8), 'local_timestamp_final_diff_1'] = np.nan

# Found more than once:  421661 10 2019-04-18 09:18:19 2019-04-18 10:13:19 -1 days +23:00:00
print_row("REMOVE duplicate\n", found.loc[421661])
df_tz.loc[(10,10), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,11), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,10), 'local_timestamp_final_diff_1'] = np.nan

# Found more than once:  426626 10 2019-05-22 10:05:03 2019-05-22 09:00:03 0 days 01:00:00
print_row("REMOVE duplicate\n", found.loc[426626])
df_tz.loc[(10,13), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,14), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,13), 'local_timestamp_final_diff_1'] = np.nan

# Found more than once:  426660 10 2019-05-22 13:55:02 2019-05-22 12:50:03 0 days 00:59:59
print_row("REMOVE duplicate\n", found.loc[426660])
df_tz.loc[(10,13), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(10,14), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(10,13), 'local_timestamp_final_diff_2'] = np.nan

df_tz.loc[(10,14), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,15), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,14), 'local_timestamp_final_diff_1'] = np.nan

# COMBINE
# 426626 10 2019-05-22 10:05:03 2019-05-22 09:00:03 (+1)
notfound.loc[426626] = get_timezone_entry(10, 426626)

# 426660 10 2019-05-22 13:55:02 2019-05-22 12:50:03 (+1)
notfound.loc[426660] = get_timezone_entry(10, 426660)

print("SUBSTITUTE\n", notfound.loc[[426626, 426660]])
df_tz = substitute_rows(df_tz, [(10,14), (10,15)], notfound, [426626, 426660])
notfound.drop([426626, 426660], inplace=True)

# COMBINE
# 427680 10 2019-05-26 09:04:57 2019-05-26 09:59:57 (-1)
notfound.loc[427680] = get_timezone_entry(10, 427680)

# 427706 10 2019-05-26 10:14:57 2019-05-26 11:09:57 (-1)
notfound.loc[427706] = get_timezone_entry(10, 427706)

df_tz.loc[(10,15), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,16), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,15), 'local_timestamp_final_diff_1'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[427680, 427706]])
df_tz = substitute_rows(df_tz, [(10,15), (10,16)], notfound, [427680, 427706])
notfound.drop([427680, 427706], inplace=True)

# 428012 10 2019-06-01 00:04:49 2019-05-27 13:29:55 0 days 01:00:02
print_row("FILL\n", notfound.loc[428012])
df_tz.loc[(10,16), 'local_timestamp_final_max_0'] = notfound.loc[428012, 't_max'] #'2019-06-01 00:04:49'
df_tz.loc[(10,17), 'local_timestamp_final_min_0'] = notfound.loc[428012, 't_min'] #'2019-05-27 13:29:55'
df_tz.loc[(10,16), 'local_timestamp_final_diff_0'] = notfound.loc[428012, 'tz_diff']#.round('h') #'0 days 01:00:02'
notfound.drop(428012, inplace=True)

# Found more than once:  443386 10 2019-09-08 21:57:55 2019-09-08 22:52:55 -1 days +23:00:00
print_row("REMOVE duplicate\n", found.loc[443386])
df_tz.loc[(10,24), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,25), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,24), 'local_timestamp_final_diff_1'] = np.nan

# 443980 10 2019-09-17 23:42:27 2019-09-11 09:57:46 0 days 01:00:00
print_row("FILL\n", notfound.loc[443980])
df_tz.loc[(10,26), 'local_timestamp_final_max_0'] = notfound.loc[443980, 't_max'] #'2019-09-17 23:42:27'
df_tz.loc[(10,27), 'local_timestamp_final_min_0'] = notfound.loc[443980, 't_min'] #'2019-09-11 09:57:46'
df_tz.loc[(10,26), 'local_timestamp_final_diff_0'] = notfound.loc[443980, 'tz_diff']#.round('h') #'0 days 01:00:00'
notfound.drop(443980, inplace=True)

# COMBINE
# 445183 10 2019-10-06 11:36:28 2019-10-06 10:31:29 0 days 01:04:59
notfound.loc[445183] = get_timezone_entry(10, 445183)

# 445342 10 2019-10-07 06:51:26 2019-10-07 00:46:26 0 days 06:05:00
notfound.loc[445342] = get_timezone_entry(10, 445342)

df_tz.loc[(10,28), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(10,29), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(10,28), 'local_timestamp_final_diff_1'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[445183, 445342]])
df_tz = substitute_rows(df_tz, [(10,28), (10,29)], notfound, [445183, 445342])
notfound.drop([445183, 445342], inplace=True)

# 449228 10 2019-11-04 10:04:56 2019-10-24 02:00:33 -1 days +22:59:57
# 449404 10 2019-11-04 19:44:56 2019-11-05 00:39:55 -1 days +19:00:01
# 450245 10 2019-11-08 01:59:44 2019-11-08 00:24:44 0 days 01:30:00
# 450275 10 2019-11-08 04:19:44 2019-11-08 04:29:44 -1 days +22:30:00
print("SUBSTITUTE\n", notfound.loc[[449228, 449404, 450245, 450275]])
df_tz = substitute_rows(df_tz, [(10,30), (10,31)], notfound, [449228, 449404, 450245, 450275])
notfound.drop([449228, 449404, 450245, 450275], inplace=True)


print("--------- Fix timezones - RIDER 12")

# Note: it seems he changes his timezone a month later than when he travels, 
# which leads me to the question whether the date on his device is correct at all
# For now, I think we should drop all data of this athlete

"""
# TODO: weird!
# NOTICE: I think he might have the month setup wrong as 
# He participated in the Japan cup in October 2018, so most likely the time was reset then
# Time in Japan is +9h, so most likely everything until then was at +9
# He made a change of -7h so did he change it to +2 then? 
# This would be slightly weird as the time in spain then should be +1h in winter
454571 12 2018-12-17 21:19:47 2018-12-18 04:14:46 -1 days +17:00:01
df_tz.loc[(12,0), 'local_timestamp_max_0'] = notfound.loc[454571, 't_max']
df_tz.loc[(12,1), 'local_timestamp_min_0'] = notfound.loc[454571, 't_min']
df_tz.loc[(12,0), 'local_timestamp_diff_0'] = notfound.loc[454571, 'tz_diff']
df_tz.loc[(12,0), 'timezone'] = pd.to_datetime('0 days 09:00:00')

# Japan cup again?? But why would he change the dates when he is already back?
476476 12 2019-11-06 15:47:19 2019-11-06 22:42:19 -1 days +17:00:00
479784 12 2019-11-19 14:26:49 2019-11-19 01:46:50 0 days 07:00:00

#df_tz.drop((12,8))
"""
# For now just drop lozano
drop_idx = df_tz[df_tz.index.get_level_values(0) == 12].index #12
df_tz.drop(drop_idx, inplace=True)


print("--------- Fix timezones - RIDER 13")

# Combine
# 527859 13 2019-05-20 07:53:36 2019-05-19 07:48:50 -1 days +09:00:43
notfound.loc[527859] = get_timezone_entry(13, 527859)

# 527860 13 2019-05-19 07:48:50 2019-05-19 22:43:07 0 days 23:59:46
notfound.loc[527860] = get_timezone_entry(13, 527860)

df_tz.loc[(13,4), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(13,5), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(13,4), 'local_timestamp_final_diff_1'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[527859, 527860]])
df_tz = substitute_rows(df_tz, [(13,4), (13,5)], notfound, [527859, 527860])
notfound.drop([527859, 527860], inplace=True)

# Found more than once:  528560 13 2019-05-22 19:13:07 2019-05-22 18:08:24 0 days 00:59:43
print_row("REMOVE duplicate\n", found.loc[528560])
df_tz.loc[(13,4), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(13,5), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(13,4), 'local_timestamp_final_diff_2'] = np.nan

print("--------- Fix timezones - RIDER 14")

# 598682 14 2019-08-03 13:42:21 2019-07-12 08:11:54 0 days 01:06:42
print_row("FILL\n", notfound.loc[598682])
df_tz.loc[(14,9), 'local_timestamp_final_max_0'] = notfound.loc[598682, 't_max'] #'2019-08-03 13:42:21'
df_tz.loc[(14,10), 'local_timestamp_final_min_0'] = notfound.loc[598682, 't_min'] #'2019-07-12 08:11:54'
df_tz.loc[(14,9), 'local_timestamp_final_diff_0'] = notfound.loc[598682, 'tz_diff']#.round('h') #'0 days 01:06:42'
notfound.drop(598682, inplace=True)


print("--------- Fix timezones - RIDER 15")

# COMBINE
# 15 2019-10-23 12:55:05 2019-10-23 19:51:13 -1 days +16:58:52
notfound.loc[626935] = get_timezone_entry(15, 626935)

# 628373 15 2019-10-28 19:44:16 2019-10-28 17:39:45 0 days 01:59:31
notfound.loc[628373] = get_timezone_entry(15, 628373)

# 629069 15 2019-10-30 09:03:05 2019-10-31 08:59:07 -2 days +23:58:58
notfound.loc[629069] = get_timezone_entry(15, 629069)

# 629580 15 2019-11-02 12:01:02 2019-11-02 14:52:53 -1 days +21:05:09
notfound.loc[629580] = get_timezone_entry(15, 629580)

df_tz.loc[(15,9), 'local_timestamp_final_max_1'] = np.nan
df_tz.loc[(15,10), 'local_timestamp_final_min_1'] = np.nan
df_tz.loc[(15,9), 'local_timestamp_final_diff_1'] = np.nan

df_tz.loc[(15,9), 'local_timestamp_final_max_2'] = np.nan
df_tz.loc[(15,10), 'local_timestamp_final_min_2'] = np.nan
df_tz.loc[(15,9), 'local_timestamp_final_diff_2'] = np.nan

df_tz.loc[(15,9), 'local_timestamp_final_max_3'] = np.nan
df_tz.loc[(15,10), 'local_timestamp_final_min_3'] = np.nan
df_tz.loc[(15,9), 'local_timestamp_final_diff_3'] = np.nan

print("SUBSTITUTE\n", notfound.loc[[626935, 628373, 629069, 629580]])
df_tz = substitute_rows(df_tz, [(15,9), (15,10)], notfound, [626935, 628373, 629069, 629580])
notfound.drop([626935, 628373, 629069, 629580], inplace=True)

# -------- Restructuring
# Reset counter n
df_tz.reset_index(drop=False, inplace=True)
df_tz.rename(columns={'n':'n_prev'}, inplace=True)
df_tz['n'] = df_tz.groupby('RIDER').cumcount()
df_tz = df_tz.set_index(['RIDER', 'n'])

# Fill first and last date
max_n = df_tz.reset_index().groupby('RIDER')['n'].max()
for i in max_n.index:
	# first date of season
	df_tz.loc[(i,0), 'local_timestamp_final_min_0'] = '2018-12-01 00:00:00'

	# last date of season
	df_tz.loc[(i,max_n[i]), 'local_timestamp_final_max_0'] = '2019-11-30 23:59:59'

# -------- Censoring
# Here we assume that we have all the changes in timezone that were in the glucose files in the file
# censoring at 2,9
df_tz.loc[(2,8), 'local_timestamp_final_max_0'] = get_maxts_before_censoring(2,8)
df_tz.drop((2,9), inplace=True)

# censoring at 3,1
df_tz.loc[(3,0), 'local_timestamp_final_max_0'] = get_maxts_before_censoring(3,0)
df_tz.loc[(3,2), 'local_timestamp_final_min_0'] = get_mints_after_censoring(3,2)
df_tz.drop((3,1), inplace=True)
df_tz = merge_rows(df_tz, (3,0), (3,2))

# censoring at 3,4
df_tz.loc[(3,3), 'local_timestamp_final_max_0'] = get_maxts_before_censoring(3,3)
df_tz.loc[(3,5), 'local_timestamp_final_min_0'] = get_mints_after_censoring(3,5)
df_tz.drop((3,4), inplace=True)

# censoring at 4,10
df_tz.loc[(4,9), 'local_timestamp_final_max_0'] = get_maxts_before_censoring(4,9)
df_tz.loc[(4,11), 'local_timestamp_final_min_0'] = get_mints_after_censoring(4,11)
df_tz.drop((4,10), inplace=True)

# censoring at 10,0
df_tz.loc[(10,1), 'local_timestamp_final_min_0'] = get_mints_after_censoring(10,1)
df_tz.drop((10,0), inplace=True)

# censoring 12,3-10
#df_tz.loc[(12,2), 'local_timestamp_final_max_0'] = get_maxts_before_censoring(12,2)
#df_tz.loc[(12,11), 'local_timestamp_final_min_0'] = get_mints_after_censoring(12,11)

# censoring 15,1-6
df_tz.loc[(15,0), 'local_timestamp_final_max_0'] = get_maxts_before_censoring(15,0)
df_tz.loc[(15,7), 'local_timestamp_final_min_0'] = get_mints_after_censoring(15,7)
df_tz.drop([(15,n) for n in range(1,7)], inplace=True)

# -------- Timezone change not found
# --- athlete 6
# 323067      6 2018-12-21 23:36:24      0 days 12:11:36
# 348603      6 2019-03-27 09:58:15      0 days 04:33:59
# 363876      6 2019-08-12 11:31:09     45 days 20:05:25
# 375875      6 2019-09-25 09:50:25      0 days 12:51:09

# 6,49
df_tz.drop((6,49), inplace=True)
df_tz = merge_rows(df_tz, (6,48), (6,50))

# 6,57
# only option remains is that it was reset during the transmitter change??
# between 2019-05-31 05:51:51 and 2019-10-03 10:19:58
# this could have happened on 2019-08-12 and on 2019-09-25
# of these options 2019-08-12 is the most likely
# as this was closest to the time they actually travelled
# it also was a long gap

# 363876 2019-08-12 11:31:09 2019-06-27 15:25:44
# censoring of 45 days
df_tz.loc[(6,57), 'local_timestamp_final_max_0'] = '2019-06-27 15:25:44'
df_tz.loc[(6,58), 'local_timestamp_final_min_0'] = '2019-08-12 11:31:09'

# --- athlete 13
# 13                            [PL82917592]
# 496844     13 2019-01-19 00:20:39      0 days 02:16:16
# 521859     13 2019-04-19 19:27:14      0 days 03:04:04
# 544785     13 2019-07-23 23:11:12      1 days 01:28:23
# 562323     13 2019-10-14 23:36:11      3 days 00:38:19

df_tz.drop([(13,9), (13,10)], inplace=True)
df_tz = merge_rows(df_tz, (13,8), (13,11))

# TODO CHECK: does it happen that this athlete changes the time
# when he changes transmitters?
# In that case, between 2019-07-23 23:11:12 and 2019-10-14 23:36:11
# we do not know the timezone 
# However, I think this is not the case.

# --- athlete 14
# 14                            [PL82598347]
# 586062     14 2019-01-19 00:48:56      0 days 02:12:41
# 595437     14 2019-05-31 11:30:24     96 days 14:38:43
# 600906     14 2019-08-12 00:07:55      0 days 14:56:16
# 605433     14 2019-11-08 02:05:04      0 days 15:32:40

df_tz.drop([(14,n) for n in range(1,9)], inplace=True)
df_tz = merge_rows(df_tz, (14,0), (14,9), keep_tz='last')
df_tz.drop([(14,n) for n in range(11,19)], inplace=True)
df_tz = merge_rows(df_tz, (14,10), (14,19))

# note now we are assuming that he had it at +2 and changed it to +3
# because of switch from winter to summer time, but we should check this!!

# TODO CHECK: if Henttala has right timezone
# TODO CHECK: if Henttala ever changes timezone during change of transmitter
# convert any remaining strings to datetime




df_tz['local_timestamp_final_min'] = pd.to_datetime(df_tz['local_timestamp_final_min_0'])
df_tz['local_timestamp_final_max'] = pd.to_datetime(df_tz['local_timestamp_final_max_0'])
df_tz['local_timestamp_final_diff'] = pd.to_timedelta(df_tz['local_timestamp_final_diff_0'])

df_tz.drop(['local_timestamp_final_%s_%s'%(j,i) for i in range(df_min.shape[1]) for j in ('min', 'max', 'diff')], axis=1, inplace=True)


df_tz['timezone_error'] = df_tz['timezone_final'].round('h') != df_tz['timezone'].round('h')
print("Number of times the timezones differ: ", df_tz['timezone_error'].sum())

# final columns
df_tz.drop(['n_prev', 'date_min', 'date_max', 'glucose', 'local_timestamp_final_diff', 'timezone_error'], axis=1, inplace=True)
df_tz.drop(['timezone', 'timestamp_min', 'timestamp_max', 'local_timestamp_min', 'local_timestamp_max'], axis=1, inplace=True)
df_tz.rename(columns={'local_timestamp_final_min':'local_timestamp_min',
					  'local_timestamp_final_max':'local_timestamp_max',
					  'timezone_final':'timezone'}, inplace=True)
df_tz.to_csv(path+'timezone/timezone_dexcom.csv')