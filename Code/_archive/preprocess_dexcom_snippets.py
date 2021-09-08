


# identify dup (these make the timezone switch a little more hard)
#df_changes['dup'] = df_changes.groupby('RIDER')['timezone'].transform(lambda x: x.diff() < '0s')
df_changes['WARNING'] = df_changes['local_timestamp_max'].shift(1) > df_changes['local_timestamp_min']
df_changes.loc[df_changes.index.get_level_values(1) == 0, 'WARNING'] = False

# calculate gaps and dups
# note that a timezone change does not necessarily have to be a gap or a dup
df.loc[df['Event Type'] == 'EGV', 'local_timestamp_diff'] = df.loc[df['Event Type'] == 'EGV', 'local_timestamp'].diff()
df.loc[df['Event Type'] == 'EGV', 'transmitter_diff'] = df.loc[df['Event Type'] == 'EGV', 'Transmitter Time (Long Integer)'].diff()

df['timediff'] = df['local_timestamp_diff'] - pd.to_timedelta(df['transmitter_diff'], 'sec')
df.loc[df['transmitter_order'].diff() != 0, 'timediff'] = np.nan # correct for transmitter change

df['gap'] = (df['timediff'] > '5min')
print("Number of gaps: ", df['gap'].sum())

df['dup'] = (df['timediff'] < '-5min')
print("Number of dups: ", df['dup'].sum())


n_max = df_changes.groupby('RIDER').count()['local_timestamp_min']
prev_idx_max = 0
for (i,n), (t_min, t_max, tz, _) in df_changes.iterrows():
	# check whether there was a duplicate before or during
	dup_min = df_changes.loc[(i,n), 'WARNING'] # after dup
	dup_max = False # before dup
	if n != n_max[i] - 1:
		dup_max = df_changes.loc[(i,n+1), 'WARNING']

	# min index
	if dup_min: # after dup
		idx_min = df[(df.RIDER == i) & df['dup'] & (df.local_timestamp == t_min)].index[0]
	else:
		# use that they are sorted and thus idx_min > prev_idx_max
		idx_min = df[(df.RIDER == i) & (df.local_timestamp >= t_min)].loc[prev_idx_max:].index[0]

	# max index
	if dup_max: # before dup
		idx_max = df[(df.RIDER == i) & df['dup'].shift(-1).fillna(False) & (df.local_timestamp == t_max)].index[0]
	else:
		idx_max = df[(df.RIDER == i) & (df.local_timestamp <= t_max)].index[-1]

	print(i, n, idx_min, idx_max)

	df.loc[idx_min:idx_max, 'timestamp'] = df.loc[idx_min:idx_max, 'local_timestamp'] - tz

	prev_idx_max = idx_max
	del idx_max, idx_min





# Figure out actual timezone changes
df_tz = pd.read_csv('../TrainingPeaks/2019/timezone/timezone_final.csv', index_col=[0,1])
df_tz.timezone = pd.to_timedelta(df_tz.timezone)
df_tz.local_timestamp_min = pd.to_datetime(df_tz.local_timestamp_min)
df_tz.local_timestamp_max = pd.to_datetime(df_tz.local_timestamp_max)

# recalculate local timestamp
df.rename(columns={'local_timestamp':'local_timestamp_raw'}, inplace=True)


"""
df['timezone'] = (df['local_timestamp_raw'] - df['timestamp']).round('min')
df['timezone_change'] = (df['timezone'].diff() != '0s') & df['timezone'].notna() 
df.loc[df['timezone_change'], df.columns.drop(['Event Subtype', 'Insulin Value (u)', 'Carb Value (grams)', 'Duration (hh:mm:ss)'])].to_csv('timezone/timezone_change_dexcom.csv')
"""

for i in df_egv.RIDER.unique():
	for n in df_tz.loc[i].index[:-1]:
		t_max = df_tz.loc[i].loc[n, 'local_timestamp_max']
		t_min = df_tz.loc[i].loc[n+1, 'local_timestamp_min']
		tz_change = df_tz.loc[i].loc[n+1, 'timezone'] - df_tz.loc[i].loc[n, 'timezone']

		df_inbetween = df_egv[(df_egv.RIDER == i) & (df_egv.local_timestamp > t_max) & (df_egv.local_timestamp < t_min)]

		# any gaps in local_timestamp (regardless of transmitter_diff)
		#print((df_inbetween['local_timestamp_diff'] > '55min').sum())

		# any gaps in transmitter_diff (regardless of local_timestamp)
		#print((df_inbetween['transmitter_diff'] > 55*300).sum())

		plt.figure(figsize=(15,4))
		ax = sns.scatterplot(df_inbetween['local_timestamp'], df_inbetween['Glucose Value (mg/dL)'], 
			hue=(df_inbetween['local_timestamp_diff'] > '55min') | (df_inbetween['transmitter_diff'] < 300))
		ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

		plt.title('Rider %s from %s to %s'%(i,t_max, t_min))
		plt.ylabel('Glucose Value (mg/dL) EGV')
		plt.legend(title=tz_change)
		plt.savefig(path+'timezone/%s_%s_%s.pdf'%(i,t_max,t_min), bbox_inches='tight')
		plt.savefig(path+'timezone/%s_%s_%s.png'%(i,t_max,t_min), dpi=300, bbox_inches='tight')
		plt.close()



# check if there are readings with known timezone, that still contain duplicates
df_itv = df[df.timestamp.notna()].groupby(['RIDER', pd.Grouper(key='timestamp', freq='5min')])\
	.agg({'timestamp'						:['unique', 'nunique'], 
		  'Glucose Value (mg/dL)'			:'nunique', 
		  'Transmitter Time (Long Integer)'	:['nunique', 'unique']})

df_itv[('Transmitter Time (Long Integer)', 'diff')] = df_itv[('Transmitter Time (Long Integer)', 'unique')].apply(lambda x: x[1] - x[0] if len(x) == 2 else ([x[1]-x[0],x[2]-x[1]] if len(x) == 3 else np.nan))
df_itv[('Transmitter Time (Long Integer)', 'negdiff')] = df_itv[('Transmitter Time (Long Integer)', 'diff')].apply(lambda x: x < 0 if isinstance(x, float) else ((np.array(x) < 0).any() if isinstance(x, list) else np.nan))
mask_dupl = ((df_itv[('timestamp', 'nunique')] > 1) | (df_itv[('Glucose Value (mg/dL)', 'nunique')] > 1)) & df_itv[('Transmitter Time (Long Integer)', 'negdiff')]
print("For readings with known timezones, number of duplicate timestamps (in UTC) still out there: ", 
	mask_dupl.sum())

df_dupl = df_itv[mask_dupl].reset_index()[[('RIDER',''), ('timestamp', '')]]
df_dupl.columns = df_dupl.columns.get_level_values(0)
print("Number of dates it applies to: ", 
	df_dupl.groupby('RIDER')['timestamp'].apply(lambda x: np.unique(x.dt.date)))

# MANUAL WORK: for all duplicate warnings, check manually 
# whether  there is actually some travelling involved that we did not see

# identify travel times
# this is only for dup
mask_travel = df.loc[df['Event Type'] == 'EGV', 'local_timestamp'].diff() < datetime.timedelta(0)
df[mask_travel & (df['Event Type'] == 'EGV')]

del df_itv, mask_dupl, df_dupl



df_interval['dup'] = 
# check how many duplicates are with unknown timezone

# if it is the first or last day of the season, check if there are not any duplicates or gaps, and if not, we can just assume they have the same timezone
# group readings in intervals
df_interval = df[df['Event Type'] == 'EGV']\
	.groupby(['RIDER', pd.Grouper(key='local_timestamp', freq='5min')])\
	.agg({'local_timestamp':['unique', 'nunique'],
		  'Glucose Value (mg/dL)':'nunique',
		  'Transmitter Time (Long Integer)':['nunique', 'unique']})
df_interval.rename(columns={'Transmitter Time (Long Integer)':'transmitter', 
							'Glucose Value (mg/dL)':'glucose'}, inplace=True)

# check if there are more than 2 or 3 readings anywhere
print("Number of intervals with more than 2 measurements: ", 
	((df_interval[('local_timestamp', 'nunique')] > 2) |
	 (df_interval[('glucose', 'nunique')] > 2) |
	 (df_interval[('transmitter', 'nunique')] > 2)).sum())
print("Number of intervals with more than 3 measurements: ", 
	((df_interval[('local_timestamp', 'nunique')] > 3) |
	 (df_interval[('glucose', 'nunique')] > 3) |
	 (df_interval[('transmitter', 'nunique')] > 3)).sum())

df_interval.columns = [x[0]+'_'+x[1] for x in df_interval.columns]

# calculate difference in transmitter time between readings in the same interval
df_interval['transmitter_diff'] = df_interval['transmitter_unique'].apply(lambda x: x[1] - x[0] if len(x) == 2 else ([x[1]-x[0],x[2]-x[1]] if len(x) == 3 else np.nan))
df_interval['transmitter_negdiff'] = df_interval['transmitter_diff'].apply(lambda x: x < 0 if isinstance(x, float) else ((np.array(x) < 0).any() if isinstance(x, list) else np.nan))
df_interval.drop('transmitter_unique', axis=1, inplace=True)

# fill up missing intervals
year_interval = pd.DataFrame(np.nan, columns=['nunique'], index=pd.MultiIndex.from_product([df.RIDER.unique(),
					pd.date_range(start=datetime.datetime(2018,12,1,0), end=datetime.datetime(2019,12,1), freq='5min')],
				names=['RIDER', 'local_timestamp']))

df_interval = pd.merge(df_interval, year_interval, how='right', 
	left_index=True, right_index=True, validate='one_to_one')
df_interval.drop('nunique', axis=1, inplace=True)

# print how often intervals have more than 2 readings
print("local_timestamp nunique > 2: ", (df_interval['local_timestamp_nunique'] > 1).sum())
print("glucose nunique > 2: ", (df_interval['glucose_nunique'] > 1).sum())
print("transmitter nunique > 2: ", (df_interval['transmitter_nunique'] > 1).sum())
print("transmitter diff < 0: ", (df_interval['transmitter_negdiff'].sum()))



# Duplicates because of travelling: either 2 or more local_timestamps from readings in one interval, 
# or (when the two timestamps are the same) there are 2 or more different glucose values in one interval
# AND the difference between two transmitter times is negative (PROVIDED the correct sorting by transmitter time above)

# NOTE: we don't need to include the first two statements, only filtering by negative transmitter time gives the same results
# but we did this initially because negative transmitter time can also be due to changing the transmitter (instead of a timezone change)

# NOTE that nunique transmitter > 2 does not work as there can also be other types of readings
 
df_interval['dup'] = ((df_interval['local_timestamp_nunique'] > 1) | (df_interval['glucose_nunique'] > 1)) & df_interval['transmitter_negdiff']
df_interval['gap'] = df_interval['local_timestamp_nunique'].isna()
df_interval.reset_index(inplace=True)

df_tz['local_timestamp_change'] = np.nan
df_tz = df_tz.astype({'local_timestamp_change':object})
n_per_rider = df_tz.reset_index().groupby('RIDER')['n'].max()

# identify travel times from glucose
for i, n in df_tz.index:
	print("RIDER %s - n = %s"%(i,n))

	# identify datetime range with unknown timezone
	if n != n_per_rider[i]: # if it is not the last n for this rider
		tz_unknown = df_tz.loc[(i,n), 'local_timestamp_max'], df_tz.loc[(i,n+1), 'local_timestamp_min']
	else:
		tz_unknown = df_tz.loc[(i,n), 'local_timestamp_max'], '2019-11-30 23:59:59'

	# gap or dup
	if n == n_per_rider[i]:
		change = 'unk'
	elif df_tz.timezone.diff()[(i,n+1)] > datetime.timedelta(0):
		change = 'gap'
	else:
		change = 'dup'

	# check if there is a duplicate or gap in the dexcom glucose file within tz_uknown
	df_unknown = df_interval.loc[(df_interval.RIDER == i)\
		& (df_interval.local_timestamp > tz_unknown[0])
		& (df_interval.local_timestamp < tz_unknown[1])]

	if change == 'gap':
		if df_unknown.dup.sum() != 0:
			print("unnecessary dups: %s"%(df_unknown.dup.sum()))

		# set df_tz with all possible changepoints
		df_tz.at[(i,n), 'local_timestamp_change'] = df_unknown[df_unknown[change].diff().shift(-1) == True]\
			.dropna(subset=['local_timestamp_unique'])['local_timestamp_unique'].apply(lambda x: x[0]).tolist()
		print("GAPS: ", df_tz.loc[(i,n), 'local_timestamp_change'])
	
	elif change == 'dup':
		if df_unknown.gap.sum() != 0:
			print("unnecessary gaps: %s"%(df_unknown.gap.sum()))

		if not df_unknown.local_timestamp_unique.dropna().empty # no glucose values recorded
			and df_unknown[df_unknown[change]].empty # no duplicates
		# set df_tz with all possible changepoints
		df_tz.at[(i,n), 'local_timestamp_change'] = df_unknown[df_unknown[change].diff().shift(-1) == True]\
			.dropna(subset=['local_timestamp_unique'])['local_timestamp_unique'].apply(lambda x: x[0]).tolist()#
		print("DUPS: ", df_tz.loc[(i,n), 'local_timestamp_change'])

	elif change == 'unk':
		df_tz.at[(i,n), 'local_timestamp_change'] = df_unknown[(df_unknown[['gap', 'dup']].diff().shift(-1) == True).any(axis=1)]\
			.dropna(subset=['local_timestamp_unique'])['local_timestamp_unique'].apply(lambda x: x[0]).tolist()#
		print("DUPS or GAPS: ", df_tz.loc[(i,n), 'local_timestamp_change'])

print("Number of timezone changes left to check manually: ", df_tz.local_timestamp_change.apply(lambda x: len(x) == 0).sum())
df_tz.to_csv('../TrainingPeaks/2019/timezone_final_change.csv')
df_interval.to_csv('../Dexcom/interval.csv')

# NOW there is unfortunately some manual cleaning involved

# For each timezone that we have identified, select the best one


# 1 [2019-03-13T18:58:11.000000000, 2019-03-13T18:58:12.000000000]

for i in df.RIDER.unique():
	df[df.RIDER == i]

"""
# number of readings per interval
itv = pd.interval_range(start=datetime.datetime(2018,12,1,0), end=datetime.datetime(2019,12,1), freq='5min')
readings_itv = {}
for i in df.RIDER.unique():
	readings = np.zeros(len(itv))
	for t in df.loc[(df.RIDER == i) & (df['Event Type'] == 'EGV'), 'local_timestamp']:
		readings += itv.contains(t)
	readings_itv[i] = readings
readings_itv = pd.DataFrame.from_dict(readings_itv).unstack().to_frame().set_index(pd.MultiIndex.from_product([df.RIDER.unique(), itv]))
"""


idx_change = df_tz.loc[df_tz['timezone_change_combine'], ['RIDER', 'date']]


df['date'] = df['local_timestamp'].dt.date

# TODO: select EGV
# plot glucose for days with timezone change
matplotlib.use('TkAgg')
for _, (i, d) in idx_change.iterrows():
	df_i = df[(df.RIDER == i) & (df.date == d)]
	ax = sns.scatterplot(df_i['local_timestamp'], df_i['Glucose Value (mg/dL)'], 
		hue=(df_i.local_timestamp.diff() < '4min') | (df_i.local_timestamp.diff() > '6min'))
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(i,d))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'tz_change/glucose_%s_%s.pdf'%(i,d), bbox_inches='tight')
	plt.savefig(path+'tz_change/glucose_%s_%s.png'%(i,d), dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# plot glucose for all days
df.loc[df.local_timestamp.diff() < '4min', 'change'] = '<4min'
df.loc[df.local_timestamp.diff() > '6min', 'change'] = '>6min'
df.loc[df.local_timestamp.diff() > '11min', 'change'] = '>11min'
df.loc[df.local_timestamp.duplicated(keep=False), 'change'] = 'dupl'

# closest tz change
df.join(df_tz[['RIDER', 'date', 'timezone_diff_combine']], on=['RIDER', 'date'])

for _, (i, d) in df[['RIDER', 'date']].iterrows():
	df_i = df[(df.RIDER == i) & (df.date == d)]
	ax = sns.scatterplot(df_i['local_timestamp'], df_i['Glucose Value (mg/dL)'], hue=df_i['change'])
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(i,d))
	plt.legend(title='N = %s'%len(df_i))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'all/glucose_%s_%s.pdf'%(i,d), bbox_inches='tight')
	plt.savefig(path+'all/glucose_%s_%s.png'%(i,d), dpi=300, bbox_inches='tight')
	plt.close()


# try to identify point of change

# also print measurements per day




""" TODO: don't drop duplicate timestamps here until issue with max ts is solved
df.drop_duplicates(subset=['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype'], keep='last', inplace=True)
print("DROPPED %s duplicate timestamps (per rider, event type and event subtype)"%df_dupl_ts.duplicated(subset=cols_dupl_ts).sum())
"""
matplotlib.use('TkAgg')
# plot duplicates
for i, (r, d) in enumerate(count_dupl_ts.index):
	df_dupl_ts_i = df[(df.RIDER == r) & (df.date == d) & (df['Event Type'] == 'EGV')]

	ax = sns.scatterplot(df_dupl_ts_i['local_timestamp'], df_dupl_ts_i['Glucose Value (mg/dL)'], 
		hue=df_dupl_ts_i[cols_dupl_ts].duplicated(keep=False))
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(r,d))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'dupl/glucose_%s_%s.pdf'%(r,d), bbox_inches='tight') # TODO: CHANGE DROP
	plt.savefig(path+'dupl/glucose_%s_%s.png'%(r,d), dpi=300, bbox_inches='tight') # TODO: CHANGE DROP
	plt.show()
	plt.close()
# note sometimes no hue because duplicated are not in EGV

df.reset_index(drop=True, inplace=True)


# measurements per day per athlete
count_readings = df[df['Event Type'] == 'EGV'].groupby(['RIDER', 'date'])['local_timestamp', 'Event Subtype', 'Source Device ID', 'Transmitter ID'].nunique()

# glucose file with readings more than the max per day
max_readings = 24*60/5 # max number of readings on one day on 5 min interval
count_exceed = count_readings[count_readings['local_timestamp'] > max_readings]
count_exceed.to_csv('exceed/glucose_exceed.csv')
print("Days with more than the max number of readings:\n",
	count_exceed)

# save clean glucose file with max readings exceeded
df_exceed = pd.DataFrame(columns=df.columns)
for r, d in count_exceed.index:
	df_exceed = df_exceed.append(df[(df.RIDER == r) & (df.date == d)])
df_exceed.to_csv(path+'exceed/dexcom_clean_exceed.csv')

for i, (r, d) in enumerate(count_exceed.index):
	df_exceed_i = df_exceed[(df_exceed.RIDER == r) & (df_exceed.date == d) & (df_exceed['Event Type'] == 'EGV')]
	df_exceed_i['min'] = df_exceed_i['local_timestamp'].dt.round('min')
	df_exceed_i['dupl'] = df_exceed_i['min'].duplicated(keep=False).astype(int)

	ax = sns.scatterplot(df_exceed_i['local_timestamp'], df_exceed_i['Glucose Value (mg/dL)'], 
		hue=df_exceed_i['dupl'])
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(r,d))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'exceed/glucose_%s_%s.pdf'%(r,d), bbox_inches='tight')
	plt.savefig(path+'exceed/glucose_%s_%s.png'%(r,d), dpi=300, bbox_inches='tight')
	plt.close()

# interval within 4 min
df_egv = df[df['Event Type'] == 'EGV']
short = df_egv.loc[df_egv.local_timestamp.diff() < '4min', ['RIDER', 'date']].drop_duplicates()

df_short = pd.DataFrame(columns=df_egv.columns)
for i, (r, d) in short.iterrows():
	df_short = df_short.append(df_egv[(df_egv.RIDER == r) & (df_egv.date == d)])
df_short.to_csv(path+'short/dexcom_clean_short.csv')

for i, (r, d) in short.iterrows():
	df_short_i = df_short[(df_short.RIDER == r) & (df_short.date == d)]

	ax = sns.scatterplot(df_short_i['local_timestamp'], df_short_i['Glucose Value (mg/dL)'], 
		hue=df_short_i.local_timestamp.diff() < '4min')
	ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=4))   # every 4 hours
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))  # hours and minutes

	plt.title('Rider %s - %s'%(r,d))
	plt.ylabel('Glucose Value (mg/dL) EGV')
	plt.savefig(path+'short/glucose_%s_%s.pdf'%(r,d), bbox_inches='tight')
	plt.savefig(path+'short/glucose_%s_%s.png'%(r,d), dpi=300, bbox_inches='tight')
	plt.close()

df.reset_index(drop=True, inplace=True)
df.drop('date', axis=1, inplace=True)

df.to_csv(path+'dexcom_clean.csv')

# group readings in intervals
df_interval = df[df['Event Type'] == 'EGV']\
	.groupby(['RIDER', pd.Grouper(key='local_timestamp', freq='5min')])\
	.agg({'timestamp':['unique', 'nunique'],
		  'Glucose Value (mg/dL)':'nunique',
		  'Transmitter Time (Long Integer)':['nunique', 'unique']})
df_interval.rename(columns={'Transmitter Time (Long Integer)':'transmitter', 
							'Glucose Value (mg/dL)':'glucose'}, inplace=True)

# check if there are more than 2 or 3 readings anywhere
print("Number of intervals with more than 2 measurements: ", 
	((df_interval[('timestamp', 'nunique')] > 2) |
	 (df_interval[('glucose', 'nunique')] > 2) |
	 (df_interval[('transmitter', 'nunique')] > 2)).sum())
print("Number of intervals with more than 3 measurements: ", 
	((df_interval[('timestamp', 'nunique')] > 3) |
	 (df_interval[('glucose', 'nunique')] > 3) |
	 (df_interval[('transmitter', 'nunique')] > 3)).sum())

df_interval.columns = [x[0]+'_'+x[1] for x in df_interval.columns]

# calculate difference in transmitter time between readings in the same interval
df_interval['transmitter_diff'] = df_interval['transmitter_unique'].apply(lambda x: x[1] - x[0] if len(x) == 2 else ([x[1]-x[0],x[2]-x[1]] if len(x) == 3 else np.nan))
df_interval['transmitter_negdiff'] = df_interval['transmitter_diff'].apply(lambda x: x < 0 if isinstance(x, float) else ((np.array(x) < 0).any() if isinstance(x, list) else np.nan))
df_interval.drop('transmitter_unique', axis=1, inplace=True)

# fill up missing intervals
year_interval = pd.DataFrame(np.nan, columns=['nunique'], index=pd.MultiIndex.from_product([df.RIDER.unique(),
					pd.date_range(start=datetime.datetime(2018,12,1,0), end=datetime.datetime(2019,12,1), freq='5min')],
				names=['RIDER', 'local_timestamp']))

df_interval = pd.merge(df_interval, year_interval, how='right', 
	left_index=True, right_index=True, validate='one_to_one')
df_interval.drop('nunique', axis=1, inplace=True)

# print how often intervals have more than 2 readings
print("local_timestamp nunique > 2: ", (df_interval['local_timestamp_nunique'] > 1).sum())
print("glucose nunique > 2: ", (df_interval['glucose_nunique'] > 1).sum())
print("transmitter nunique > 2: ", (df_interval['transmitter_nunique'] > 1).sum())
print("transmitter diff < 0: ", (df_interval['transmitter_negdiff'].sum()))


if not os.path.exists(path+'exceed/'):
	os.mkdir(path+'exceed/')
if not os.path.exists(path+'short/'):
	os.mkdir(path+'short/')
if not os.path.exists(path+'tz_change/'):
	os.mkdir(path+'tz_change/')
if not os.path.exists(path+'all/'):
	os.mkdir(path+'all/')

# TODO: can remove many things
"""
# Note: somehow the rider reset it by 1:30 and 22:30
df_tz.loc[(10,30), 'local_timestamp_final_diff_0'].iloc[2] = notfound.loc[450245, 'tz_diff']#.round('min')
df_tz.loc[(10,30), 'timezone'].iloc[3] = df_tz.loc[(10,30), 'timezone'].iloc[2] + notfound.loc[450245, 'tz_diff']#.round('min')
#df_tz.loc[(10,30), 'timestamp_max'].iloc[2] = notfound.loc[450245, 't_max'] - df_tz.loc[(10,30), 'timezone'].iloc[2]
df_tz.loc[(10,30), 'timestamp_min'].iloc[3] = notfound.loc[450245, 't_min'] - df_tz.loc[(10,30), 'timezone'].iloc[3]

df_tz.loc[(10,30), 'local_timestamp_final_diff_0'].iloc[3] = notfound.loc[450275, 'tz_diff']#.round('min')
#df_tz.loc[(10,31), 'timezone'] = df_tz.loc[(10,30), 'timezone'].iloc[3] + notfound.loc[450275, 'tz_diff'].round('min')
df_tz.loc[(10,30), 'timestamp_max'].iloc[3] = notfound.loc[450275, 't_max'] - df_tz.loc[(10,30), 'timezone'].iloc[3]
#df_tz.loc[(10,31), 'timestamp_min'] = notfound.loc[450275, 't_min'] - df_tz.loc[(10,31), 'timezone']
"""


df_egv['device_diff'] = df_egv['Source Device ID'] != df_egv['Source Device ID'].shift()

# Obtain all timezone changes due to device changes
df_egv[df_egv['device_diff'] & (df_egv.RIDER.diff() == 0)\
		& (df_egv['transmitter_order'].diff() == 0)\
		& ((df_egv.local_timestamp_diff > '5min')\
		| (df_egv.local_timestamp_diff < '-5min'))
		& ~(df_egv['change'])].to_csv('device_diff.csv')

df_egv['transmitter_localtime_diff'] = pd.to_timedelta(df_egv['transmitter_diff'], 'sec') - df_egv['local_timestamp_diff']
df_egv[(df_egv['transmitter_localtime_diff'].abs() > '5min')\
		& (df_egv.RIDER.diff() == 0)\
		& (df_egv['transmitter_order'].diff() == 0)\
		& ~df_egv['change']] # travel
df_egv[(df_egv['transmitter_localtime_diff'].abs() > '23h')\
		& (df_egv.RIDER.diff() == 0)\
		& (df_egv['transmitter_order'].diff() == 0)\
		& ~df_egv['change']].to_csv('check_anomalies.csv') # anomalies

# TODO!! Device changes to identify weird timezone changes


df_tz.loc[(2,8), 'local_timestamp_final_max_0'] = df_egv.loc[(df_egv.RIDER == 2) \
	& (df_egv.local_timestamp <= df_tz.loc[(2,9), 'local_timestamp_min'].values[0]), 'local_timestamp'].iloc[-1]



notfound.loc[193552] = [4, pd.to_datetime('2018-12-09 07:32:28'),
						pd.to_datetime('2018-12-09 05:28:12'),
						pd.to_timedelta('0 days 02:00:00')] # TODO
notfound.loc[193561] = [4, pd.to_datetime('2018-12-09 17:08:13'),
						pd.to_datetime('2018-12-09 08:12:28'),
						pd.to_timedelta('-1 days +12:00:00')] # TODO
notfound.loc[193779] = [4, pd.to_datetime('2018-12-10 21:18:07'),
						pd.to_datetime('2018-12-10 11:13:07'),
						pd.to_timedelta('0 days 10:00:00')] # TODO


notfound.loc[227457] = [4, pd.to_datetime('2019-10-23 20:37:45'),
						pd.to_datetime('2019-10-23 18:32:46'),
						pd.to_timedelta('0 days 02:04:59')] # TODO
notfound.loc[227066] = [4, pd.to_datetime('2019-10-20 20:12:50'),
						pd.to_datetime('2019-10-20 19:07:51'),
						pd.to_timedelta('0 days 01:04:59')] # TODO

notfound.loc[224283] = [4, pd.to_datetime('2019-10-09 22:00:06'),
						pd.to_datetime('2019-10-10 21:58:03'),
						pd.to_timedelta('-1 days +00:02:03')] # TODO

notfound.loc[224410] = [4, pd.to_datetime('2019-10-10 10:15:07'),
						pd.to_datetime('2019-10-11 10:13:04'),
						pd.to_timedelta('-1 days +00:02:03')] # TODO

notfound.loc[306605] = [5, pd.to_datetime('2019-10-17 07:16:15'),
						pd.to_datetime('2019-10-17 06:11:15'),
						pd.to_timedelta('0 days 01:00:00')] # TODO

notfound.loc[306626] = [5, pd.to_datetime('2019-10-17 08:01:14'),
						pd.to_datetime('2019-10-17 08:56:13'),
						pd.to_timedelta('-1 days +23:05:01')] # TODO

notfound.loc[308126] = [5, pd.to_datetime('2019-10-22 14:55:58'),
						pd.to_datetime('2019-10-22 21:50:58'),
						pd.to_timedelta('-1 days +17:00:00')] # TODO
notfound.loc[309386] = [5, pd.to_datetime('2019-11-01 01:20:32'),
						pd.to_datetime('2019-10-26 23:55:46'),
						pd.to_timedelta('-1 days +23:00:00')] # TODO


notfound.loc[319713] = [6, pd.to_datetime('2018-12-09 11:09:18'),
						pd.to_datetime('2018-12-09 03:04:18'),
						pd.to_timedelta('0 days 08:05:00')]

notfound.loc[409993] = [10, pd.to_datetime('2019-02-20 15:50:03'),
						pd.to_datetime('2019-02-20 14:45:03'),
						pd.to_timedelta('0 days 01:05:00')] # TODO

notfound.loc[410248] = [10, pd.to_datetime('2019-02-21 15:50:01'),
						pd.to_datetime('2019-02-21 13:45:01'),
						pd.to_timedelta('0 days 02:05:00')] # TODO

notfound.loc[410261] = [10, pd.to_datetime('2019-02-21 20:05:01'),
						pd.to_datetime('2019-02-21 18:20:02'),
						pd.to_timedelta('0 days 01:00:00')] # TODO
notfound.loc[412469] = [10, pd.to_datetime('2019-03-03 13:29:45'),
						pd.to_datetime('2019-03-03 16:24:44'),
						pd.to_timedelta('-1 days +21:00:00')] # TODO
notfound.loc[412518] = [10, pd.to_datetime('2019-03-06 06:19:38'),
						pd.to_datetime('06:19:38 2019-03-03'),
						pd.to_timedelta('-1 days +23:00:00')] # TODO

notfound.loc[426626] = [10, pd.to_datetime('2019-05-22 10:05:03'),
						pd.to_datetime('2019-05-22 09:00:03'),
						pd.to_timedelta('0 days 01:00:00')] # TODO
notfound.loc[426660] = [10, pd.to_datetime('2019-05-22 13:55:02'),
						pd.to_datetime('2019-05-22 12:50:03'),
						pd.to_timedelta('0 days 00:59:59')] # TODO

notfound.loc[427680] = [10, pd.to_datetime('2019-05-26 09:04:57'),
						pd.to_datetime('2019-05-26 09:59:57'),
						pd.to_timedelta('-1 days +23:05:00')] # TODO

notfound.loc[427706] = [10, pd.to_datetime('2019-05-26 10:14:57'),
						pd.to_datetime('2019-05-26 11:09:57'),
						pd.to_timedelta('-1 days +23:05:00')] # TODO

notfound.loc[445183] = [10, pd.to_datetime('2019-10-06 11:36:28'),
						pd.to_datetime('2019-10-06 10:31:29'),
						pd.to_timedelta('0 days 01:04:5')] # TODO
notfound.loc[445342] = [10, pd.to_datetime('2019-10-07 06:51:26'),
						pd.to_datetime('2019-10-07 00:46:26'),
						pd.to_timedelta('0 days 06:05:00')] # TODO

notfound.loc[527859] = [13, pd.to_datetime('2019-05-19 07:48:50'),
						pd.to_datetime('2019-05-19 22:43:07'),
						pd.to_timedelta('-1 days +09:05:43')] # TODO
notfound.loc[527860] = [13, pd.to_datetime('2019-05-20 07:53:36'),
						pd.to_datetime('2019-05-19 07:48:50'),
						pd.to_timedelta('1 days 00:04:46')] # TODO

notfound.loc[626935] = [15, pd.to_datetime('2019-10-23 12:55:05'),
						pd.to_datetime('2019-10-23 19:51:13'),
						pd.to_timedelta('-1 days +17:03:52')] # TODO


notfound.loc[629069] = [15, pd.to_datetime('2019-10-30 09:03:05'),
						pd.to_datetime('2019-10-31 08:59:07'),
						pd.to_timedelta('-1 days +00:03:58')] # TODO
notfound.loc[628373] = [15, pd.to_datetime('2019-10-28 19:44:16'),
						pd.to_datetime('2019-10-28 17:39:45'),
						pd.to_timedelta('0 days 02:04:31')] # TODO

notfound.loc[629580] = [15, pd.to_datetime('2019-11-02 12:01:02'),
						pd.to_datetime('2019-11-02 14:52:53'),
						pd.to_timedelta('-1 days +21:08:09')] # TODO

print("INSERT\n", notfound.loc[[317729, 317751, 317937, 317939, 318128, 318150, 318167, 318205, 
	318363, 318364, 318400, 318438, 318443, 318464, 318626, 318636, 318680, 
	318681, 318706, 318734, 318750, 318761, 318916, 318943, 318961, 318972, 
	318977, 319009, 319015, 319025, 319030, 319048, 319064, 319091, 319233, 
	319259, 319276, 319292, 319342, 319364, 319381, 319435, 319476, 319493,
	319498, 319652]])
df_tz = insert_rows(df_tz, (6,0), notfound, 
	[317729, 317751, 317937, 317939, 318128, 318150, 318167, 318205, 
	318363, 318364, 318400, 318438, 318443, 318464, 318626, 318636, 318680, 
	318681, 318706, 318734, 318750, 318761, 318916, 318943, 318961, 318972, 
	318977, 319009, 319015, 319025, 319030, 319048, 319064, 319091, 319233, 
	319259, 319276, 319292, 319342, 319364, 319381, 319435, 319476, 319493,
	319498, 319652])
notfound.drop([317729, 317751, 317937, 317939, 318128, 318150, 318167, 
	318205, 318363, 318364, 318400, 318438, 318443, 318464, 318626, 318636, 
	318680, 318681, 318706, 318734, 318750, 318761, 318916, 318943, 318961, 
	318972, 318977, 319009, 319015, 319025, 319030, 319048, 319064, 319091, 
	319233, 319259, 319276, 319292, 319342, 319364, 319381, 319435, 319476, 
	319493, 319498, 319652], inplace=True)

317641 6 2018-12-01 00:04:38 2018-12-01 17:26:39 -1 days +00:03:00
317729 6 2018-12-02 09:06:37 2018-12-01 08:14:39 0 days 23:56:58
317751 6 2018-12-01 16:34:39 2018-12-02 15:46:36 -1 days +00:03:04
317937 6 2018-12-03 10:11:34 2018-12-02 07:59:37 0 days 23:56:57
317939 6 2018-12-02 12:44:36 2018-12-03 10:16:34 -1 days +00:03:02
318128 6 2018-12-04 10:16:30 2018-12-03 08:04:34 0 days 23:56:56
318150 6 2018-12-03 12:59:35 2018-12-04 12:51:29 -1 days +00:03:06
318167 6 2018-12-04 14:21:29 2018-12-03 14:19:33 0 days 23:56:56
318205 6 2018-12-03 17:34:34 2018-12-04 17:26:29 -1 days +00:03:05
318363 6 2018-12-05 06:36:32 2018-12-04 06:39:32 0 days 23:56:55
318364 6 2018-12-04 06:44:32 2018-12-05 06:36:32 -1 days +00:03:05
318400 6 2018-12-05 09:41:27 2018-12-04 09:39:30 0 days 23:56:57
318438 6 2018-12-04 12:54:30 2018-12-05 12:46:27 -1 days +00:03:03
318443 6 2018-12-05 13:16:26 2018-12-04 13:14:31 0 days 23:56:55
318464 6 2018-12-04 15:04:31 2018-12-05 14:56:26 -1 days +00:03:05
318626 6 2018-12-06 04:31:23 2018-12-05 04:29:28 0 days 23:56:55
318636 6 2018-12-05 05:24:29 2018-12-06 05:16:24 -1 days +00:03:05
318680 6 2018-12-06 11:01:37 2018-12-05 11:04:29 0 days 23:56:55
318681 6 2018-12-05 11:09:28 2018-12-06 11:01:37 -1 days +00:03:04
318706 6 2018-12-06 13:11:24 2018-12-05 13:09:28 0 days 23:56:56
318734 6 2018-12-05 15:34:28 2018-12-06 15:26:22 -1 days +00:03:06
318750 6 2018-12-06 16:51:23 2018-12-05 16:49:28 0 days 23:56:55
318761 6 2018-12-05 17:49:28 2018-12-06 17:41:22 -1 days +00:03:06
318916 6 2018-12-07 06:41:21 2018-12-06 06:39:26 0 days 23:56:55
318943 6 2018-12-06 08:59:27 2018-12-07 08:51:20 -1 days +00:03:07
318961 6 2018-12-07 10:26:20 2018-12-06 10:24:25 0 days 23:56:55
318972 6 2018-12-06 11:24:25 2018-12-07 11:16:20 -1 days +00:03:05
318977 6 2018-12-07 11:46:21 2018-12-06 11:44:26 0 days 23:56:55
319015 6 2018-12-07 14:56:19 2018-12-06 14:54:25 0 days 23:56:54
319030 6 2018-12-07 16:11:19 2018-12-06 16:09:24 0 days 23:56:55
319064 6 2018-12-07 19:01:19 2018-12-06 18:59:25 0 days 23:56:54
319233 6 2018-12-08 09:06:17 2018-12-07 09:04:24 0 days 23:56:53
# 319009     6 2018-12-06 14:29:24 2018-12-07 14:21:19 -1 days +00:08:05
# 319025     6 2018-12-06 15:49:24 2018-12-07 15:41:19 -1 days +00:08:05
# 319048     6 2018-12-06 17:44:25 2018-12-07 17:36:18 -1 days +00:08:07
# 319091     6 2018-12-06 21:19:24 2018-12-07 21:11:18 -1 days +00:08:06
# 319259     6 2018-12-07 11:19:23 2018-12-08 11:11:16 -1 days +00:08:07
# 319276     6 2018-12-08 12:41:16 2018-12-07 12:39:23   1 days 00:01:53
# 319292     6 2018-12-07 14:04:24 2018-12-08 13:56:16 -1 days +00:08:08
# 319342     6 2018-12-08 18:11:16 2018-12-07 18:09:21   1 days 00:01:55
# 319364     6 2018-12-07 20:04:22 2018-12-08 19:56:15 -1 days +00:08:07
# 319381     6 2018-12-08 21:26:15 2018-12-07 21:24:21   1 days 00:01:54
# 319435     6 2018-12-08 03:59:21 2018-12-09 01:51:14 -1 days +02:08:07
# 319476     6 2018-12-09 05:21:14 2018-12-08 07:19:21   0 days 22:01:53
# 319493     6 2018-12-08 08:49:21 2018-12-09 06:41:14 -1 days +02:08:07
# 319498     6 2018-12-09 07:11:14 2018-12-08 09:09:20   0 days 22:01:54
# 319652     6 2018-12-08 22:04:19 2018-12-09 19:56:12 -1 days +02:08:07

---

#317729     6 2018-12-02 09:06:37 2018-12-01 08:14:39   1 days 00:51:58
#317751     6 2018-12-01 16:34:39 2018-12-02 15:46:36 -1 days +00:48:03
#317937     6 2018-12-03 10:11:34 2018-12-02 07:59:37   1 days 02:11:57
#317939     6 2018-12-02 12:44:36 2018-12-03 10:16:34 -1 days +02:28:02
#318128     6 2018-12-04 10:16:30 2018-12-03 08:04:34   1 days 02:11:56
#318150     6 2018-12-03 12:59:35 2018-12-04 12:51:29 -1 days +00:08:06
#318167     6 2018-12-04 14:21:29 2018-12-03 14:19:33   1 days 00:01:56
#318205     6 2018-12-03 17:34:34 2018-12-04 17:26:29 -1 days +00:08:05
#318363     6 2018-12-05 06:36:32 2018-12-04 06:39:32   0 days 23:57:00
#318364     6 2018-12-04 06:44:32 2018-12-05 06:36:32 -1 days +00:08:00
#318400     6 2018-12-05 09:41:27 2018-12-04 09:39:30   1 days 00:01:57
#318438     6 2018-12-04 12:54:30 2018-12-05 12:46:27 -1 days +00:08:03
#318443     6 2018-12-05 13:16:26 2018-12-04 13:14:31   1 days 00:01:55
#318464     6 2018-12-04 15:04:31 2018-12-05 14:56:26 -1 days +00:08:05
#318626     6 2018-12-06 04:31:23 2018-12-05 04:29:28   1 days 00:01:55
#318636     6 2018-12-05 05:24:29 2018-12-06 05:16:24 -1 days +00:08:05
#318680     6 2018-12-06 11:01:37 2018-12-05 11:04:29   0 days 23:57:08
#318681     6 2018-12-05 11:09:28 2018-12-06 11:01:37 -1 days +00:07:51
#318706     6 2018-12-06 13:11:24 2018-12-05 13:09:28   1 days 00:01:56
#318734     6 2018-12-05 15:34:28 2018-12-06 15:26:22 -1 days +00:08:06
#318750     6 2018-12-06 16:51:23 2018-12-05 16:49:28   1 days 00:01:55
#318761     6 2018-12-05 17:49:28 2018-12-06 17:41:22 -1 days +00:08:06
#318916     6 2018-12-07 06:41:21 2018-12-06 06:39:26   1 days 00:01:55
#318943     6 2018-12-06 08:59:27 2018-12-07 08:51:20 -1 days +00:08:07
#318961     6 2018-12-07 10:26:20 2018-12-06 10:24:25   1 days 00:01:55
#318972     6 2018-12-06 11:24:25 2018-12-07 11:16:20 -1 days +00:08:05
#318977     6 2018-12-07 11:46:21 2018-12-06 11:44:26   1 days 00:01:55
#319009     6 2018-12-06 14:29:24 2018-12-07 14:21:19 -1 days +00:08:05
#319015     6 2018-12-07 14:56:19 2018-12-06 14:54:25   1 days 00:01:54
#319025     6 2018-12-06 15:49:24 2018-12-07 15:41:19 -1 days +00:08:05
#319030     6 2018-12-07 16:11:19 2018-12-06 16:09:24   1 days 00:01:55
#319048     6 2018-12-06 17:44:25 2018-12-07 17:36:18 -1 days +00:08:07
#319064     6 2018-12-07 19:01:19 2018-12-06 18:59:25   1 days 00:01:54
#319091     6 2018-12-06 21:19:24 2018-12-07 21:11:18 -1 days +00:08:06
#319233     6 2018-12-08 09:06:17 2018-12-07 09:04:24   1 days 00:01:53
#319259     6 2018-12-07 11:19:23 2018-12-08 11:11:16 -1 days +00:08:07
#319276     6 2018-12-08 12:41:16 2018-12-07 12:39:23   1 days 00:01:53
#319292     6 2018-12-07 14:04:24 2018-12-08 13:56:16 -1 days +00:08:08
#319342     6 2018-12-08 18:11:16 2018-12-07 18:09:21   1 days 00:01:55
#319364     6 2018-12-07 20:04:22 2018-12-08 19:56:15 -1 days +00:08:07
#319381     6 2018-12-08 21:26:15 2018-12-07 21:24:21   1 days 00:01:54
#319435     6 2018-12-08 03:59:21 2018-12-09 01:51:14 -1 days +02:08:07
#319476     6 2018-12-09 05:21:14 2018-12-08 07:19:21   0 days 22:01:53
#319493     6 2018-12-08 08:49:21 2018-12-09 06:41:14 -1 days +02:08:07
#319498     6 2018-12-09 07:11:14 2018-12-08 09:09:20   0 days 22:01:54
#319652     6 2018-12-08 22:04:19 2018-12-09 19:56:12 -1 days +02:08:07


# MANUAL finds
print(df_egv[df_egv['local_timestamp_diff'] > '21h'].loc[316000:320000])
for i in [, , , , , 319276, 319342, 319381, 319476, 319498]:
	notfound.loc[i] = [6, df_egv.loc[i, 'local_timestamp'],
						df_egv.loc[i-1, 'local_timestamp'],
						df_egv.loc[i, 'local_timestamp_diff']]
df_egv.loc[[317729, 317937, 318128, 318363, 318680, 319276, 319342, 319381, 319476, 319498], 'change'] = True
print("MANUAL add\n", notfound.loc[[317729, 317937, 318128, 318363, 318680, 319276, 319342, 319381, 319476, 319498]])

print(df_egv[df_egv['local_timestamp_diff'] < '-21h'].loc[316000:320000])
for i in [, , , 319009, 319025, 319048, 319091, 319259, 319292, 319364, 319435, 319493, 319652]:
	notfound.loc[i] = [6, df_egv.loc[i, 'local_timestamp'],
						df_egv.loc[i-1, 'local_timestamp'],
						df_egv.loc[i, 'local_timestamp_diff']]
df_egv.loc[[317751, 317939, 318681, 319009, 319025, 319048, 319091, 319259, 319292, 319364, 319435, 319493, 319652], 'change'] = True
print("MANUAL add\n", notfound.loc[[317751, 317939, 318681, 319009, 319025, 319048, 319091, 319259, 319292, 319364, 319435, 319493, 319652]])



# Some day changes that we found manually
notfound.loc[224417] = [4, pd.to_datetime('2019-10-11 11:13:03'),
						pd.to_datetime('2019-10-10 10:45:05'),
						pd.to_timedelta('1 days 00:27:58')]
notfound = notfound.sort_index()
df_egv.loc[224417, 'change'] = True
print_row("MANUAL add\n", notfound.loc[224417])



notfound.loc[193548] = [4, pd.to_datetime('2018-12-09 05:13:14'),
						pd.to_datetime('2018-12-08 19:08:15'),
						pd.to_timedelta('0 days 10:00:00')]


# MANUAL find
print(df_egv[df_egv['local_timestamp_diff'] > '5h'].loc[74000:77000])
for i in [74792, 74805, 75514, 76085, 76658]:
	notfound.loc[i] = [4, df_egv.loc[i, 'local_timestamp'],
						df_egv.loc[i-1, 'local_timestamp'],
						df_egv.loc[i, 'local_timestamp_diff']]
df_egv.loc[[74792, 74805, 75514, 76085, 76658], 'change'] = True
print("MANUAL add\n", notfound.loc[[74792, 74805, 75514, 76085, 76658]])

print(df_egv[df_egv['local_timestamp_diff'] < '-5h'].loc[74000:77000])
for i in [74791, 74797, 75503, 76081, 76655]:
	notfound.loc[i] = [4, df_egv.loc[i, 'local_timestamp'],
						df_egv.loc[i-1, 'local_timestamp'],
						df_egv.loc[i, 'local_timestamp_diff']]
df_egv.loc[[74791, 74797, 75503, 76081, 76655], 'change'] = True
print("MANUAL add\n", notfound.loc[[74791, 74797, 75503, 76081, 76655]])

print("INSERT\n", notfound.loc[[74791, 74792, 74797, 74805, 75503, 75514, 76081, 76085, 76655, 76658]])
df_tz = insert_rows(df_tz, (4,9), notfound, [74791, 74792, 74797, 74805, 75503, 75514, 76081, 76085, 76655, 76658])


# Identify travelling gap: when transmitter time diff == 300 and local_timestamp_diff > 5 min
# Note: do a little bit more than 5 min, in case there are issues with the receiver (this is quite often)
# A gap from travelling should be at least 1 hour anyway
df_egv['gap'] = (df_egv['transmitter_diff'] >= 290) & (df_egv['transmitter_diff'] <= 310)\
			  & (df_egv['local_timestamp_diff'] > datetime.timedelta(minutes=45))\
			  & (df_egv['transmitter_order'].diff() == 0) # no change between transmitters
print("Number of gaps: ", df_egv['gap'].sum())

# DUP
# identify travelling dup: when transmitter time diff == 300 and local_timestamp_diff < 5 min
df_egv['dup'] = (df_egv['transmitter_diff'] >= 290) & (df_egv['transmitter_diff'] <= 310)\
			  & (df_egv['local_timestamp_diff'] < datetime.timedelta(0))\
			  & (df_egv['transmitter_order'].diff() == 0) # no change between transmitters
print("Number of dups: ", df_egv['dup'].sum())


# Check if there is overlapp between transmitters (of the same rider)
# Note: there is overlap because the riders double up when the transmitter is at the end of its lifetime.
# In this case select the data of the transmitter that is newest.
# TODO: check how many measurements there are
df_drop_transmitter_overlap = []
for i in df_transmitter.index.get_level_values(0).unique():
	for j, tid in enumerate(df_transmitter.loc[i].index):
		try:
			date_curr_max = df_transmitter.loc[i].loc[tid, ('local_timestamp', 'max')]
			date_next_min = df_transmitter.loc[i].iloc[j+1][('local_timestamp', 'min')]

			# if there is overlap
			if date_curr_max >= date_next_min:
				print("Overlap transmitters")
				print(i, df_transmitter.loc[i].iloc[j:j+2])

				# select the egv data from the newest transmitter (so drop data from the old transmitter)
				print("SELECT newest transmitter")
				drop = df[(df.RIDER == i) & (df['Event Type'] == 'EGV') & (df['Transmitter ID'] == tid) &
					(df.local_timestamp >= date_next_min) & (df.local_timestamp <= date_curr_max)]
				df_drop_transmitter_overlap.append(drop)
				df.drop(drop.index, inplace=True)
		except IndexError:
			pass
pd.concat(df_drop_transmitter_overlap).to_csv(path+'drop/transmitter_overlap.csv')
del df_drop_transmitter_overlap, drop

# It seems like some riders manually reset their transmitters.
# this gives problems later when we sort by transmitter time
# TODO: this will go wrong if there is also a timezone change at the same time

# see 6_transmitter_80QJ2F_time_reset.png
df.loc[386337:401634, 'Transmitter Time (Long Integer)'] += df['local_timestamp'].diff().loc[386337].seconds - df['Transmitter Time (Long Integer)'].diff().loc[386337]
print("FIX transmitter time reset of 6: 80QJ2F")

# see 6_transmitter_80RE8H_time_reset.png
df.loc[413638:438840, 'Transmitter Time (Long Integer)'] += df['local_timestamp'].diff().loc[413638].seconds - df['Transmitter Time (Long Integer)'].diff().loc[413638]
print("FIX transmitter time reset of 6: 80RE8H")

# see 14_transmitter_80RRBL_time_reset.png
df.loc[651268:660645, 'Transmitter Time (Long Integer)'] += df['local_timestamp'].diff().loc[651268].seconds - df['Transmitter Time (Long Integer)'].diff().loc[651268]
print("FIX transmitter time reset of 14: 80RRBL")

# check if any other transmitters have been reset


# Solution: 
# 1) assign each transmitter an order index
# 2) look at all readings within the time window of the transmitter and assign them the same order index
# 3) sort by: rider - order index - transmitter time


# TODO: all non-EGV readings
# note: there are still measurements left that are not EGV
# todo: measurements with nan transmitter that are not EGV
# Problem: there are still non-EGV readings in there, that do not have a transmitter ID
# or that already have the transmitter ID of the next transmitter if they are calibrations.
# This will give problems when we sort by transmitter ID.


count_dupl = df_dupl.groupby(['RIDER', 'date'])['local_timestamp', 'Event Type', 'Event Subtype', 'Source Device ID', 'Transmitter ID'].nunique()
count_dupl.to_csv(path+'drop/glucose_dupl.csv')
print("Duplicate rows count:\n", count_dupl)

# duplicate timestamps (per rider and event type)
cols_dupl_ts = ['RIDER', 'local_timestamp', 'Event Type', 'Event Subtype']
df_dupl_ts = df[df.duplicated(subset=cols_dupl_ts, keep=False)]
df_dupl_ts.to_csv(path+'drop/dexcom_clean_dupl_ts.csv')

print("CHECK Number of duplicate entries for the same RIDER, Event type, Event Subtype and local timestamp: ", len(df_dupl_ts))
print("Duplicate timestamps for each group of activity type:\n",
	df_dupl_ts.groupby(['Event Type'])['local_timestamp'].nunique())

count_dupl_ts = df_dupl_ts.groupby(['RIDER', 'date'])['local_timestamp', 'Event Type', 'Event Subtype', 'Source Device ID', 'Transmitter ID'].nunique()
count_dupl_ts.to_csv(path+'drop/glucose_dupl_ts.csv')
print("Duplicate timestamps count:\n", count_dupl_ts)
