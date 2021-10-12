# -------------------------- Glucose availability (1)
# create calendar with glucose availability
max_readings = 24*60/5+1
df_frame = pd.DataFrame(index=pd.MultiIndex.from_product([df.source.unique(), df.RIDER.unique()]))
df_avail = df.groupby(['source', 'RIDER', df.timestamp.dt.date])['Glucose Value (mg/dL)'].count().unstack() / max_readings
df_avail = pd.concat([df_frame, df_avail], axis=1)

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(df_avail.loc['Dexcom CLARITY EU'], cmap='Greys', vmax=1, alpha=1, cbar_kws=dict(label='fraction of possible CGM readings per day'))
ax = sns.heatmap(df_avail.loc['TrainingPeaks'].clip(1), cmap='Reds', vmin=0, cbar=False)
ax = sns.heatmap(df_avail.loc['Dexcom CLARITY EU'], cmap='Blues', vmax=1, cbar=False)
ax = sns.heatmap(df_avail.loc['Dexcom CLARITY US'], cmap='Greens', vmax=1, cbar=False)

plt.legend(handles=[Patch(color=matplotlib.cm.Blues(360), label='Dexcom CLARITY EU'),
					Patch(color=matplotlib.cm.Greens(360), label='Dexcom CLARITY US'),
					Patch(color=matplotlib.cm.Reds(360), label='Proof missing data exists elsewhere (TrainingPeaks)')], 
			ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.01))
""" try to plot completeness
ax2 = ax.twinx()
ax2.set_yticklabels((df_avail.loc['Dexcom CLARITY EU'].clip(0,1).fillna(0) + df_avail.loc['Dexcom CLARITY US'].clip(0,1).fillna(0)).mean(axis=1).tolist())
"""
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig(SAVE_PATH+'availability/glucose_availability.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(SAVE_PATH+'availability/glucose_availability_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# -------------------------- Glucose availability (2) : during training sessions
df_training = pd.read_csv(DATA_PATH+'training.csv', index_col=0)
df_training['timestamp_min'] = pd.to_datetime(df_training['timestamp_min'])
df_training['timestamp_max'] = pd.to_datetime(df_training['timestamp_max'])
df_training['local_timestamp_min'] = pd.to_datetime(df_training['local_timestamp_min'])
df_training['local_timestamp_max'] = pd.to_datetime(df_training['local_timestamp_max'])

df_training['timestamp_min_r'] = df_training['timestamp_min'].round('5min')
df_training['timestamp_max_r'] = df_training['timestamp_max'].round('5min')
df_training['local_timestamp_min_r'] = df_training['local_timestamp_min'].round('5min')
df_training['local_timestamp_max_r'] = df_training['local_timestamp_max'].round('5min')

df_training.set_index(['RIDER', 'file_id'], inplace=True)

max_time = (df_training['timestamp_max_r'] - df_training['timestamp_min_r']).max()

training_avail = pd.DataFrame(index=df_training.index, columns=np.arange(max_time/'5min'))
for (i, n), (t_min, t_max) in df_training[['timestamp_min_r', 'timestamp_max_r']].iterrows():
	glucose = df[(df.RIDER == i) & (df.timestamp >= t_min) & (df.timestamp <= t_max)]
	glucose.set_index((glucose['timestamp'] - t_min)/pd.to_timedelta('5min'), inplace=True)
	training_avail.loc[(i,n)] = glucose['Glucose Value (mg/dL)'].notna()

training_avail = training_avail.groupby(level=0, axis=0).sum()

fig, ax = plt.subplots(figsize=(15,6))
sns.heatmap(training_avail, cmap='Blues', cbar_kws=dict(label='number of CGM readings'))
plt.xlabel('Time in training session (min)')
plt.ylabel('Rider')
plt.xticks(ticks=training_avail.columns[::6], 
	labels=pd.to_datetime(training_avail.columns[::6]*5, unit='m').strftime('%H:%M'), rotation=0)
plt.xlim((0,100))
plt.savefig(SAVE_PATH+'availability/glucose_availability_training.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability_training.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(SAVE_PATH+'availability/glucose_availability_training_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability_training_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()



dates_2019 = pd.date_range(start='12/1/2018', end='11/30/2019').date
df_avail = pd.DataFrame(index=df.RIDER.unique(), columns=dates_2019)
for i in df.RIDER.unique():
	df_avail.loc[i, df[df.RIDER == i].timestamp.dt.date.unique()] = 1
df_avail.fillna(0, inplace=True)

# plot glucose availability per day
plt.figure(figsize=(25,10))
ax = sns.heatmap(df_avail, cmap='Blues', cbar=False, linewidths=.01) 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig(savepath+'glucose_availability_day_all.pdf', bbox_inches='tight')
plt.savefig(savepath+'glucose_availability_day_all.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(savepath+'glucose_availability_day_all_NAME.pdf', bbox_inches='tight')
plt.savefig(savepath+'glucose_availability_day_all_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# aggregate to month level
df_avail.columns = pd.MultiIndex.from_arrays([pd.to_datetime(df_avail.columns).month, df_avail.columns])
df_avail = df_avail.T.groupby(level=0).sum().T
df_avail = df_avail[[df_avail.columns[-1]] + list(df_avail.columns[:-1])]
df_avail.columns = df_avail.columns.map(month_mapping)

# plot glucose availability per month
ax = sns.heatmap(df_avail, annot=True, linewidth=.5, cmap='Greens', cbar_kws={'label': 'days'})
plt.xlabel('month')
plt.ylabel('rider')
plt.savefig(savepath+'glucose_availability_month_all.pdf', bbox_inches='tight')
plt.savefig(savepath+'glucose_availability_month_all.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(savepath+'glucose_availability_month_all_NAME.pdf', bbox_inches='tight')
plt.savefig(savepath+'glucose_availability_month_all_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# calculate percentage glucose availability per day
max_readings = 24*60/5+1
df_avail_perc = df.groupby(['RIDER', df.timestamp.dt.date])['Glucose Value (mg/dL)'].count() / max_readings
df_avail_perc = df_avail_perc.unstack()
df_avail_perc.fillna(0, inplace=True)

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(df_avail_perc, cmap='Blues', vmax=1, cbar_kws=dict(extend='max', label='fraction of possible CGM readings per day'))
fig.axes[1].collections[0].cmap.set_over('orange') 
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.savefig(savepath+'glucose_availability_day_perc_all.pdf', bbox_inches='tight')
plt.savefig(savepath+'glucose_availability_day_perc_all.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(savepath+'glucose_availability_day_perc_all_NAME.pdf', bbox_inches='tight')
plt.savefig(savepath+'glucose_availability_day_perc_all_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


def plot_barh(data, y, height=.5, colors=dict(h_neg=10, h_pos=10, s=0, l=50)):
	color_palette = sns.diverging_palette(**colors, n=5)
	left = 0
	for i, (label, x) in enumerate(data.items()):
		plt.barh(y=y, width=x, height=height, left=left, color=color_palette[i])
		left += x

# plot for all athletes, hue: stages
fig, ax = plt.subplots()
for k, sec in enumerate(sections):
	pil = get_percinlevel(df[df[sec]])
	plot_barh(pil, y=k, colors=colors[k])

plt.xticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
plt.yticks(np.arange(len(sections)), [j.rstrip() for j in sections])
plt.xlim((0,100))
plt.xlabel('CGM readings')

ax.legend(handles=legend_elements, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.01))
ax.invert_yaxis()
ax.xaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(savepath+'perc_in_zone/perc_in_glucoselevel.pdf', bbox_inches='tight')
plt.savefig(savepath+'perc_in_zone/perc_in_glucoselevel.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# stratify by athlete, hue: stages
for i in athletes:
	fig, ax = plt.subplots()
	for k, sec in enumerate(sections):
		pil = get_percinlevel(df[df[sec]][df[df[sec]].RIDER == i])
		plot_barh(pil, y=k, colors=colors[k])

	plt.xticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.yticks(np.arange(len(sections)), [j.rstrip() for j in sections])
	plt.xlim((0,100))
	plt.xlabel('CGM readings')
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$'.format(i), y=1.09)

	ax.legend(handles=legend_elements, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.01))
	ax.invert_yaxis()
	ax.xaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.savefig(savepath+'perc_in_zone/perc_in_glucoselevel_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'perc_in_zone/perc_in_glucoselevel_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$'.format(rider_mapping_inv[i]), y=1.09)
	plt.savefig(savepath+'perc_in_zone/perc_in_glucoselevel_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'perc_in_zone/perc_in_glucoselevel_NAME_%s.png'%i, dpi=300, bbox_inches='tight')

	plt.show()
	plt.close()

def half_violin(v, side='left'):
	color = {'left':color_race['normal'], 'right':color_race['race']}
	for b in v['bodies']:
		# get the center
	    m = np.mean(b.get_paths()[0].vertices[:, 0])
	    # modify the paths to not go further right than the center
	    if side == 'left':
	    	b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
	    if side == 'right':
	    	b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)

	    b.set_color(color[side])

# plot for all athletes, hue: stages
fig, ax = plt.subplots(figsize=(3, 3.5))
for k, sec in enumerate(sections):
	vl = plt.violinplot(df.loc[df[sec] & df['race']==False, col], positions=[k], showextrema=False, labels=['normal'])
	half_violin(vl, 'left')
	vr = plt.violinplot(df.loc[df[sec] & df['race']==True, col], positions=[k], showextrema=False, label='race')
	half_violin(vr, 'right')

plt.xticks(np.arange(len(sections)), sections)
plt.ylabel('Glucose Value (mg/dL)')
plt.legend()

# select only EGV measurements for glucose during the entire day
# for all the others we did this already
sec = 'day     '
df[df[sec]] = df[df[sec]][df[df[sec]]['Event Type'] == 'EGV']

for sec in sections:
	df[df[sec]]['timestamp'] = pd.to_datetime(df[df[sec]]['timestamp'])
	df[df[sec]]['local_timestamp'] = pd.to_datetime(df[df[sec]]['local_timestamp'])

	# resample glucose values to every 5 min
	df[df[sec]] = df[df[sec]].set_index('timestamp')
	if sec == 'training' or sec == 'after   ':
		df[df[sec]] = df[df[sec]].groupby(['RIDER', pd.Grouper(freq='5min')])[[col, 'tid']].mean()
	else:
		df[df[sec]] = df[df[sec]].groupby(['RIDER', pd.Grouper(freq='5min')])[col].mean()
	df[df[sec]] = df[df[sec]].reset_index()

def from_time_to_timedelta(x):
	from datetime import datetime, date
	return datetime.combine(date.min, x) - datetime.min

# plot day utc
k = 0 ; sec = 'day     '
for i in athletes:
	df_i = df[df[sec]][df[df[sec]].RIDER == i]

	# calculate time on xaxis
	df_i['date'] = df_i['timestamp'].dt.date
	df_i['time'] = df_i['timestamp'].dt.time.apply(from_time_to_timedelta)
	df_i['5min'] = df_i['time'].dt.seconds / 300

	ax = sns.lineplot(data=df_i, x='5min', y=col)

	ax.set_xticklabels(pd.to_datetime(ax.get_xticks()*300, unit='s').strftime('%H:%M'))
	plt.xlabel('Local Time (hh:mm)')

	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df_i)/(365*24*60/5)*100, 
			df_i[col].mean(),
			df_i[col].std()/df_i[col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_%s_%s.pdf'%(sec,i), bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_%s_%s.png'%(sec,i), dpi=300, bbox_inches='tight')

	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df_i)/(365*24*60/5)*100,
			df_i[col].mean(),
			df_i[col].std()/df_i[col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_%s_%s_NAME.pdf'%(sec,i), bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_%s_%s_NAME.png'%(sec,i), dpi=300, bbox_inches='tight')

	plt.show()
	plt.close()

# TODO: combine all in one figure
# TODO: now only local time

# plot training session timedelta
for i in athletes:
	df_train = df_['training'][df_['training'].RIDER == i]
	df_after = df_['after   '][df_['after   '].RIDER == i]

	last_ts = df_train.loc[df_train['tid'].diff().shift(-1) != 0, 'timestamp']
	last_ts[df_train.index[-1]] = df_train.iloc[-1]['timestamp']
	last_ts[df_train.index[-1]+1] = df_train.iloc[-1]['timestamp'] + datetime.timedelta(days=1)

	# calculate time on xaxis
	df_train['min'] = df_train.groupby('tid')['timestamp'].transform('min').dt.time.apply(from_time_to_timedelta)
	tid_to_min = df_train.groupby('tid')['timestamp'].min().dt.time.apply(from_time_to_timedelta).to_dict()
	df_after['min'] = df_after['tid'].map(tid_to_min)

	df_train['time'] = df_train['timestamp'].dt.time.apply(from_time_to_timedelta) - df_train['min']
	df_after['time'] = df_after['timestamp'].dt.time.apply(from_time_to_timedelta) - df_after['min']

	# fix time if after training time is overnight
	df_after.loc[df_after['time'] < '0s', 'time'] += pd.to_timedelta('1day')

	df_train['5min'] = df_train['time'].dt.seconds / 300
	df_after['5min'] = df_after['time'].dt.seconds / 300

	"""
	# filter out training sessions that last shorter than one hour
	short_sessions = df_train.loc[df_train['new_tid'].shift(-1).fillna(False) & (df_train['time'] < '1h'), 'tid'].to_numpy()
	df_train = df_train[~df_train.tid.isin(short_sessions)]
	df_after = df_after[~df_after.tid.isin(short_sessions)]
	"""
	
	ax = sns.lineplot(data=df_train, x='5min', y=col)
	#ax = sns.lineplot(data=df_after, x='5min', y=col)

	ax.set_xticklabels(pd.to_datetime(ax.get_xticks()*300, unit='s').strftime('%H:%M'))
	plt.xlabel('Time in Training Session (hh:mm)')

	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df[df.RIDER == i][col])/(365*24*60/5)*100, 
			df[df.RIDER == i][col].mean(),
			df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_training_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_training_%s.png'%i, dpi=300, bbox_inches='tight')

	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df[df.RIDER == i][col])/(365*24*60/5)*100,
			df[df.RIDER == i][col].mean(),
			df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_training_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_training_%s_NAME.png'%i, dpi=300, bbox_inches='tight')

	plt.show()
	plt.close()


for k, sec in enumerate(sections):
	df_i = df[df[sec]]

	df_i['date'] = df_i['local_timestamp'].dt.date
	df_i['time'] = df_i['local_timestamp'].dt.time

	sns.lineplot(data=df_i, x='time', y=col)
	plt.xlabel('Local time (hh:mm:ss)')
	plt.ylabel(col)

	for i in athletes:
		df_i = df[df[sec]][df[df[sec]].RIDER == i]

		df_i['date'] = df_i['local_timestamp'].dt.date
		df_i['time'] = df_i['local_timestamp'].dt.time.apply(from_time_to_timedelta)
		df_i['sec'] = df_i['time'].dt.seconds

		idx = pd.MultiIndex.from_arrays([df_i['date'], df_i['sec']])
		df_i.set_index(idx)

		sns.lineplot(data=df_i, x='sec', y=col)
		plt.xlabel('Local time (hh:mm:ss)')
		plt.ylabel(col)
		plt.show()


"""
	# find out where the glucose belongs

	matplotlib.use('TkAgg')
	df['date'] = df.timestamp.dt.date
	for d in df['date'].unique():
		fig, ax = plt.subplots(figsize=(15,6))
		mask = (df_dc.RIDER == i) & (df_dc['timestamp'].dt.date == d)
		sns.lineplot(df_dc.loc[mask, 'timestamp'], df_dc.loc[mask, 'Glucose Value (mg/dL)'], 
			ax=ax, marker='o', dashes=False, label='Dexcom', alpha=.8)

		sns.lineplot(df.loc[df['date'] == d, 'timestamp'],
				df.loc[df['date'] == d, 'glucose'], 
				ax=ax, marker='o', dashes=False, label='TP', alpha=.8)

		ax.set_xticklabels(pd.to_datetime(ax.get_xticks(), unit='D').strftime('%H:%M'))
		plt.xlabel('UTC (hh:mm)')

		plt.title("RIDER %s - %s"%(i, d.strftime('%d-%m-%Y')))
		plt.savefig(path+'glucose/glucose_%s_%s.pdf'%(i,d.strftime('%Y%m%d')), bbox_inches='tight')
		plt.savefig(path+'glucose/glucose_%s_%s.png'%(i,d.strftime('%Y%m%d')), dpi=300, bbox_inches='tight')
		
		plt.title("%s - %s"%(rider_mapping_inv[i], d.strftime('%d-%m-%Y')))
		plt.savefig(path+'glucose/glucose_%s_%s_NAME.pdf'%(i,d.strftime('%Y%m%d')), bbox_inches='tight')
		plt.savefig(path+'glucose/glucose_%s_%s_NAME.png'%(i,d.strftime('%Y%m%d')), dpi=300, bbox_inches='tight')

		plt.close()
"""