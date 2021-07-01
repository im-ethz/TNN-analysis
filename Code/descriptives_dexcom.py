import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *
from calc import *

from config import rider_mapping

path = 'Data/Dexcom/'
savepath = 'Descriptives/dexcom/'

# -------------------------- Read data

rider_mapping_inv = {v:k for k, v in rider_mapping.items()}

df_glucose = pd.read_csv(path+'dexcom_clean.csv', index_col=0)
df_glucose.drop('local_timestamp_raw', axis=1, inplace=True)

df_glucose['timestamp'] = pd.to_datetime(df_glucose['timestamp'])
df_glucose['local_timestamp'] = pd.to_datetime(df_glucose['local_timestamp'])

athletes = df_glucose.RIDER.unique()

# select glucose measurements
df_glucose = df_glucose[df_glucose['Event Type'] == 'EGV']

# glucose divided in day sections
df_glucose_ = { 'day     '	: pd.read_csv(path+'dexcom_clean.csv', index_col=0),
				'wake    '	: pd.read_csv(path+'dexcom_clean_wake.csv', index_col=0),
				'training'	: pd.read_csv(path+'dexcom_clean_training.csv', index_col=0),
				'after   '	: pd.read_csv(path+'dexcom_clean_after.csv', index_col=0),
				'sleep   '	: pd.read_csv(path+'dexcom_clean_sleep.csv', index_col=0)}

# -------------------------- Glucose availability (1)
# create calendar with glucose availability
dates_2019 = pd.date_range(start='12/1/2018', end='11/30/2019').date
glucose_avail = pd.DataFrame(index=df_glucose.RIDER.unique(), columns=dates_2019)
for i in df_glucose.RIDER.unique():
	glucose_avail.loc[i, df_glucose[df_glucose.RIDER == i].timestamp.dt.date.unique()] = 1
glucose_avail.fillna(0, inplace=True)

# plot glucose availability per day
plt.figure(figsize=(25,10))
ax = sns.heatmap(glucose_avail, cmap='Blues', cbar=False, linewidths=.01) 
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
glucose_avail.columns = pd.MultiIndex.from_arrays([pd.to_datetime(glucose_avail.columns).month, glucose_avail.columns])
glucose_avail = glucose_avail.T.groupby(level=0).sum().T
glucose_avail = glucose_avail[[glucose_avail.columns[-1]] + list(glucose_avail.columns[:-1])]
glucose_avail.columns = glucose_avail.columns.map(month_mapping)

# plot glucose availability per month
ax = sns.heatmap(glucose_avail, annot=True, linewidth=.5, cmap='Greens', cbar_kws={'label': 'days'})
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
max_readings = 24*60/5
glucose_avail_perc = df_glucose.groupby(['RIDER', df_glucose.timestamp.dt.date])['Glucose Value (mg/dL)'].count() / max_readings
glucose_avail_perc = glucose_avail_perc.unstack()
glucose_avail_perc.fillna(0, inplace=True)

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(glucose_avail_perc, cmap='Blues', vmax=1, cbar_kws=dict(extend='max', label='fraction of possible CGM readings per day'))
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

# TODO: 70% selection?

# -------------------------- Glucose distributions
def plot_hist_glucose_settings(ax, ax0, col, xlim=(20,410), ylabel='Probability', loc_legend=(0.98, 0.93)):
	ax.set_xlim((20, 410))
	ax.set_xlabel(col)
	ax.set_ylabel(ylabel)
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_label_position('left')
	ax0.yaxis.set_visible(False)
	ax0.set_ylabel('')
	plt.legend(loc='upper right', bbox_to_anchor=loc_legend, prop={'family': 'DejaVu Sans Mono'})

col = 'Glucose Value (mg/dL)'

# plot all glucose data that we have
fig, ax0 = plt.subplots()
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
sns.kdeplot(df_glucose[col], ax=ax, linewidth=2,
	label=r'all ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
	.format(df_glucose[col].mean(), df_glucose[col].std()/df_glucose[col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
plt.savefig(savepath+'hist_glucose.pdf', bbox_inches='tight')
plt.savefig(savepath+'hist_glucose.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot hue stages
type_palette = sns.color_palette("Set1")

fig, ax0 = plt.subplots()
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for k, sec in enumerate(df_glucose_.keys()):
	sns.kdeplot(df_glucose_[sec][col], ax=ax, linewidth=2, color=type_palette[k],
		label=sec+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'.format(df_glucose_[sec][col].mean(), df_glucose_[sec][col].std()/df_glucose_[sec][col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
plt.savefig(savepath+'hist_glucose_sec.pdf', bbox_inches='tight')
plt.savefig(savepath+'hist_glucose_sec.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot for athletes individually, hue: stages
for i in df_glucose.RIDER.unique():
	type_palette = sns.color_palette("Set1")

	fig, ax0 = plt.subplots()
	ax0 = plot_glucose_levels(ax0, color=False)
	ax = ax0.twinx()
	
	for k, sec in enumerate(df_glucose_.keys()):
		sns.kdeplot(df_glucose_[sec][df_glucose_[sec].RIDER == i][col], ax=ax, linewidth=2, color=type_palette[k],
			label=sec+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
			.format(df_glucose_[sec][df_glucose_[sec].RIDER == i][col].mean(), 
					df_glucose_[sec][df_glucose_[sec].RIDER == i][col].std()/df_glucose_[sec][df_glucose_[sec].RIDER == i][col].mean()*100))
	plot_hist_glucose_settings(ax, ax0, col)
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df_glucose[df_glucose.RIDER == i][col])/(365*24*60/5)*100,
				df_glucose[df_glucose.RIDER == i][col].mean(),
				df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'hist_glucose_sec_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'hist_glucose_sec_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df_glucose[df_glucose.RIDER == i][col])/(365*24*60/5)*100,
				df_glucose[df_glucose.RIDER == i][col].mean(),
				df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'hist_glucose_sec_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'hist_glucose_sec_NAME_%s.png'%i, dpi=300, bbox_inches='tight')	
	plt.show()
	plt.close()

# plot hue athletes
type_palette = sns.color_palette("viridis_r", n_colors=11)

fig, ax0 = plt.subplots()
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for c, i in enumerate(athletes):
	sns.kdeplot(df_glucose[df_glucose.RIDER == i][col], ax=ax, 
		linewidth=1.5, color=type_palette[c], alpha=.8,
		label=str(i)+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
		.format(df_glucose[df_glucose.RIDER == i][col].mean(), 
				df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
plt.savefig(savepath+'hist_glucose_riders.pdf', bbox_inches='tight')
plt.savefig(savepath+'hist_glucose_riders.png', dpi=300, bbox_inches='tight')
for c, i in enumerate(athletes):
	text = ax.get_legend().get_texts()[c].get_text().split()
	ax.get_legend().get_texts()[c].set_text(rider_mapping_inv[int(text[0])]+' '+' '.join(text[1:]))
plt.savefig(savepath+'hist_glucose_riders_NAME.pdf', bbox_inches='tight')
plt.savefig(savepath+'hist_glucose_riders_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot for stages individually, hue: athletes
for k, sec in enumerate(df_glucose_.keys()):
	type_palette = {'day     '	: sns.color_palette('Reds', n_colors=11),
					'wake    '	: sns.color_palette('Blues', n_colors=11),
					'training'	: sns.color_palette('Greens', n_colors=11),
					'after   '	: sns.color_palette('Purples', n_colors=11),
					'sleep   '	: sns.color_palette('Oranges', n_colors=11)}

	fig, ax0 = plt.subplots()
	ax0 = plot_glucose_levels(ax0, color=False)
	ax = ax0.twinx()
	for c, i in enumerate(athletes):
		sns.kdeplot(df_glucose_[sec][df_glucose_[sec].RIDER == i][col], ax=ax, 
			linewidth=1.5, color=type_palette[sec][c], alpha=.8,
			label=str(i)+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
			.format(df_glucose_[sec][df_glucose_[sec].RIDER == i][col].mean(), 
					df_glucose_[sec][df_glucose_[sec].RIDER == i][col].std()/df_glucose_[sec][df_glucose_[sec].RIDER == i][col].mean()*100))
	plot_hist_glucose_settings(ax, ax0, col)
	plt.title(sec.rstrip(), y=1.06)
	plt.savefig(savepath+'hist_glucose_%s.pdf'%sec.rstrip(), bbox_inches='tight')
	plt.savefig(savepath+'hist_glucose_%s.png'%sec.rstrip(), dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# -------------------------- Glucose barcharts
# TODO: either this or impute and then time spent in hypo (kind of depends if there are less measurements when someone is in hypo)

def get_percinlevel(df, col='Glucose Value (mg/dL)'):
	return {level: ((df[col] >= lmin) & (df[col] <= lmax)).sum() / len(df)*100 for level, (lmin, lmax) in glucose_levels.items()}

def plot_barh(data, y, height=.5, colors=dict(h_neg=10, h_pos=10, s=0, l=50)):
	color_palette = sns.diverging_palette(**colors, n=5)
	left = 0
	for i, (label, x) in enumerate(data.items()):
		plt.barh(y=y, width=x, height=height, left=left, color=color_palette[i])
		left += x

colors = [dict(zip(['h_neg', 'h_pos', 'l', 's'], [c[0]*360, c[0]*360, c[1]*100, c[2]*100])) \
			for c in [rgb_to_hls(*j) for j in sns.color_palette("Set1")]]

legend_elements = [Patch(facecolor=c, edgecolor='white', label=l) \
					for c, l in zip(sns.diverging_palette(10, 10, s=0, n=5), glucose_levels.keys())] 

# plot for all athletes, hue: stages
fig, ax = plt.subplots()
for k, sec in enumerate(df_glucose_.keys()):
	pil = get_percinlevel(df_glucose_[sec])
	plot_barh(pil, y=k, colors=colors[k])

plt.xticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
plt.yticks(np.arange(len(df_glucose_.keys())), [j.rstrip() for j in df_glucose_.keys()])
plt.xlim((0,100))
plt.xlabel('CGM readings')

ax.legend(handles=legend_elements, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.01))
ax.invert_yaxis()
ax.xaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(savepath+'timeperc_in_glucoselevel.pdf', bbox_inches='tight')
plt.savefig(savepath+'timeperc_in_glucoselevel.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# stratify by athlete, hue: stages
for i in athletes:
	fig, ax = plt.subplots()
	for k, sec in enumerate(df_glucose_.keys()):
		pil = get_percinlevel(df_glucose_[sec][df_glucose_[sec].RIDER == i])
		plot_barh(pil, y=k, colors=colors[k])

	plt.xticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.yticks(np.arange(len(df_glucose_.keys())), [j.rstrip() for j in df_glucose_.keys()])
	plt.xlim((0,100))
	plt.xlabel('CGM readings')
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$'.format(i), y=1.09)

	ax.legend(handles=legend_elements, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.01))
	ax.invert_yaxis()
	ax.xaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.savefig(savepath+'timeperc_in_glucoselevel_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'timeperc_in_glucoselevel_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$'.format(rider_mapping_inv[i]), y=1.09)
	plt.savefig(savepath+'timeperc_in_glucoselevel_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'timeperc_in_glucoselevel_NAME_%s.png'%i, dpi=300, bbox_inches='tight')

	plt.show()
	plt.close()

# -------------------------- Glucose longitudinal
# remove number 12 for now
athletes = athletes[athletes != 12]

col = 'Glucose Value (mg/dL)'

def from_time_to_timedelta(x):
	from datetime import datetime, date
	return datetime.combine(date.min, x) - datetime.min

# select only EGV measurements for glucose during the entire day
# for all the others we did this already
sec = 'day     '
df_glucose_[sec] = df_glucose_[sec][df_glucose_[sec]['Event Type'] == 'EGV']

for sec in df_glucose_.keys():
	df_glucose_[sec]['timestamp'] = pd.to_datetime(df_glucose_[sec]['timestamp'])
	df_glucose_[sec]['local_timestamp'] = pd.to_datetime(df_glucose_[sec]['local_timestamp'])

	# resample glucose values to every 5 min
	df_glucose_[sec] = df_glucose_[sec].set_index('timestamp')
	if sec == 'training' or sec == 'after   ':
		df_glucose_[sec] = df_glucose_[sec].groupby(['RIDER', pd.Grouper(freq='5min')])[[col, 'tid']].mean()
	else:
		df_glucose_[sec] = df_glucose_[sec].groupby(['RIDER', pd.Grouper(freq='5min')])[col].mean()
	df_glucose_[sec] = df_glucose_[sec].reset_index()

# plot day utc
k = 0 ; sec = 'day     '
for i in athletes:
	df_i = df_glucose_[sec][df_glucose_[sec].RIDER == i]

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
	plt.savefig(savepath+'glucose_utctime_%s_%s.pdf'%(sec.rstrip(),i), bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_%s_%s.png'%(sec.rstrip(),i), dpi=300, bbox_inches='tight')

	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df_i)/(365*24*60/5)*100,
			df_i[col].mean(),
			df_i[col].std()/df_i[col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_%s_%s_NAME.pdf'%(sec.rstrip(),i), bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_%s_%s_NAME.png'%(sec.rstrip(),i), dpi=300, bbox_inches='tight')

	plt.show()
	plt.close()

# TODO: combine all in one figure
# TODO: now only local time

# plot training session timedelta
for i in athletes:
	df_train = df_glucose_['training'][df_glucose_['training'].RIDER == i]
	df_after = df_glucose_['after   '][df_glucose_['after   '].RIDER == i]

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
		.format(i, len(df_glucose[df_glucose.RIDER == i][col])/(365*24*60/5)*100, 
			df_glucose[df_glucose.RIDER == i][col].mean(),
			df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_training_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_training_%s.png'%i, dpi=300, bbox_inches='tight')

	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df_glucose[df_glucose.RIDER == i][col])/(365*24*60/5)*100,
			df_glucose[df_glucose.RIDER == i][col].mean(),
			df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'glucose_utctime_training_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'glucose_utctime_training_%s_NAME.png'%i, dpi=300, bbox_inches='tight')

	plt.show()
	plt.close()


for k, sec in enumerate(df_glucose_.keys()):
	df_i = df_glucose_[sec]

	df_i['date'] = df_i['local_timestamp'].dt.date
	df_i['time'] = df_i['local_timestamp'].dt.time

	sns.lineplot(data=df_i, x='time', y=col)
	plt.xlabel('Local time (hh:mm:ss)')
	plt.ylabel(col)

	for i in athletes:
		df_i = df_glucose_[sec][df_glucose_[sec].RIDER == i]

		df_i['date'] = df_i['local_timestamp'].dt.date
		df_i['time'] = df_i['local_timestamp'].dt.time.apply(from_time_to_timedelta)
		df_i['sec'] = df_i['time'].dt.seconds

		idx = pd.MultiIndex.from_arrays([df_i['date'], df_i['sec']])
		df_i.set_index(idx)

		sns.lineplot(data=df_i, x='sec', y=col)
		plt.xlabel('Local time (hh:mm:ss)')
		plt.ylabel(col)
		plt.show()