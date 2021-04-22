import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *
from calc import *

from config import rider_mapping

path = 'Data/Dexcom/'
path_trainingpeaks = 'Data/TrainingPeaks/2019/clean2/'
savepath = 'Descriptives/'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path_trainingpeaks)])
rider_mapping_inv = {v:k for k, v in rider_mapping.items()}

df_glucose = pd.read_csv(path+'dexcom_clean.csv', index_col=0)
df_glucose['local_timestamp'] = pd.to_datetime(df_glucose['local_timestamp'])

# select glucose measurements
df_glucose = df_glucose[df_glucose['Event Type'] == 'EGV']

# -------------------------- Glucose availability (1)
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

glucose_avail_perc = glucose_count['local_timestamp'].unstack() / max_readings
glucose_avail_perc.fillna(0, inplace=True)

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
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

# divide glucose in sections
df_glucose_ = { 'day     '	: pd.read_csv('Data/Dexcom/dexcom_clean_day.csv', index_col=0),
				'wake    '	: pd.read_csv('Data/Dexcom/dexcom_clean_wake.csv', index_col=0),
				'training'	: pd.read_csv('Data/Dexcom/dexcom_clean_training.csv', index_col=0),
				'after   '	: pd.read_csv('Data/Dexcom/dexcom_clean_after.csv', index_col=0),
				'sleep   '	: pd.read_csv('Data/Dexcom/dexcom_clean_sleep.csv', index_col=0)}

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
	plt.show()
	plt.close()

# not ANON
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
plt.show()
plt.close()

# not ANON
type_palette = sns.color_palette("viridis_r", n_colors=11)

fig, ax0 = plt.subplots()
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for c, i in enumerate(athletes):
	sns.kdeplot(df_glucose[df_glucose.RIDER == i][col], ax=ax, 
		linewidth=1.5, color=type_palette[c], alpha=.8,
		label=rider_mapping_inv[i]+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
		.format(df_glucose[df_glucose.RIDER == i][col].mean(), 
				df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
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
	return = {level: ((df[col] >= lmin) & (df[col] <= lmax)).sum() / len(df)*100 for level, (lmin, lmax) in glucose_levels.items()}

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
	plt.show()
	plt.close()

# not ANON
for i in athletes:
	fig, ax = plt.subplots()
	for k, sec in enumerate(df_glucose_.keys()):
		pil = get_percinlevel(df_glucose_[sec][df_glucose_[sec].RIDER == i])
		plot_barh(pil, y=k, colors=colors[k])

	plt.xticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.yticks(np.arange(len(df_glucose_.keys())), [j.rstrip() for j in df_glucose_.keys()])
	plt.xlim((0,100))
	plt.xlabel('CGM readings')
	plt.title(r'$\bf{:s}$'.format(rider_mapping_inv[i]), y=1.09)

	ax.legend(handles=legend_elements, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.01))
	ax.invert_yaxis()
	ax.xaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.savefig(savepath+'timeperc_in_glucoselevel_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'timeperc_in_glucoselevel_NAME_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()