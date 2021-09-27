# TODO: figures in local time
# TODO: race and training days
# TODO: amoung of hypo and hyper during each part of the day
# Can also make a flat distribution diagram for this

import numpy as np
import pandas as pd
import datetime
import os
from copy import copy

from plot import *
from helper import month_mapping, month_firstday
from calc import glucose_levels, mmoll_mgdl, mgdl_mmoll
from config import DATA_PATH as RAW_PATH

from secret import rider_mapping
rider_mapping_inv = {v:k for k, v in rider_mapping.items()}

DATA_PATH = 'data/'
SAVE_PATH = 'descriptives/dexcom/'

if not os.path.exists(SAVE_PATH):
	os.mkdir(SAVE_PATH)
if not os.path.exists(SAVE_PATH+'availability/'):
	os.mkdir(SAVE_PATH+'availability/')
if not os.path.exists(SAVE_PATH+'hist/'):
	os.mkdir(SAVE_PATH+'hist/')
if not os.path.exists(SAVE_PATH+'boxplot/'):
	os.mkdir(SAVE_PATH+'boxplot/')
if not os.path.exists(SAVE_PATH+'time_in_zone/'):
	os.mkdir(SAVE_PATH+'time_in_zone/')
if not os.path.exists(SAVE_PATH+'time_training/'):
	os.mkdir(SAVE_PATH+'time_training/')

# -------------------------- Read data
df = pd.read_csv(RAW_PATH+'dexcom.csv', index_col=0)
df.drop('local_timestamp_raw', axis=1, inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

athletes = df.RIDER.unique()

# select glucose measurements
df = df[df['Event Type'] == 'EGV']
df = df[['RIDER', 'timestamp', 'local_timestamp', 'Source Device ID', 'Glucose Value (mg/dL)',
		 'Transmitter ID', 'Transmitter Time (Long Integer)', 'source', 'training']]

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

# -------------------------- Read CGM data : resampled + completeness selection + incl sections
df = pd.read_csv(DATA_PATH+'dexcom_resampled_selectcompleteness.csv', index_col=0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

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

# -------------------------- Glucose distributions
col = 'Glucose Value (mg/dL)'
sections = ('wake', 'train', 'after', 'sleep')
for sec in sections:
	df[sec] = df[sec].astype(bool)
# TODO: so far no exclusions from different sections

# color palettes
color_palette = sns.color_palette("Set1")
color_sec = {'wake'	: color_palette[1],
			 'train': color_palette[4],
			 'after': color_palette[2],
			 'sleep': color_palette[3]}
palette_sec = {	'wake'	: sns.color_palette('Blues', n_colors=11),
				'train'	: sns.color_palette('Oranges', n_colors=11),
				'after'	: sns.color_palette('Greens', n_colors=11),
				'sleep'	: sns.color_palette('Purples', n_colors=11)}
palette_ath = sns.color_palette("tab10")+[(0.106, 0.62, 0.467)]#sns.color_palette("viridis_r", n_colors=11)
color_race = {'normal':color_palette[8], 'race':(0.8455062527192158, 0.21363575247920147, 0.4145075850498335)}

def plot_hist_glucose_settings(ax, ax0, col='Glucose Value (mg/dL)', xlim=(20,410), ylabel='Probability', loc_legend=(1., 0.96)):
	ax.set_xlim((20, 410))
	ax.set_xlabel(col)
	ax.xaxis.set_visible(True)
	ax.set_ylabel(ylabel)
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_label_position('left')
	ax0.yaxis.set_visible(False)
	ax0.set_ylabel('')
	plt.legend(loc='upper right', bbox_to_anchor=loc_legend, prop={'family': 'DejaVu Sans Mono', 'size':8})

# plot all glucose data that we have
fig, ax0 = plt.subplots(figsize=(5, 3.5))
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
sns.kdeplot(df[col], ax=ax, linewidth=2,
	label=r'all ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
	.format(df[col].mean(), df[col].std()/df[col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
ax.set_xlabel(col)
plt.savefig(SAVE_PATH+'hist/hist_glucose.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'hist/hist_glucose.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot hue stages
fig, ax0 = plt.subplots(figsize=(5, 3.5))
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for k, sec in enumerate(sections):
	sns.kdeplot(df[df[sec]][col], ax=ax, linewidth=2, color=color_sec[sec],
		label=sec+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'.format(df[df[sec]][col].mean(), df[df[sec]][col].std()/df[df[sec]][col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
plt.savefig(SAVE_PATH+'hist/hist_glucose_sec.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'hist/hist_glucose_sec.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot for athletes individually, hue: stages
for i in df.RIDER.unique():
	fig, ax0 = plt.subplots(figsize=(5, 3.5))
	ax0 = plot_glucose_levels(ax0, color=False)
	ax = ax0.twinx()
	
	for k, sec in enumerate(sections):
		sns.kdeplot(df[df[sec]][df[df[sec]].RIDER == i][col], ax=ax, linewidth=2, color=color_sec[sec],
			label=sec+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
			.format(df[df[sec]][df[df[sec]].RIDER == i][col].mean(), 
					df[df[sec]][df[df[sec]].RIDER == i][col].std()/df[df[sec]][df[df[sec]].RIDER == i][col].mean()*100))
	plot_hist_glucose_settings(ax, ax0, col)
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_NAME_%s.png'%i, dpi=300, bbox_inches='tight')	
	plt.show()
	plt.close()

# plot hue athletes
fig, ax0 = plt.subplots(figsize=(5, 3.5))
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for c, i in enumerate(athletes):
	sns.kdeplot(df[df.RIDER == i][col], ax=ax, 
		linewidth=1.5, color=palette_ath[c], alpha=.8,
		label=str(i)+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
		.format(df[df.RIDER == i][col].mean(), 
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
for text in ax.get_legend().get_texts(): # smaller fontsize
	text.set_fontsize(6)
plt.savefig(SAVE_PATH+'hist/hist_glucose_riders.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'hist/hist_glucose_riders.png', dpi=300, bbox_inches='tight')
for c, i in enumerate(athletes):
	text = ax.get_legend().get_texts()[c].get_text().split()
	ax.get_legend().get_texts()[c].set_text(rider_mapping_inv[int(text[0])]+' '+' '.join(text[1:]))
plt.savefig(SAVE_PATH+'hist/hist_glucose_riders_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'hist/hist_glucose_riders_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# plot for sections individually, hue: athletes
for k, sec in enumerate(sections):
	fig, ax0 = plt.subplots()
	ax0 = plot_glucose_levels(ax0, color=False)
	ax = ax0.twinx()
	for c, i in enumerate(athletes):
		sns.kdeplot(df[df[sec]][df[df[sec]].RIDER == i][col], ax=ax, 
			linewidth=1.5, color=palette_sec[sec][c], alpha=.8,
			label=str(i)+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
			.format(df[df[sec]][df[df[sec]].RIDER == i][col].mean(), 
					df[df[sec]][df[df[sec]].RIDER == i][col].std()/df[df[sec]][df[df[sec]].RIDER == i][col].mean()*100))
	plot_hist_glucose_settings(ax, ax0, col)
	plt.title(sec, y=1.06)
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_%s.pdf'%sec, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_%s.png'%sec, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# -------------------------- Glucose barcharts
def get_percinlevel(df, col='Glucose Value (mg/dL)'):
	return {level: ((df[col] >= lmin) & (df[col] <= lmax)).sum() / len(df)*100 for level, (lmin, lmax) in glucose_levels.items()}

def plot_bar(data, x, width=.8, colors=dict(h_neg=10, h_pos=10, s=0, l=50)):
	hatch = ('\\\\', '\\\\', None, '//', '//')
	color_palette = sns.diverging_palette(**colors, n=5)
	bottom = 0
	for i, (label, y) in enumerate(data.items()):
		plt.bar(x=x, height=y, width=width, bottom=bottom, color=color_palette[i], hatch=hatch[i])
		bottom += y

legend_elements = [Patch(facecolor=c, edgecolor='white', hatch=h, label=l) \
					for c, l, h in zip(sns.diverging_palette(10, 10, s=0, n=5), 
									   list(glucose_levels.keys())[::-1], 
									   ('///', '///', None, '\\\\\\', '\\\\\\'))] 

colors = [dict(zip(['h_neg', 'h_pos', 'l', 's'], [c[0]*360, c[0]*360, c[1]*100, c[2]*100])) \
			for c in [rgb_to_hls(*j) for j in color_sec.values()]]

# plot for all athletes, hue: stages
fig, ax = plt.subplots(figsize=(3, 3.5))
for k, sec in enumerate(sections):
	pil = get_percinlevel(df[df[sec]])
	plot_bar(pil, x=k, colors=colors[k])

plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
plt.xticks(np.arange(len(sections)), sections)
plt.ylim((0,100))
plt.ylabel('Time in target glucose values (%)')

ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
ax.yaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# stratify by athlete, hue: stages
for i in athletes:
	fig, ax = plt.subplots(figsize=(3, 3.5))
	for k, sec in enumerate(sections):
		pil = get_percinlevel(df[df[sec]][df[df[sec]].RIDER == i])
		plot_bar(pil, x=k, colors=colors[k])

	plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.xticks(np.arange(len(sections)), sections)
	plt.ylim((0,100))
	plt.ylabel('Time in target glucose values (%)')

	ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
	ax.yaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

colors = [dict(zip(['h_neg', 'h_pos', 'l', 's'], [c[0]*360, c[0]*360, c[1]*100, c[2]*100])) \
			for c in [rgb_to_hls(*j) for j in palette_ath]]

# hue: athletes
fig, ax = plt.subplots(figsize=(8,4))
for n, i in enumerate(athletes):
	pil = get_percinlevel(df[df.RIDER == i])
	plot_bar(pil, x=n, width=.7, colors=colors[n])

plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
plt.xticks(np.arange(len(athletes)), [j for j in athletes])
plt.ylim((0,100))
plt.ylabel('Time in target glucose values (%)')

ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.2, 0.5))
ax.yaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_riders.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_riders.png', dpi=300, bbox_inches='tight')
ax.set_xticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_xticklabels()], rotation=90)
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_riders_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_riders_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# race vs no race
colors = [dict(zip(['h_neg', 'h_pos', 'l', 's'], [c[0]*360, c[0]*360, c[1]*100, c[2]*100])) \
			for c in [rgb_to_hls(*j) for j in color_race.values()]]

# plot for all athletes, hue: race
fig, ax = plt.subplots(figsize=(1,4))
for k, b in enumerate([False, True]):
	pil = get_percinlevel(df[df['race'] == b])
	plot_bar(pil, x=k, colors=colors[k])

plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
plt.xticks(np.arange(2), [j for j in color_race.keys()])
plt.ylim((0,100))
plt.ylabel('Time in target glucose values (%)')

ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
ax.yaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# stratify by athlete, hue: race
for i in athletes:
	fig, ax = plt.subplots(figsize=(1,4))
	for k, b in enumerate([False, True]):
		pil = get_percinlevel(df[df['race'] == b])
		plot_bar(pil, x=k, colors=colors[k])

	plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.xticks(np.arange(2), [j for j in color_race.keys()])
	plt.ylim((0,100))
	plt.ylabel('Time in target glucose values (%)')

	ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
	ax.yaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# -------------------------- Glucose boxplots
# x: different sections, hue: race or no race
rc = color_race['race'] ; nc = color_race['normal']

kws_box = {'race':	dict(boxprops=dict(color='w', facecolor=rc, hatch='\\\\\\'),
						 medianprops=dict(color='w', linewidth=2),
						 whiskerprops=dict(color=rc), capprops=dict(color=rc)),
		   'normal':dict(boxprops=dict(color='w', facecolor=nc, hatch='///'),
						 medianprops=dict(color='w', linewidth=2),
						 whiskerprops=dict(color=nc), capprops=dict(color=nc))}

n_nc = len(df.loc[df['race'] == False, 'local_timestamp'].dt.date.unique())
n_rc = len(df.loc[df['race'] == True, 'local_timestamp'].dt.date.unique())

fig, ax = plt.subplots(figsize=(5,3.5))
for k, sec in enumerate(sections):
	plt.boxplot(df.loc[df[sec] & df['race']==False, col], positions=[k+0.5+k*2], widths=[0.8],
		patch_artist=True, showfliers=False, **kws_box['normal'])
	plt.boxplot(df.loc[df[sec] & df['race']==True, col], positions=[k+1.5+k*2], widths=[0.8],
		patch_artist=True, showfliers=False, **kws_box['race'])
plt.xticks([1,4,7,10], sections)
plt.ylabel(col)
plt.legend(handles=[Patch(facecolor=nc, edgecolor='white', hatch='///', label='normal'+r' ($n = {:.0f}$)'.format(n_nc)),
					Patch(facecolor=rc, edgecolor='white', hatch='\\\\\\', label='race'r' ($n = {:.0f}$)'.format(n_rc))], 
			loc='upper right')
plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

for i in athletes:
	n_nc = len(df.loc[(df.RIDER == i) & (df['race'] == False), 'local_timestamp'].dt.date.unique())
	n_rc = len(df.loc[(df.RIDER == i) & (df['race'] == True), 'local_timestamp'].dt.date.unique())

	fig, ax = plt.subplots(figsize=(5,3.5))
	for k, sec in enumerate(sections):
		plt.boxplot(df.loc[(df.RIDER == i) & df[sec] & df['race']==False, col], positions=[k+0.5+k*2], widths=[0.8],
			patch_artist=True, showfliers=False, **kws_box['normal'])
		plt.boxplot(df.loc[(df.RIDER == i) & df[sec] & df['race']==True, col], positions=[k+1.5+k*2], widths=[0.8],
			patch_artist=True, showfliers=False, **kws_box['race'])
	plt.xticks([1,4,7,10], sections)
	plt.ylabel(col)
	plt.legend(handles=[Patch(facecolor=nc, edgecolor='white', hatch='///', label='normal'+r' ($n = {:.0f}$)'.format(n_nc)),
						Patch(facecolor=rc, edgecolor='white', hatch='\\\\\\', label='race'r' ($n = {:.0f}$)'.format(n_rc))], 
				loc='upper right')
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# -------------------------- Glucose longitudinal
# create TID and t
for j, ((i,n), (t_min, t_max)) in enumerate(df_training[['local_timestamp_min_r', 'local_timestamp_max_r']].iterrows()):
	mask = (df.RIDER == i) & (df.local_timestamp >= t_min) & (df.local_timestamp <= t_max)
	df.loc[mask, 'tid'] = j
	df.loc[mask, 't'] = df.loc[mask, 'local_timestamp'] - t_min

df['t'] = df['t'].dt.seconds / 60 # time in minutes

n_nc = len(df.loc[df['race'] == False, 'local_timestamp'].dt.date.unique())
n_rc = len(df.loc[df['race'] == True, 'local_timestamp'].dt.date.unique())

t_max = df['t'].max()

fig, ax = plt.subplots(1,2, figsize=(5, 3), sharey=True, gridspec_kw={'width_ratios': [5, 1], 'wspace':0}, tight_layout=True)
sns.lineplot(df['t'], df[col], hue=df['race'], palette=list(color_race.values()), ax=ax[0])
ax[0].set_xlabel('Time in training session (min)')
ax[0].legend(loc='upper right', labels=['normal'+r' ($n = {:.0f}$)'.format(n_nc), 
										'race'r' ($n = {:.0f}$)'.format(n_rc)])
sns.kdeplot(data=df.loc[df['t'].notna()], y=col, hue='race', ax=ax[1],
	fill=True, alpha=.5, palette=list(color_race.values()), linewidth=0, legend=False)
sns.despine(bottom=True, right=True, top=True, ax=ax[1])
ax[1].set(xlabel=None, xticks=[])
plt.ylim((0,350))
ax[0].set_xlim((0, t_max))
plt.savefig(SAVE_PATH+'time_training/glucose_training.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_training/glucose_training.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

for i in athletes:
	n_nc = len(df.loc[(df.RIDER == i) & (df['race'] == False), 'local_timestamp'].dt.date.unique())
	n_rc = len(df.loc[(df.RIDER == i) & (df['race'] == True), 'local_timestamp'].dt.date.unique())

	t_max = df.loc[df.RIDER == i, 't'].max()

	fig, ax = plt.subplots(1,2, figsize=(5, 3), sharey=True, gridspec_kw={'width_ratios': [5, 1], 'wspace':0}, tight_layout=True)
	sns.lineplot(df['t'], df[col], hue=df['race'], palette=list(color_race.values()), ax=ax[0])
	ax[0].set_xlabel('Time in training session (min)')
	ax[0].legend(loc='upper right', labels=['normal'+r' ($n = {:.0f}$)'.format(n_nc), 
											'race'r' ($n = {:.0f}$)'.format(n_rc)])
	sns.kdeplot(data=df.loc[df['t'].notna()], y=col, hue='race', ax=ax[1],
		fill=True, alpha=.5, palette=list(color_race.values()), linewidth=0, legend=False)
	sns.despine(bottom=True, right=True, top=True, ax=ax[1])
	ax[1].set(xlabel=None, xticks=[])
	plt.ylim((0,350))
	ax[0].set_xlim((0, t_max))
	ax[0].set_title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(i, len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s.png'%i, dpi=300, bbox_inches='tight')
	ax[0].set_title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df[df.RIDER == i][col])/(365*24*60/5)*100,
				df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100))
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# -------------------------- Descriptives
# TODO: age (years), gender (m/f), bmi (kg m-2), fat mass (%), diabetes duration (years), HbA1c (mmol mol-1 or %)
# CGM completeness
# CSII / MDI (1/10)
# VO2max (mL min-1 kg-1)
# IPAQ score (MET min week-1)
# Mean km/year travelled (cycling) in daily life
# TODO: elevation gain up/down
fitness = pd.read_csv(RAW_PATH+'./fitness.csv', index_col=[0,1], header=[0,1])
fitness = fitness[fitness.index.get_level_values(0).isin(athletes)]

cols = [('ID and ANTROPOMETRY', 'Age'),
		('ID and ANTROPOMETRY', 'HbA1C'),
		('ID and ANTROPOMETRY', 'bf(%)'),
		('VT2 (RCP)', 'W'), # TODO: check
		('VO2peak', 'HR'), # TODO: check
		('VO2peak', 'VO2/Kg')] # TODO: check

fitness = fitness.loc[pd.IndexSlice[:, 'Jan_2019'], cols]
fitness = fitness.droplevel(level=0, axis=1).droplevel(level=1, axis=0)
fitness.rename(columns={'Age'		: 'Age (years)',
						'HbA1C'		: 'HbA1C (%)',
						'bf(%)'		: 'Fat mass (%)',
						'W'			: 'FTP (W)',
						'HR'		: 'HR_max (bpm)',
						'VO2/Kg'	: 'VO2_max (mL/min/kg)'}, inplace=True)

stats_fit = fitness.apply(['mean', 'std', 'min', 'max'])

trainingpeaks = pd.read_csv(DATA_PATH+'./trainingpeaks_day.csv', index_col=[0,1])
stats_train_session = trainingpeaks[['training_stress_score', 'intensity_factor']].apply(['mean', 'std', 'min', 'max'])

stats_train_year = trainingpeaks.groupby('RIDER')[['timestamp_count', 'race', 'distance_max']].sum()
stats_train_year = stats_train_year.apply(['mean', 'std', 'min', 'max'])
stats_train_year['timestamp_count'] /= 3600 # to hour
stats_train_year['distance_max'] /= 1000
stats_train_year.rename(columns={'race'				:'Number of races', 
								 'timestamp_count'	:'Time cycled (h/year)',
								 'distance_max'		:'Distance cycled (km/year)'}, inplace=True)

# TODO:
# mean
# std/mean (Glycemic variability)
# % time in hypoglycemic range
# % time in hyperglycemic range
# hypoglycemia definition:
# hyperglycemia definition
# area under blood glucose curve
# lgbi and hgbi
# source: International Consensus on Use of Continuous Glucose Monitoring


df.apply({'Glucose Value (mg/dL)':['mean',
								   'std',
								   lambda x: x.std() / x.mean()],
		  ''})
