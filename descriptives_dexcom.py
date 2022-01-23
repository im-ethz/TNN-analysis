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
info = pd.read_csv(DATA_PATH+'info.csv', index_col=0)
info.set_index('RIDER', inplace=True)

df = pd.read_csv(RAW_PATH+'dexcom.csv', index_col=0)
df.drop('local_timestamp_raw', axis=1, inplace=True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

athletes = df.RIDER.unique()

# select glucose measurements
df = df[df['Event Type'] == 'EGV']
df = df[['RIDER', 'timestamp', 'local_timestamp', 'Source Device ID', 'Glucose Value (mg/dL)',
		 'Transmitter ID', 'Transmitter Time (Long Integer)', 'source', 'training']]

# -------------------------- Glucose availability (1 - raw)
# create calendar with glucose availability
max_readings = 24*60/5+1
df_avail = df.groupby(['RIDER', df.timestamp.dt.date])['Glucose Value (mg/dL)'].count().unstack() / max_readings

"""
dates = pd.date_range('2016-01-01', '2021-12-31')
df_empty = pd.DataFrame(index=dates, columns=df.RIDER.unique()).T
df_avail = df_empty.fillna(df_avail).fillna(0)
"""

# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
# fig, ax = plt.subplots(figsize=(30,6))
ax = sns.heatmap(df_avail, cmap='Blues', vmin=0, vmax=1)

# put total percentage on RHS
ax2 = ax.secondary_yaxis("right")
ax2.set_yticks(np.arange(len(df_avail.index))+0.5)
ax2.set_yticklabels([r"$\bf{:.0f}\%$".format(i) for i in df_avail.sum(axis=1)/365*100])
ax2.tick_params(axis='y', length=0)
ax2.spines['right'].set_visible(False)
ax.text(0.99, 1.02, r'$\bf{:s}$'.format('Total (\%)'), ha='left', transform=ax.transAxes)

# adjust ticks colorbar
cbar_ax = fig.get_axes()[1]
cbar_ax.set_yticks([0, .2, .4, .6, .8, 1.])
cbar_ax.set_yticklabels(["{:.0f}%".format(i*100) for i in [0, .2, .4, .6, .8, 1.]])
cbar_ax.text(3., 0.5, 'Percentage of max. CGM readings per day', va='center', rotation=270)

#plt.xticks(ticks=np.arange(0, len(dates)-1, 90), labels=dates[::90].strftime('%Y-%b'), rotation=45)
plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.title('CGM availability', fontweight='bold', fontsize=12)

plt.savefig(SAVE_PATH+'availability/glucose_availability_RAW.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability_RAW.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(SAVE_PATH+'availability/glucose_availability_RAW_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability_RAW_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# -------------------------- Read CGM data : resampled + completeness selection + incl sections
df = pd.read_csv(DATA_PATH+'dexcom_clean_nocomp.csv', index_col=0)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

# filter completeness
df = df[df['completeness'] >= 0.7]
df.dropna(subset=['Glucose Value (mg/dL)'], inplace=True)
df.reset_index(drop=True, inplace=True)

# rename sections
df.rename(columns={'train':'ex', 'post':'post-ex'}, inplace=True)

# create calendar with glucose availability
max_readings = 24*60/5+1
df_avail = df.groupby(['RIDER', df.timestamp.dt.date])['Glucose Value (mg/dL)'].count().unstack() / max_readings
info['cgm_days'] = df_avail.count(axis=1)

# -------------------------- Glucose availability (2 - clean)
# plot glucose availability per day
fig, ax = plt.subplots(figsize=(15,6))
ax = sns.heatmap(df_avail, cmap='Blues', vmin=0, vmax=1)

# put total percentage on RHS
ax2 = ax.secondary_yaxis("right")
ax2.set_yticks(np.arange(len(df_avail.index))+0.5)
ax2.set_yticklabels([r"$\bf{:.0f}$".format(i) for i in df_avail.count(axis=1)])
ax2.tick_params(axis='y', length=0)
ax2.spines['right'].set_visible(False)
ax.text(0.99, 1.02, r'$\bf{:s}$'.format('Total (days)'), ha='left', transform=ax.transAxes)

# adjust ticks colorbar
cbar_ax = fig.get_axes()[1]
cbar_ax.set_yticks([0, .2, .4, .6, .8, 1.])
cbar_ax.set_yticklabels(["{:.0f}%".format(i*100) for i in [0, .2, .4, .6, .8, 1.]])
#cbar_ax.set_title('Days with CGM readings')
cbar_ax.text(3., 0.5, 'Percentage of max. CGM readings per day', va='center', rotation=270)

plt.xticks(ticks=[d+15 for d in month_firstday.values()], labels=[list(month_firstday.keys())[-1]]+list(month_firstday.keys())[:-1], rotation=0)
plt.ylabel('rider')
plt.title('CGM availability', fontweight='bold', fontsize=12)

plt.savefig(SAVE_PATH+'availability/glucose_availability.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability.png', dpi=300, bbox_inches='tight')
ax.set_yticklabels([rider_mapping_inv[int(i.get_text())] for i in ax.get_yticklabels()], rotation=0)
plt.savefig(SAVE_PATH+'availability/glucose_availability_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'availability/glucose_availability_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# -------------------------- Glucose distributions
col = 'Glucose Value (mg/dL)'
sections = ('wake', 'ex', 'post-ex', 'sleep')
for sec in sections:
	df[sec] = df[sec].astype(bool)
# TODO: so far no exclusions from different sections

# color palettes
color_palette = sns.color_palette("Set1")
color_sec = {'wake'	: color_palette[1],
			 'ex': color_palette[4],
			 'post-ex': color_palette[2],
			 'sleep': color_palette[3]}
palette_sec = {	'wake'	: sns.color_palette('Blues', n_colors=11),
				'ex'	: sns.color_palette('Oranges', n_colors=11),
				'post-ex'	: sns.color_palette('Greens', n_colors=11),
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
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(i, df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']), y=1.06)
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(rider_mapping_inv[i], df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']), y=1.06)
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'hist/hist_glucose_sec_NAME_%s.png'%i, dpi=300, bbox_inches='tight')	
	plt.show()
	plt.close()

# plot hue athletes
fig, ax0 = plt.subplots(figsize=(5, 3.5))
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for c, i in enumerate(df.RIDER.unique()):
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
for c, i in enumerate(df.RIDER.unique()):
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
	for c, i in enumerate(df.RIDER.unique()):
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
plt.ylabel('Time in glucose ranges (%)')

ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
ax.yaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# stratify by athlete, hue: stages
for i in df.RIDER.unique():
	fig, ax = plt.subplots(figsize=(3, 3.5))
	for k, sec in enumerate(sections):
		pil = get_percinlevel(df[df[sec]][df[df[sec]].RIDER == i])
		plot_bar(pil, x=k, colors=colors[k])

	plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.xticks(np.arange(len(sections)), sections)
	plt.ylim((0,100))
	plt.ylabel('Time in glucose ranges (%)')

	ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
	ax.yaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(i, df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(rider_mapping_inv[i], df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

"""
colors = [dict(zip(['h_neg', 'h_pos', 'l', 's'], [c[0]*360, c[0]*360, c[1]*100, c[2]*100])) \
			for c in [rgb_to_hls(*j) for j in palette_ath]]
"""

# hue: athletes
fig, ax = plt.subplots(figsize=(8,4))
for n, i in enumerate(df.RIDER.unique()):
	pil = get_percinlevel(df[df.RIDER == i])
	plot_bar(pil, x=n, width=.7)

plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
plt.xticks(np.arange(len(df.RIDER.unique())), df.RIDER.unique())
plt.ylim((0,100))
plt.ylabel('Time in glucose ranges (%)')

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
plt.ylabel('Time in glucose ranges (%)')

ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
ax.yaxis.grid(True)
sns.despine(left=True, bottom=True, right=True)

plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# stratify by athlete, hue: race
for i in df.RIDER.unique():
	fig, ax = plt.subplots(figsize=(1,4))
	for k, b in enumerate([False, True]):
		pil = get_percinlevel(df[(df.RIDER == i) & (df['race'] == b)])
		plot_bar(pil, x=k, colors=colors[k])

	plt.yticks(np.arange(0, 101, 20), ['{}%'.format(j) for j in np.arange(0, 101, 20)])
	plt.xticks(np.arange(2), [j for j in color_race.keys()])
	plt.ylim((0,100))
	plt.ylabel('Time in glucose ranges (%)')

	ax.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.5, 0.5))
	ax.yaxis.grid(True)
	sns.despine(left=True, bottom=True, right=True)

	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(i, df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_in_zone/time_in_glucoselevel_race_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(rider_mapping_inv[i], df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
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

df['date'] = df['local_timestamp'].dt.date

n_nc = len(df.loc[df['race'] == False, ['RIDER', 'date']].drop_duplicates())
n_rc = len(df.loc[df['race'] == True, ['RIDER', 'date']].drop_duplicates())

fig, ax = plt.subplots(figsize=(5,3.5))
for k, sec in enumerate(sections):
	plt.boxplot(df.loc[df[sec] & df['race']==False, col], positions=[k+0.5+k*2], widths=[0.8],
		patch_artist=True, showfliers=False, **kws_box['normal'])
	plt.boxplot(df.loc[df[sec] & df['race']==True, col], positions=[k+1.5+k*2], widths=[0.8],
		patch_artist=True, showfliers=False, **kws_box['race'])
plt.xticks([1,4,7,10], sections)
plt.ylabel(col)
plt.legend(handles=[Patch(facecolor=nc, edgecolor='white', hatch='///', label='normal day'+r' ($n = {:.0f}$)'.format(n_nc)),
					Patch(facecolor=rc, edgecolor='white', hatch='\\\\\\', label='race day'r' ($n = {:.0f}$)'.format(n_rc))], 
			loc='upper right')
plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

for i in df.RIDER.unique():
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
	plt.legend(handles=[Patch(facecolor=nc, edgecolor='white', hatch='///', label='normal day'+r' ($n = {:.0f}$)'.format(n_nc)),
						Patch(facecolor=rc, edgecolor='white', hatch='\\\\\\', label='race day'r' ($n = {:.0f}$)'.format(n_rc))], 
				loc='upper right')
	plt.title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(i, df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.title(r'$\bf{:s}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(rider_mapping_inv[i], df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'boxplot/box_glucose_sections_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

n_nc = len(df.loc[df['race'] == False, ['RIDER', 'date']].drop_duplicates())
n_rc = len(df.loc[df['race'] == True, ['RIDER', 'date']].drop_duplicates())

# hue: athlete
fig, ax = plt.subplots(figsize=(10,4))
for n, i in enumerate(df.RIDER.unique()):
	plt.boxplot(df.loc[(df.RIDER == i) & (df['race'] == False), col], positions=[n+0.5+n*2], widths=[0.8],
		patch_artist=True, showfliers=False, **kws_box['normal'])
	plt.boxplot(df.loc[(df.RIDER == i) & (df['race'] == True), col], positions=[n+1.5+n*2], widths=[0.8],
		patch_artist=True, showfliers=False, **kws_box['race'])
plt.ylabel(col)
plt.legend(handles=[Patch(facecolor=nc, edgecolor='white', hatch='///', label='normal day'+r' ($n = {:.0f}$)'.format(n_nc)),
					Patch(facecolor=rc, edgecolor='white', hatch='\\\\\\', label='race day'r' ($n = {:.0f}$)'.format(n_rc))], 
			loc='upper right')
plt.xticks([1,4,7,10,13,16,19,22,25,28,31], df.RIDER.unique())
plt.savefig(SAVE_PATH+'boxplot/box_glucose_riders.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'boxplot/box_glucose_riders.png', dpi=300, bbox_inches='tight')
plt.xticks([1,4,7,10,13,16,19,22,25,28,31], [rider_mapping_inv[i] for i in df.RIDER.unique()])
plt.savefig(SAVE_PATH+'boxplot/box_glucose_riders_NAME.pdf', bbox_inches='tight')
plt.savefig(SAVE_PATH+'boxplot/box_glucose_riders_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# -------------------------- Glucose longitudinal
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

# create TID and t
for j, ((i,n), (t_min, t_max)) in enumerate(df_training[['local_timestamp_min_r', 'local_timestamp_max_r']].iterrows()):
	mask = (df.RIDER == i) & (df.local_timestamp >= t_min) & (df.local_timestamp <= t_max)
	df.loc[mask, 'tid'] = j
	df.loc[mask, 't'] = df.loc[mask, 'local_timestamp'] - t_min

df['t'] = df['t'].dt.seconds / 60 # time in minutes

n_nc = len(df.loc[df['race'] == False, ['RIDER', 'date']].drop_duplicates())
n_rc = len(df.loc[df['race'] == True, ['RIDER', 'date']].drop_duplicates())

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

for i in df.RIDER.unique():
	n_nc = len(df.loc[(df.RIDER == i) & (df['race'] == False), 'local_timestamp'].dt.date.unique())
	n_rc = len(df.loc[(df.RIDER == i) & (df['race'] == True), 'local_timestamp'].dt.date.unique())

	t_max = df.loc[df.RIDER == i, 't'].max()

	fig, ax = plt.subplots(1,2, figsize=(5, 3), sharey=True, gridspec_kw={'width_ratios': [5, 1], 'wspace':0}, tight_layout=True)
	sns.lineplot(df.loc[df.RIDER == i,'t'], df.loc[df.RIDER == i, col], hue=df['race'], palette=list(color_race.values()), ax=ax[0])
	ax[0].set_xlabel('Time in training session (min)')
	ax[0].legend(loc='upper right', labels=['normal'+r' ($n = {:.0f}$)'.format(n_nc), 
											'race'r' ($n = {:.0f}$)'.format(n_rc)])
	sns.kdeplot(data=df.loc[df['t'].notna() & (df.RIDER == i)], y=col, hue='race', ax=ax[1],
		fill=True, alpha=.5, palette=list(color_race.values()), linewidth=0, legend=False)
	sns.despine(bottom=True, right=True, top=True, ax=ax[1])
	ax[1].set(xlabel=None, xticks=[])
	plt.ylim((0,350))
	ax[0].set_xlim((0, t_max))
	ax[0].set_title(r'$\bf{Rider}$ '+r'$\bf{:d}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(i, df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s.png'%i, dpi=300, bbox_inches='tight')
	ax[0].set_title(r'$\bf{:s}$ - $\mu = {:.1f}$  $\sigma/\mu = {:.0f}\%$  T1D duration:{:.0f}yr  data:{:.0f}days'\
		.format(rider_mapping_inv[i], df[df.RIDER == i][col].mean(),
				df[df.RIDER == i][col].std()/df[df.RIDER == i][col].mean()*100,
				info.loc[i, 'diabetes_duration'],
				info.loc[i, 'cgm_days']))
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s_NAME.pdf'%i, bbox_inches='tight')
	plt.savefig(SAVE_PATH+'time_training/glucose_training_%s_NAME.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()

# -------------------------- Descriptives
# TODO: gender (m/f), CSII/MDI
# CGM completeness
# CSII / MDI (1/10)
# VO2max (mL min-1 kg-1)
# IPAQ score (MET min week-1)
dict_stats = {'age'								: 'Age (yr)',
			  'height'							: 'Height (cm)',
			  'weight'							: 'Weight (cm)',
			  'bf(%)'							: 'Fat mass (%)',
			  'diabetes_duration'				: 'Diabetes duration (yr)',
			  'HbA1c'							: 'HbA_{1c} (%)',
			  'cgm_days'						: 'Days with CGM coverage >70 %',
			  'FTP_per_kg'						: 'Functional threshold power (W/kg)',
			  'LTHR'							: 'Lactate threshold heart rate (bpm)',
			  'HRmax'							: 'HR_{max} (bpm)',
			  'VO2max'							: 'VO_{2max} (mL/min/kg)',
			  'days_cycled'						: 'Days cycled',
			  'days_raced'						: 'Days participated in competition',
			  'km_cycled'						: 'Distance cycled (km/yr)',
			  'hours_cycled_per_cycling_day'	: 'Mean time cycled (h/day)',
			  'km_cycled_per_cycling_day'		: 'Mean distance cycled (km/day)',
			  'km_ascended_per_cycling_day'		: 'Mean distance ascended (km/day)'}

dict_cgm = {  'cgm_mean'						: 'Mean glucose (mg/dL)',
			  'cgm_cv'							: 'Glycemic variability (%)',
			  'hypo L2'							: 'hypoglycemia L2 (<54 mg/dL)',
			  'hypo L1'							: 'hypoglycemia L1 (54-69 mg/dL)',
			  'target'							: 'target range (70-180 mg/dL)',
			  'hyper L1'						: 'hyperglycemia L1 (181-250 mg/dL)',
			  'hyper L2'						: 'hyperglycemia L2 (>250 mg/dL)'}

glucose_levels = {'hypo L2': (0,53),
				  'hypo L1': (54,69),
				  'target' : (70,180),
				  'hyper L1': (181,250),
				  'hyper L2': (251,10000)}

# info
info = pd.read_csv(DATA_PATH+'./info.csv', index_col=0)
info.set_index('RIDER', inplace=True)
info['FTP_per_kg'] = info['FTP'] / info['weight']
info.drop('FTP', axis=1, inplace=True)

cols_info = ['age', 'height', 'weight', 'bf(%)']
cols_diabetes = ['diabetes_duration', 'HbA1c']
cols_ex = ['FTP_per_kg', 'LTHR', 'HRmax', 'VO2max']

# diabetes
stats_diabetes = df.groupby(['RIDER', df.timestamp.dt.date])['Glucose Value (mg/dL)'].count().unstack().count(axis=1).rename('cgm_days')

# trainingpeaks
tp = pd.read_csv(DATA_PATH+'./trainingpeaks_day.csv', index_col=[0,1])
stats_cycling_year = tp.groupby('RIDER').agg({'date'			:'count', # cycling days / yr
											  'race'			:'sum', # race days / yr
											  'distance_max'	:lambda x: x.sum()/1000 # km cycled / yr
											  })\
							.rename(columns={'date'				:'days_cycled',
											 'race'				:'days_raced',
											 'distance_max'		:'km_cycled'})
stats_cycling_day = tp.groupby('RIDER').agg({ 'timestamp_count'			:lambda x: x.mean()/3600, #h cycled / cycling day
											  'distance_max'			:lambda x: x.mean()/1000, # km cycled / cycling day
											  'elevation_gain_up_sum'	:lambda x: x.mean()/1000, # km ascended / cycling day
											})\
							.rename(columns={'timestamp_count'			:'hours_cycled_per_cycling_day',
											 'distance_max'				:'km_cycled_per_cycling_day',
											 'elevation_gain_up_sum'	:'km_ascended_per_cycling_day'})

# combine
descriptives = pd.concat([info[cols_info], info[cols_diabetes], stats_diabetes, info[cols_ex], stats_cycling_year, stats_cycling_day], axis=1)
descriptives_sum = pd.concat([descriptives.mean(), descriptives.std()], axis=1)
descriptives_sum = descriptives_sum.round(1).apply(lambda x: '%s $\pm$ %s'%(x[0], x[1]), axis=1).rename(index=dict_stats)
with open(SAVE_PATH+"descriptives.tex", 'w') as file:
	file.write(descriptives_sum.to_latex(column_format='c'))


df['total'] = True
descriptives_cgm = {}
for sec in list(sections)+['total']:
	cgm_mean = df[df[sec]].groupby(['RIDER'])['Glucose Value (mg/dL)'].mean().rename('cgm_mean')
	cgm_cv = df[df[sec]].groupby('RIDER')['Glucose Value (mg/dL)'].apply(lambda x: x.std()/x.mean()*100).rename('cgm_cv')
	cgm_times = df[df[sec]].groupby(['RIDER']).apply(lambda x: get_percinlevel(x)).apply(pd.Series)
	stats_cgm = pd.concat([cgm_mean, cgm_cv, cgm_times], axis=1)
	descriptives_cgm[sec] = pd.concat([stats_cgm.mean(), stats_cgm.std()], axis=1)
	descriptives_cgm[sec] = descriptives_cgm[sec].round(1).apply(lambda x: '%s $\pm$ %s'%(x[0], x[1]), axis=1).rename(index=dict_cgm)
descriptives_cgm = pd.concat(descriptives_cgm, axis=1)

descriptives_cgm = descriptives_cgm[['total', 'wake', 'sleep', 'ex', 'post-ex']]

with open(SAVE_PATH+"descriptives_cgm.tex", 'w') as file:
	file.write(descriptives_cgm.to_latex(column_format='c'))