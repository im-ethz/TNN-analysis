import numpy as np
import pandas as pd
import datetime
import os
import gc

from plot import *
from helper import *

from matplotlib.colors import ListedColormap
pink = '#c41a7c' ; pinkl = '#f3a5d3'
green = '#4c9121' ; greenl = '#c5ecac'
gray = '#f7f7f6' ; grayd = '#c3c3bb'
yellowl = '#ffffe6'
cmap = ListedColormap([yellowl, pinkl, pink, grayd, grayd, green, greenl])

lv_path = 'Data/LibreView/clean/'
tp_path = 'Data/TrainingPeaks/clean/'

lv_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(lv_path) if i.endswith('.csv')])

for i in lv_athletes:
	# -------------------- Libreview
	df_lv = pd.read_csv(lv_path+str(i)+'.csv')

	df_lv['glucose_present'] = True
	df_lv_glucose = df_lv[['Device Timestamp', 'glucose_present']]

	# create an array with dates and whether the glucose is present in the file
	df_lv_glucose['date'] = pd.to_datetime(df_lv_glucose['Device Timestamp'], dayfirst=True).dt.date
	df_lv_glucose.drop(['Device Timestamp'], axis=1, inplace=True)
	df_lv_glucose = df_lv_glucose.groupby(['date']).sum()
	df_lv_glucose_measurements = df_lv_glucose.copy()
	df_lv_glucose['glucose_present'] = df_lv_glucose['glucose_present'].astype(bool)

	# plot calendar with glucose availaibility
	df_lv_glucose_calendar = create_calendar_array(df_lv_glucose.copy(), 'glucose_present')
	df_lv_glucose_calendar_measurements = create_calendar_array(df_lv_glucose_measurements.copy(), 'glucose_present')

	binary_mapping_lv = {np.nan:0, True:1}
	df_lv_glucose_calendar = df_lv_glucose_calendar.applymap(lambda x:binary_mapping_lv[x])

	# set calendar items up to first item to nan
	start_lv = df_lv_glucose.index.min()
	for j in range(1,start_lv.day):
		df_lv_glucose_calendar.loc[month_mapping[start_lv.month],j] = np.nan

	# transform dtype to int
	df_lv_glucose_calendar_measurements.fillna(0, inplace=True)
	df_lv_glucose_calendar_measurements = df_lv_glucose_calendar_measurements.astype(int)

	# -------------------- TrainingPeaks
	df_tp = pd.read_csv(tp_path+str(i)+'_data.csv')

	try:
		df_tp_glucose = df_tp[['timestamp', 'glucose']]
	except KeyError:
		continue
	df_tp_glucose.drop_duplicates(ignore_index=True, inplace=True)

	"""
	# make a list similar to the one from libreview with all glucose entries
	df_tp_glucose_list = df_tp_glucose[~df_tp_glucose['glucose'].isna()]
	"""

	df_tp_glucose['glucose_present'] = ~df_tp_glucose['glucose'].isna()
	df_tp_glucose.drop('glucose', axis=1, inplace=True)

	# create an array with dates and whether the glucose is present in the file
	df_tp_glucose['date'] = pd.to_datetime(df_tp_glucose['timestamp']).dt.date
	df_tp_glucose.drop(['timestamp'], axis=1, inplace=True)
	df_tp_glucose = df_tp_glucose.groupby(['date']).sum()
	df_tp_glucose_measurements = df_tp_glucose.copy()
	df_tp_glucose['glucose_present'] = df_tp_glucose['glucose_present'].astype(bool)

	# plot calendar with glucose availaibility
	df_tp_glucose_calendar = create_calendar_array(df_tp_glucose.copy(), 'glucose_present')
	df_tp_glucose_calendar_measurements = create_calendar_array(df_tp_glucose_measurements.copy(), 'glucose_present')

	binary_mapping_tp = {False:-1, np.nan:0, True:1} # False = cycling no glucose ; nan = no cycling ; True = cycling glucose
	df_tp_glucose_calendar = df_tp_glucose_calendar.applymap(lambda x:binary_mapping_tp[x])

	# -------------------- Combinations
	"""
	Old combinations: (left LV, right TP)
	nan + 		-> -6 -5 -4 outside available datarange
	0 + -1 = -1 -> -3	LibreView no glucose & TrainingPeaks cycling no glucose (good)
	0 + 0  = 0  -> -2	LibreView no glucose & TrainingPeaks no cycling (ok)
	0 + 1  = 1  -> -1	LibreView no glucose & TrainingPeaks cycling and glucose (ALARM)
	1 + -1 = 0  -> 0	LibreView glucose & TrainingPeaks cycling no glucose (TODO)
	1 + 0  = 1  -> 1	LibreView glucose & TrainingPeaks no cycling (ok)
	1 + 1  = 2  -> 2	LibreView glucose & TrainingPeaks cycling and glucose (good)
	df_combinations_calendar[df_lv_glucose_calendar == 0] -= 2


	New Combinations (left LV, right TP)
	0 + -1 = -1 -> -2 LibreView no glucose & TrainingPeaks cycling no glucose (good)
	1 + -1 = 0 -> -1 LibreView glucose & TrainingPeaks cycling no glucose (TODO)
	0 + 0 = 0 -> 0 LibreView no glucose & TrainingPeaks no cycling (ok, we don't care about this)
	1 + 0 = 1 -> 1 LibreView glucose & TrainingPeaks no cycling (ok, we don't care about this)
	0 + 1 = 1 -> 2 LibreView no glucose & TrainingPeaks cycling and glucose (ALARM)
	1 + 1 = 2 -> 3 LibreView glucose & TrainingPeaks cycling and glucose (good)

	"""
	df_combinations_calendar = df_tp_glucose_calendar.add(df_lv_glucose_calendar)
	df_combinations_calendar[df_tp_glucose_calendar == -1] -= 1
	df_combinations_calendar[df_tp_glucose_calendar == 1] += 1
	df_combinations_calendar.fillna(-3, inplace=True)

	combinations_mapping = {'outside available datarange':-3, 
							'LV no-gl ; TP cy no-gl (good)':-2, # pinkl
							'LV gl ; TP cy no-gl (TODO)':-1, #pink
							'LV no-gl ; TP no-cy (ok)':0, # grayd
							'LV gl ; TP no-cy (ok)':1, # grayd
							'LV no-gl ; TP cy gl (ALARM)':2, #green
							'LV gl ; TP cy gl (good)':3} #greenl

	plot = PlotData(savedir=lv_path+str(i)+'/', athlete=i, savetext='_comparison_tplv')
	plot.plot_glucose_availability_calendar(df_combinations_calendar, dtype='TrainingPeaks + LibreView\n',
		cbarticks=combinations_mapping, cmap=cmap, linewidth=.5)
		#annot=df_glucose_calendar_measurements, fmt="d", annot_kws={'size':8})


	# -------------------- Compare glucose
	# Can we combine Scan Glucose and Historic Glucose?
	print("At same timestamp value for historic glucose and scan glucose: \n", 
	df_lv[(~df_lv['Historic Glucose mg/dL'].isna()) & (~df_lv['Scan Glucose mg/dL'].isna())])
	df_lv.set_index('Device Timestamp', inplace=True)

	# create LibreView glucose list
	#df_lv_glucose_list = df_lv['Historic Glucose mg/dL'].combine_first(df_lv['Scan Glucose mg/dL']).to_frame(name='glucose')
	df_lv_glucose_list = df_lv[['Historic Glucose mg/dL', 'Scan Glucose mg/dL']]

	# create TrainingPeaks glucose list
	df_tp.set_index('timestamp', drop=True, inplace=True)
	df_tp_glucose_list = df_tp[['glucose']].dropna()

	# extract date
	df_lv_glucose_list['datetime'] = pd.to_datetime(df_lv_glucose_list.index, dayfirst=True)
	df_tp_glucose_list['datetime'] = pd.to_datetime(df_tp_glucose_list.index)
	df_lv_glucose_list['date'] = df_lv_glucose_list['datetime'].date
	df_tp_glucose_list['date'] = df_tp_glucose_list['datetime'].date
	tz = datetime.timedelta(hours=1)
	df_tp_glucose_list['datetime'] += tz

	# find measurements on the same day
	df_glucose_overlap_dates = set(df_tp_glucose_list.date) & set(df_lv_glucose_list.date)
	print("Number of TP glucose measurements that also appear in LV: ", 
		df_tp_glucose_list.date.isin(df_lv_glucose_list.date).sum())

	print("Number of LV glucose measurements that also appear in TP: ",
		df_lv_glucose_list.date.isin(df_tp_glucose_list.date).sum())

	print("Number of overlapping dates for glucose measurements in TP and LV: ",
		len(df_glucose_overlap_dates))

	#df_tp_glucose_overlap = df_tp_glucose_list[df_tp_glucose_list.date.isin(df_lv_glucose_list.date)]
	#df_lv_glucose_overlap = df_lv_glucose_list[df_lv_glucose_list.date.isin(df_tp_glucose_list.date)]

	df_test = df_tp_glucose_list[df_tp_glucose_list.date == d].reset_index().reset_index()

	for d in df_glucose_overlap_dates:
		plt.figure(figsize=(15,4))
		sns.scatterplot(data=df_tp_glucose_list[df_tp_glucose_list.date == d], x='datetime', y='glucose')
		sns.scatterplot(data=df_lv_glucose_list[df_lv_glucose_list.date == d], x='datetime', y='Historic Glucose mg/dL')
		ax = sns.scatterplot(data=df_lv_glucose_list[df_lv_glucose_list.date == d], x='datetime', y='Scan Glucose mg/dL')
		df_tp_glucose_list[df_tp_glucose_list.date == d].plot(x='datetime', y='glucose', ax=ax)
		df_lv_glucose_list[df_lv_glucose_list.date == d].plot(x='datetime', y='Historic Glucose mg/dL', ax=ax)
		df_lv_glucose_list[df_lv_glucose_list.date == d].plot(x='datetime', y='Scan Glucose mg/dL', ax=ax)
		#plt.xticks(rotation=90)
		plt.show()
		for j in df_tp_glucose_list[df_tp_glucose_list.date == d]['glucose']:
			if not df_lv_glucose_list[(df_lv_glucose_list.date == d) & (df_lv_glucose_list['Historic Glucose mg/dL'] == j)].empty:
				print(d, j, df_lv_glucose_list[(df_lv_glucose_list.date == d) & (df_lv_glucose_list['Historic Glucose mg/dL'] == j)])
			#else:
			#	print(d, j, "NO OVERLAP Historic Glucose mg/dL")
			if not df_lv_glucose_list[(df_lv_glucose_list.date == d) & (df_lv_glucose_list['Scan Glucose mg/dL'] == j)].empty:
				print(d, j, df_lv_glucose_list[(df_lv_glucose_list.date == d) & (df_lv_glucose_list['Scan Glucose mg/dL'] == j)])
				print("\n")
			#else:
			#	print(d, j, "NO OVERLAP Scan Glucose mg/dL")



"""
lv_files = {}
for i in lv_athletes:
	df = pd.read_csv(lv_path+str(i)+'.csv')
	lv_files.update({i:df})
df_lv = pd.concat(lv_files.values(), keys=lv_files.keys())
del lv_files ; gc.collect()

tp_files = {}
for i in lv_athletes:
	df = pd.read_csv(tp_path+str(i)+'_data.csv')
	tp_files.update({i:df})
del df ; gc.collect()
df_tp = pd.concat(tp_files.values(), keys=tp_files.keys())
del tp_files ; gc.collect()
"""

# drop empty columns
df_lv.dropna(axis=1, how='all', inplace=True)

# insert pandas datetime column
df_lv['Device Timestamp (datetime64)'] = pd.to_datetime(df_lv['Device Timestamp'], dayfirst=True)
df_lv['date'] = df_lv['Device Timestamp (datetime64)'].dt.date
df_lv['time'] = df_lv['Device Timestamp (datetime64)'].dt.time

