import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *

path = 'Data/LibreView/'
files = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'raw_anonymous/')])

dict_files = {}
for i in files:
	print('\n')
	print(i)
	df = pd.read_csv(path+'clean/'+str(i)+'.csv')

	if df.empty:
		print(i, " is empty")
		continue

	df['glucose_present'] = True

	# check when glucose data is present and when not
	df_glucose = df[['Device Timestamp', 'glucose_present']]

	# create an array with dates and whether the glucose is present in the file
	df_glucose['date'] = pd.to_datetime(df_glucose['Device Timestamp'], dayfirst=True).dt.date
	df_glucose.drop(['Device Timestamp'], axis=1, inplace=True)
	df_glucose = df_glucose.groupby(['date']).sum()
	df_glucose_measurements = df_glucose.copy()
	df_glucose['glucose_present'] = df_glucose['glucose_present'].astype(bool)

	dict_files.update({i:df_glucose})

	# plot calendar with glucose availaibility
	df_glucose_calendar = create_calendar_array(df_glucose.copy(), 'glucose_present')
	df_glucose_calendar_measurements = create_calendar_array(df_glucose_measurements.copy(), 'glucose_present')

	month_mapping = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july',
					 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
	df_glucose_calendar.index = df_glucose_calendar.index.map(month_mapping)
	df_glucose_calendar_measurements.index = df_glucose_calendar_measurements.index.map(month_mapping)

	binary_mapping = {np.nan:0, True:1}
	df_glucose_calendar = df_glucose_calendar.applymap(lambda x:binary_mapping[x])

	# transform dtype to int
	df_glucose_calendar_measurements.fillna(0, inplace=True)
	df_glucose_calendar_measurements = df_glucose_calendar_measurements.astype(int)

	plot = PlotData(savedir=path+'clean/', athlete=i)
	plot.plot_glucose_availability_calendar(df_glucose_calendar, dtype='LibreView',
		cbarticks=binary_mapping, cmap=custom_colormap('PiYG', 0.5, 0.9, 2), linewidth=.5,
		annot=df_glucose_calendar_measurements, fmt="d", annot_kws={'size':8})
		#mask=df_glucose_calendar_measurements.isnull(), 

df_all_glucose = pd.concat(dict_files.values(), keys=dict_files.keys())

# create calendar with glucose availability for all athletes
df_all_glucose_calendar = df_all_glucose.unstack()['glucose_present']
df_all_glucose_calendar = df_all_glucose_calendar.applymap(lambda x:binary_mapping[x])

dates = pd.to_datetime(df_all_glucose_calendar.columns)
# TODO: to function
plt.figure(figsize=(15,4))
ax = sns.heatmap(df_all_glucose_calendar, cbar=False, cmap=custom_colormap('PiYG', 0.5, 0.9, 2), linewidth=.5)
ax.set_xticks(np.arange(len(dates))[1::7])
ax.set_xticklabels(dates.strftime('%d-%b')[1::7], rotation=45)
plt.yticks(rotation=0)
plt.xlabel('Date')
plt.ylabel('Athlete')
plt.title("Glucose availability LibreView\n May - Oct 2020")
plt.savefig(path+'clean/glucose_availaibility.pdf', bbox_inches='tight')