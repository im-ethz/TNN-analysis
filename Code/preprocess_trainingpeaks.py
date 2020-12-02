import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *

path = 'Data/TrainingPeaks/'
if not os.path.exists(path+'clean/'):
	os.mkdir(path+'clean/')

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'csv/')])

dict_files = {}
for i in athletes:
	no_glucose=False
	df = pd.DataFrame()

	files = sorted(os.listdir(path+'csv/'+str(i)+'/data'))
	for f in files:
		df_tmp = pd.read_csv(path+'csv/'+str(i)+'/data/'+f)

		# open all other files and check for missing info

		df = df.append(df_tmp, ignore_index=True, verify_integrity=True)

	df['Zwift'] = False

	files_virtual = sorted(os.listdir(path+'csv/'+str(i)+'/Zwift/data'))
	for f in files_virtual:
		df_tmp = pd.read_csv(path+'csv/'+str(i)+'/Zwift/data/'+f)

		# open all other files and check for missing info

		df = df.append(df_tmp, ignore_index=True, verify_integrity=True)

	df['Zwift'].fillna(True, inplace=True)

	# save df to file
	df.to_csv(path+'clean/'+str(i)+'_data.csv', index_label=False)

	# check when glucose data is present and when not
	try:
		df_glucose = df[['timestamp', 'glucose']]
	except KeyError:
		no_glucose=True
		df['glucose'] = np.nan
		df_glucose = df[['timestamp', 'glucose']]

	# make a list similar to the one from libreview with all glucose entries
	df_glucose_list = df_glucose[~df_glucose['glucose'].isna()]
	df_glucose_list.to_csv(path+'clean/'+str(i)+'_data_glucose.csv', index_label=False)

	# check if there are duplicated entries in the glucose file that are not nan
	print("duplicate entries not nan glucose", df_glucose[(df_glucose.duplicated(keep=False)) & (~df_glucose.glucose.isna())])
	df_glucose.drop_duplicates(ignore_index=True, inplace=True)

	# check if there are duplicate entries for the same timestamp
	print("duplicate timestamps", df_glucose[df_glucose.duplicated(subset='timestamp')])
	# TODO!!!!!

	# create an array with dates and whether the glucose is present in the file
	df_glucose['date'] = pd.to_datetime(df_glucose['timestamp']).dt.date
	df_glucose.drop(['timestamp'], axis=1, inplace=True)
	df_glucose['glucose_present'] = ~df_glucose['glucose'].isna()
	df_glucose = df_glucose.groupby(['date']).sum()
	df_glucose.drop('glucose', axis=1, inplace=True)
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

	binary_mapping = {False:-1, np.nan:0, True:1}
	df_glucose_calendar = df_glucose_calendar.applymap(lambda x:binary_mapping[x])

	# TODO: include number of measurements in each square
	plot = PlotData(savedir=path+'clean/', athlete=i)
	if not no_glucose:
		plot.plot_glucose_availability_calendar(df_glucose_calendar, dtype='TrainingPeaks', 
			cbarticks=binary_mapping, cmap=custom_colormap('PiYG', 0.1, 0.9, 3), linewidth=.5,
			annot=df_glucose_calendar_measurements, mask=df_glucose_calendar_measurements.isnull())
	else:
		plot.plot_glucose_availability_calendar(df_glucose_calendar, dtype='TrainingPeaks', 
			cbarticks=binary_mapping, cmap=custom_colormap('PiYG', 0.1, 0.5, 2), linewidth=.5)

df_all_glucose = pd.concat(dict_files.values(), keys=dict_files.keys())

# create calendar with glucose availability for all athletes
df_all_glucose_calendar = df_all_glucose.unstack()['glucose_present']
df_all_glucose_calendar = df_all_glucose_calendar.applymap(lambda x:binary_mapping[x])

dates = pd.to_datetime(df_all_glucose_calendar.columns)
# TODO: to function
plt.figure(figsize=(15,4))
ax = sns.heatmap(df_all_glucose_calendar, cbar=False, cmap=custom_colormap('PiYG', 0.1, 0.9, 3), linewidth=.5)
ax.set_xticks(np.arange(len(dates))[1::7])
ax.set_xticklabels(dates.strftime('%d-%b')[1::7], rotation=45)
plt.yticks(rotation=0)
plt.xlabel('Date')
plt.ylabel('Athlete')
plt.title("Glucose availability TrainingPeaks\n Jan - Oct 2020")
plt.savefig(path+'clean/glucose_availaibility.pdf', bbox_inches='tight')
