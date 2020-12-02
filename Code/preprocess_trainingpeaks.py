import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *

tp_path = 'Data/TrainingPeaks/'
if not os.path.exists(tp_path+'clean/'):
	os.mkdir(tp_path+'clean/')

tp_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(tp_path+'csv/')])

for i in tp_athletes:
	df = pd.DataFrame()
	files = sorted(os.listdir(tp_path+'csv/'+str(i)+'/data'))
	for f in files:
		df_tmp = pd.read_csv(tp_path+'csv/'+str(i)+'/data/'+f)

		# open all other files and check for missing info


		df = df.append(df_tmp, ignore_index=True, verify_integrity=True)

	# save df to file
	df.to_csv(tp_path+'clean/'+str(i)+'_data.csv', index_label=False)

	# check when glucose data is present and when not
	df_glucose = df[['timestamp', 'glucose']]

	# make a list similar to the one from libreview with all glucose entries
	df_glucose_list = df_glucose[~df_glucose['glucose'].isna()]
	df_glucose_list.to_csv(tp_path+'clean/'+str(i)+'_data_glucose.csv', index_label=False)

	# create an array with dates and whether the glucose is present in the file
	df_glucose['date'] = pd.to_datetime(df_glucose['timestamp']).dt.date
	df_glucose.drop(['timestamp'], axis=1, inplace=True)
	df_glucose['glucose_present'] = ~df_glucose['glucose'].isna()
	df_glucose = df_glucose.groupby(['date']).sum()
	df_glucose.drop('glucose', axis=1, inplace=True)
	df_glucose['glucose_present'] = df_glucose['glucose_present'].astype(bool)

	# plot calendar with glucose availaibility
	df_glucose_calendar = create_calendar_array(df_glucose.copy(), 'glucose_present')
	month_mapping = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july',
					 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
	df_glucose_calendar.index = df_glucose_calendar.index.map(month_mapping)

	binary_mapping = {True:1, False:-1, np.nan:0}
	df_glucose_calendar = df_glucose_calendar.applymap(lambda x:binary_mapping[x])

	PlotData(savedir=tp_path+'clean/', athlete=i).plot_glucose_availability_calendar(df_glucose_calendar, cmap=custom_colormap('PiYG', 0.1, 0.9, 3), linewidth=.5)