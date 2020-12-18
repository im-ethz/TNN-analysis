# TODO: include units in column names
# remove first index col
import numpy as np
import pandas as pd
import datetime
#from dateutil.tz import tzoffset
import os

from plot import *
from helper import *

path = 'Data/TrainingPeaks/'
if not os.path.exists(path+'clean/'):
	os.mkdir(path+'clean/')

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'csv/')])

for i in athletes:
	print(i)
	if not os.path.exists(path+'clean/'+str(i)):
		os.mkdir(path+'clean/'+str(i))

	filename_date = {}

	df = pd.DataFrame()
	df_nan = pd.DataFrame()

	files = sorted(os.listdir(path+'csv/'+str(i)+'/data'))
	for f in files:
		name = f.rstrip('_data.csv')

		df_data = pd.read_csv(path+'csv/'+str(i)+'/data/'+f)
		df_info = pd.read_csv(path+'csv/'+str(i)+'/info/'+name+'_info.csv', index_col=(0,1))

		df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])#.dt.tz_localize('UTC')

		filename_date.update({f:df_data['timestamp'].dt.date.unique()})

		try:
			# calculate timezone
			tz = (pd.to_datetime(df_info.loc['activity'].loc['local_timestamp'])
				- pd.to_datetime(df_info.loc['activity'].loc['timestamp']))[0]
			#tz = tzoffset(name='', offset=tz)

			#df_data['local_timestamp'] = df_data['timestamp'].dt.tz_convert(tz=tz)
			df_data['local_timestamp'] = df_data['timestamp'] + tz
		except KeyError:
			df_data['local_timestamp'] = np.nan

		# open all other files and check for missing info
		try:
			df_n = pd.read_csv(path+'csv/'+str(i)+'/nan/'+name+'_nan.csv', index_col=0)
			df_nan = df_nan.append(df_n, ignore_index=True, verify_integrity=True)
		except FileNotFoundError:
			pass

		df = df.append(df_data, ignore_index=True, verify_integrity=True)

	df['Zwift'] = False

	# if zwift files
	if os.path.exists(path+'csv/'+str(i)+'/Zwift/data'):
		files_virtual = sorted(os.listdir(path+'csv/'+str(i)+'/Zwift/data'))
		for f in files_virtual:
			name = f.rstrip('_data.csv')

			df_data = pd.read_csv(path+'csv/'+str(i)+'/Zwift/data/'+f)
			df_info = pd.read_csv(path+'csv/'+str(i)+'/Zwift/info/'+name+'_info.csv', index_col=(0,1))

			df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])#.dt.tz_localize('UTC')

			filename_date.update({f:df_data['timestamp'].dt.date.unique()})

			try:
				# calculate timezone
				tz = (pd.to_datetime(df_info.loc['activity'].loc['local_timestamp'])
					- pd.to_datetime(df_info.loc['activity'].loc['timestamp']))[0]

				#df_data['local_timestamp'] = df_data['timestamp'].dt.tz_convert(tz=tz)
				df_data['local_timestamp'] = df_data['timestamp'] + tz
			except KeyError:
				df_data['local_timestamp'] = np.nan

			# open all other files and check for missing info
			try:
				df_n = pd.read_csv(path+'csv/'+str(i)+'/Zwift/nan/'+name+'_nan.csv', index_col=0)
				df_nan = df_nan.append(df_n, ignore_index=True, verify_integrity=True)
			except FileNotFoundError:
				pass

			df = df.append(df_data, ignore_index=True, verify_integrity=True)

		df['Zwift'].fillna(True, inplace=True)

	# create dict with dates that refers to filenames
	date_filename = pd.DataFrame.from_dict(filename_date, orient='index').stack().reset_index()\
		.rename(columns={'level_0':'file', 0:'date'}).drop('level_1', axis=1)\
		.groupby('date').apply(lambda x:np.unique(x))
	date_filename = pd.DataFrame(date_filename, columns=['files'])
	date_filename['N'] = date_filename['files'].apply(lambda x:len(x))
	date_filename.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_datefilename.csv')

	# remove unknown columns from into nans file TODO: process later
	cols_unk = df.columns[df.columns.str.startswith('unknown')]
	df_nan = df_nan.append(df[['timestamp']+list(cols_unk)].dropna(subset=cols_unk, how='all'))
	df.drop(cols_unk, axis=1, inplace=True)

	# print duplicate timestamps # TODO
	print("Number of duplicated entries: ", df.drop('Unnamed: 0', axis=1).duplicated().sum())
	print("Number of duplicated timestamps: ", df['timestamp'].duplicated().sum())
	print("Number of days with duplicate timestamps: ", len(df[df['timestamp'].duplicated()]['timestamp'].dt.date.unique()))

	df_dupl = df[df['timestamp'].duplicated(keep=False)]
	if not df_dupl.empty:
		df_dupl['date'] = df_dupl['timestamp'].dt.date
		df_dupl[df_dupl['timestamp'] == df_dupl.iloc[0]['timestamp']]

		print("Dates with multiple files but not duplicate timestamps:\n",
			*tuple(set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist()) 
				- set(df_dupl['date'].unique())))
		print("Dates with duplicate timestamps but not multiple files:\n",
			*tuple(set(df_dupl['date'].unique()) 
				- set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist())))

	# save df to file
	df.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_data.csv', index_label=False)