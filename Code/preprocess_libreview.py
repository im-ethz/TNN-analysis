import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *

path = 'Data/LibreView/'
files = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'raw_anonymous/')])

if not os.path.exists(path+'clean/'):
	os.mkdir(path+'clean/')

dict_files = {}
for i in files:
	print('\n')
	print(i)
	df = pd.read_csv(path+'raw_anonymous/'+str(i)+'.csv', skiprows=2)

	if df.empty:
		print(i, " is empty")
		continue

	# drop duplicates #TODO: why are there duplicates in the data??
	df.drop_duplicates(ignore_index=True, inplace=True)

	# drop nan rows
	df.dropna(how='all', subset=['Historic Glucose mg/dL', 'Scan Glucose mg/dL',
       'Non-numeric Rapid-Acting Insulin', 'Rapid-Acting Insulin (units)',
       'Non-numeric Food', 'Carbohydrates (grams)', 'Carbohydrates (servings)',
       'Non-numeric Long-Acting Insulin', 'Long-Acting Insulin Value (units)',
       'Notes', 'Strip Glucose mg/dL', 'Ketone mmol/L', 'Meal Insulin (units)',
       'Correction Insulin (units)', 'User Change Insulin (units)'], inplace=True)
	
	# check for items with same timestamp but different glucose levels
	df_dupl = df[df.duplicated(subset=['Device Timestamp', 'Record Type', 'Device', 'Serial Number'], keep=False)]
	print("Duplicates ", df_dupl.drop(['Device', 'Serial Number'], axis=1).dropna(axis=1, how='all'))
	timestamp_dupl = df_dupl['Device Timestamp'].unique()

	# for record type 0 or 1, keep the last item, because that's probably the right item #TODO
	df[df['Record Type'] <= 1] = df[df['Record Type'] <= 1].drop_duplicates(subset=['Device Timestamp', 'Record Type', 'Device', 'Serial Number'], keep='last', ignore_index=True)
	df.dropna(how='all', inplace=True)

	df_dupl = df[df.duplicated(subset=['Device Timestamp', 'Record Type', 'Device', 'Serial Number'], keep=False)]
	if not df_dupl.empty:
		print("Still duplicates", df_dupl.drop(['Device', 'Serial Number'], axis=1).dropna(axis=1, how='all'))

	#print("Dupl timestamp", df[(df['Device Timestamp'].isin(timestamp_dupl))].drop(['Device', 'Serial Number'], axis=1).dropna(axis=1, how='all'))

	print("Empty columns: ", set(df.columns) - set(df.dropna(axis=1, how='all').columns))
	print("Non-empty columns: ", df.dropna(axis=1, how='all').columns)

	print("Notes: ", df['Notes'].dropna())

	# TODO: process notes
	df.to_csv(path+'clean/'+str(i)+'.csv', index_label=False)