# TODO: take mean of duplicate values
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
		print("WARNING: LibreView data for Athlete ", i, " is empty")
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
	if not df_dupl.empty:
		print("WARNING: Items with same timestamp but different glucose levels\n",
			df_dupl.drop(['Device', 'Serial Number'], axis=1).dropna(axis=1, how='all'))
	timestamp_dupl = df_dupl['Device Timestamp'].unique()

	# for the items with same timestamp but different glucose levels:
	# for record type 0 or 1, keep the last item, because that's probably the right item #TODO
	df[df['Record Type'] <= 1] = df[df['Record Type'] <= 1].drop_duplicates(subset=['Device Timestamp', 'Record Type', 'Device', 'Serial Number'], keep='last', ignore_index=True)
	df.dropna(how='all', inplace=True) #this is necessary because previous step sets rows to nan somehow

	# prints items with the same timestamp but different glucose levels that did not get removed
	# (because they don't have record type 0 or 1)
	df_dupl = df[df.duplicated(subset=['Device Timestamp', 'Record Type', 'Device', 'Serial Number'], keep=False)]
	if not df_dupl.empty:
		print("WARNING: After fixing the items with record type 0 or 1, there are still items with the same timestamp but different glucose levels\n", 
			df_dupl.drop(['Device', 'Serial Number'], axis=1).dropna(axis=1, how='all'))

	#print("Dupl timestamp", df[(df['Device Timestamp'].isin(timestamp_dupl))].drop(['Device', 'Serial Number'], axis=1).dropna(axis=1, how='all'))

	print("Empty columns: ", set(df.columns) - set(df.dropna(axis=1, how='all').columns))
	print("Non-empty columns: ", set(df.dropna(axis=1, how='all').columns))

	# TODO: process notes
	if not df['Notes'].dropna().empty:
		print("TODO: The following notes are present in the file.\n", df['Notes'].dropna().values)

	"""
	# print columns associated with each record type
	for j in range(int(df['Record Type'].max())+1):
		print("Columns associated with Type %s : "%j, 
			set(df[df['Record Type'] == j].dropna(axis=1, how='all').columns) 
			- set(['Device', 'Serial Number', 'Device Timestamp', 'Record Type']))
	"""

	# create new dataframe without duplicate rows for each record type in a timestamp
	df = df.groupby('Device Timestamp').first().drop('Record Type', axis=1)

	"""
	df.set_index('Device Timestamp', drop=True, inplace=True)
	df_joinlist = {j : df[df['Record Type'] == j].dropna(axis=1, how='all').drop(['Device', 'Serial Number', 'Record Type'], axis=1) 
		for j in range(int(df['Record Type'].max())+1) if not df[df['Record Type'] == j].empty}
	df_joinstart = df[['Device', 'Serial Number']].reset_index().drop_duplicates().set_index('Device Timestamp', drop=True)
	df_join = df_joinstart.join(df_joinlist.values(), how='outer')
	"""

	df.to_csv(path+'clean/'+str(i)+'.csv')