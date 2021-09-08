import os
import gc
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd

from plot import *
from helper import *
from calc import *
from config import rider_mapping

path = './'
path_tp = '../TrainingPeaks/2019/clean2/'

if not os.path.exists(path+'drop/'):
	os.mkdir(path+'drop/')

# --------------------- (1) read data of first export
df_excel = pd.read_excel(path+'TNN_CGM_2019_EU_exported_20210224.xlsx', sheet_name=None)

# quick clean
df_excel['kamstra'] = df_excel.pop('Sheet4')
df_excel['peron'].drop(0, inplace=True)

for i, df_i in df_excel.items():
	print("\n------------------------------- Athlete ", i)

	print("Riders in the file for athlete %s: %s"%(i, df_i.RIDER.unique()))

	# timestamps
	ts = pd.to_datetime(df_i['Timestamp (YYYY-MM-DDThh:mm:ss)'])
	print("First glucose measurement: ", ts.min())
	print("Last glucose measurement: ", ts.max())
	# event types
	print("Event types: ", df_i['Event Type'].unique())

	# devices
	print("Devices: ", df_i['Source Device ID'].unique())

	# calibration
	print("Number of times Event Type == Calibration",
		(df_i['Event Type'] == 'Calibration').sum())
	print("Number of times Event Type == Calibration and Source Device ID != Android G6 or iPhone G6",
		((df_i['Event Type'] == 'Calibration') & ((df_i['Source Device ID'] != 'Android G6') | df_i['Source Device ID'] != 'iPhone G6')).sum())
	
	print("Fraction of nans")
	print(df_i.isna().sum() / len(df_i))

df = pd.DataFrame(columns=list(df_excel.values())[0].columns)
for i, df_i in df_excel.items():
	df = df.append(df_i)

df.reset_index(drop=True, inplace=True)

# anonymize file
df.RIDER = df.RIDER.apply(lambda x: rider_mapping[x.lower()])

# timestamp
df['Timestamp (YYYY-MM-DDThh:mm:ss)'] = pd.to_datetime(df['Timestamp (YYYY-MM-DDThh:mm:ss)'])
df.rename({'Timestamp (YYYY-MM-DDThh:mm:ss)':'local_timestamp'}, axis=1, inplace=True)

# remove useless and empty columns
df.drop('Index', axis=1, inplace=True)
df.drop(df.columns[df.isna().sum() / len(df) == 1], axis=1, inplace=True)

df['source'] = 'Dexcom CLARITY EU'

# check travels identified with duplicate local timestamps
df[df.duplicated(['RIDER', 'local_timestamp'], keep=False) & (df['Event Type'] == 'EGV')][['RIDER', 'local_timestamp', 'Transmitter Time (Long Integer)', 'Glucose Value (mg/dL)']]

# --------------------- (2) read data of second export (US)
df2 = pd.read_csv('TNN_CGM_2019_US_exported_20210819.csv', index_col=0)

# anonymize file
df2.RIDER = df2.RIDER.apply(lambda x: rider_mapping[x.lower()])

# timestamp
df2['Timestamp (YYYY-MM-DDThh:mm:ss)'] = pd.to_datetime(df2['Timestamp (YYYY-MM-DDThh:mm:ss)'])
df2.rename({'Timestamp (YYYY-MM-DDThh:mm:ss)':'local_timestamp'}, axis=1, inplace=True)

df2['source'] = 'Dexcom CLARITY US'

# check travels identified with duplicate local timestamps
df2[df2.duplicated(['RIDER', 'local_timestamp'], keep=False) & (df2['Event Type'] == 'EGV')][['RIDER', 'local_timestamp', 'Transmitter Time (Long Integer)', 'Glucose Value (mg/dL)']]

# --------------------- merge
# merge
df_merge = pd.merge(df, df2, 
	on=['local_timestamp', 'Event Type', 'Event Subtype', 'Source Device ID',
		'Insulin Value (u)', 'Carb Value (grams)',
		'Duration (hh:mm:ss)', 'Transmitter Time (Long Integer)',
		'Transmitter ID', 'RIDER', 'Glucose Value (mg/dL)'], how='outer')

print("Number of items overlapping: ", (df_merge['source_x'].notna() & df_merge['source_y'].notna()).sum())
print("Number of items US added to EU: ", len(df_merge) - len(df))

df_merge['source'] = df_merge['source_x'].fillna(df_merge['source_y'])
df_merge.drop(['source_x', 'source_y'], axis=1, inplace=True)

df_merge.to_csv('TNN_CGM_2019.csv')

# check if any items with the same timestamp but different sources
dupl = ~df_merge.duplicated(keep=False) & df_merge.duplicated(subset=['local_timestamp', 'RIDER', 'Event Type', 'Event Subtype', 'Transmitter Time (Long Integer)'], keep=False)
df_merge[dupl].sort_values(['RIDER', 'local_timestamp'])
# Here we see that the US values are in mg/dL and the EU values are sometimes in mmol/L