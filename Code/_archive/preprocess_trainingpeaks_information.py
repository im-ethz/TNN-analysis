# UPDATE: this code is old
# This code is meant to preprocess the _info file from trainingpeaks, after (!) running preprocess_trainingpeaks.py
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

for i in athletes:
	print(i)

	df_info = pd.read_csv(path+'clean/'+str(i)+'/'+str(i)+'_info.csv', header=[0,1], index_col=0)

	# print cols with number of nans
	cols_info0 = df_info.columns.get_level_values(0).unique()
	for col0 in cols_info0:
		print(col0, '\n', df_info[col0].isna().sum(), '\n')

	# ------------------- Clean non-device columns	
	# drop columns that only have zeros in them
	for col in df_info.columns:
		if (df_info[col].dropna() == 0).all():
			df_info.drop(col, axis=1, inplace=True)
			print('DROPPED: ', str(col))

	# ------------------- Clean non-device columns	
	# identify columns with a lot of nans (outside of "device")
	cols_nan = []
	for col in df_info.columns:
		if df_info[col].isna().sum() / df_info.shape[0] > 0.7 and col[0][:6] != 'device':
			cols_nan.append(col)
	print("WARNING: Cols outside of 'device' that high have number of nans, and that we should look at: ", cols_nan)

	# ------------------- Clean device columns	
	cols0_device = [c for c in df_info.columns.get_level_values(0).unique() if c[:6] == 'device']

	# check if device_0 equals file_id device
	print("device_0 manufacturer is file_id manufacturer: ", (df_info[('file_id', 'manufacturer')] == df_info[('device_0', 'manufacturer')]).all())
	print("device_0 product is file_id product: ", (df_info[('file_id', 'product')] == df_info[('device_0', 'product')]).all())

	"""
	# manufacturer
	for col0 in cols0_device:
		print(col0, '\n', df_info[(col0, 'manufacturer')].value_counts(), '\n')

	# product
	for col0 in cols0_device:
		print(col0, '\n', df_info[(col0, 'product')].value_counts(), '\n')

	# product name
	for col0 in cols0_device:
		print(col0, '\n', df_info[(col0, 'product_name')].value_counts(), '\n')

	# combine product info
	for col0 in cols0_device:
		print(col0, '\n', (df_info[(col0, 'manufacturer')] + ' ' + df_info[(col0, 'product_name')]).value_counts(), '\n')
	"""

	def isnan(x):
		return (x != x)

	def product_info(x, col0):
		try:
			manufacturer = x[(col0, 'manufacturer')]
		except KeyError:
			manufacturer = np.nan
		try:
			product_name = x[(col0, 'product_name')]
		except KeyError:
			product_name = np.nan
		if isnan(manufacturer) and isnan(product_name):
			return np.nan
		else:
			return str(manufacturer) + ' ' + str(product_name)

	df_info[('device_summary', '0')] = df_info.apply(lambda x: product_info(x, 'device_0'), axis=1)
	df_info[('device_summary', '1')] = df_info.apply(lambda x: sorted([product_info(x, col0) for col0 in cols0_device[1:]\
													if not isnan(product_info(x, col0))]), axis=1)

	df_info.to_csv(path+'clean/'+str(i)+'/'+str(i)+'_info_clean.csv')