# List of TODOS - overview
# - Model with and without imputation
# - Model for each athlete individually and combined
# - IDEA: don't use interpolation but GP/HMM to find true value of data
# - Try out different aggregation (time) intervals
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import os
import gc
import matplotlib

from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.model_selection import GroupShuffleSplit

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

from plot import *
from helper import *

verbose = 1

path = 'Data/TrainingPeaks+LibreView/'
savedir = 'Results/'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'1min/dropna/') if i.endswith('.csv')])

# read data
df = pd.DataFrame()
for i in athletes:
	df_i = pd.read_csv(path+'1min/dropna/'+str(i)+'.csv', header=[0,1], index_col=0)
	df_i.index = pd.to_datetime(df_i.index)

	cols_glucose = [('Historic Glucose mg/dL (filled)', 'mean'),
					('Scan Glucose mg/dL', 'mean')]
	
	cols_info 	 = [('time_training', 'first'), ('file_id', 'first')]
	
	cols_device	 = [('device_ELEMNTBOLT', 'first'), ('device_zwift', 'first')] # TODO: add elemntroam

	cols_feature = ['acceleration', 'altitude', 'cadence', 'distance', #'ascent',
					'heart_rate', 'left_pedal_smoothness', 'right_pedal_smoothness',
					'left_torque_effectiveness', 'right_torque_effectiveness', 'left_right_balance',
					'power', 'speed', 'temperature']
	cols_other = set(df_i.columns) - set(cols_info) - set(cols_glucose) - set(cols_device) \
	- set([(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']])

	# drop irrelevant cols
	df_i.drop(cols_other, axis=1, inplace=True)

	# drop nan glucose
	try:
		df_i.dropna(how='all', subset=cols_glucose, inplace=True)
	except KeyError as e:
		if str(e) == str([cols_glucose[0][0]]):
			print("KeyError: ", e)
			continue

	# select one device
	if ('device_zwift', 'first') in df_i.columns:
		df_i = df_i[df_i[('device_zwift', 'first')] == 0]
		df_i.drop(cols_device, axis=1, inplace=True)

	# smooth historic glucose

	df_i['athlete'] = i
	df = pd.concat([df, df_i])

df['training_id'] = df[[('athlete', ''), ('file_id', 'first')]].apply(lambda x: str(int(x[0])) + '_' + str(int(x[1])), axis=1)

# ----------------------------- Non-timeseries models
df.reset_index(inplace=True)

X = df[[(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']]]

Yh = df[('Historic Glucose mg/dL (filled)', 'mean')]
Xh = X[Yh.notna()]
Yh = Yh.dropna()

Ys = df[('Scan Glucose mg/dL', 'mean')]
Xs = X[Ys.notna()]
Ys = Ys.dropna()

# train test split
# split up different file_ids, since we want to be able to predict for a new training
# this means that one file_id stays within one group (either train or test)
# this also means that the 15 outcomes of historic glucose stay within one group (train or test)
# and are not split between the two
# TODO: stratify by athlete
idx_train, idx_test = next(GroupShuffleSplit(n_splits = 1).split(df, groups = df['training_id']))

Xh_train = Xh[Xh.index.isin(idx_train)]
Yh_train = Yh[Yh.index.isin(idx_train)].to_frame()
Xh_test = Xh[Xh.index.isin(idx_test)]
Yh_test = Yh[Yh.index.isin(idx_test)].to_frame()

Xs_train = Xs[Xs.index.isin(idx_train)]
Ys_train = Ys[Ys.index.isin(idx_train)].to_frame()
Xs_test = Xs[Xs.index.isin(idx_test)]
Ys_test = Ys[Ys.index.isin(idx_test)].to_frame()

# standardize data
#px = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)
px = StandardScaler().fit(Xh_train)
Xh_train = px.transform(Xh_train)
Xh_test = px.transform(Xh_test)
Xs_train = px.transform(Xs_train)
Xs_test = px.transform(Xs_test)

#py = PowerTransformer(method='yeo-johnson', standardize=True).fit(Yh_train)
py = StandardScaler().fit(Yh_train)
Yh_train = py.transform(Yh_train).ravel()
Yh_test = py.transform(Yh_test).ravel()
Ys_train = py.transform(Ys_train).ravel()
Ys_test = py.transform(Ys_test).ravel()
# TODO: check if we just standardize

# todo: include feature selection

# TODO: check why the non-linear models aren't working

M = {'LinearRegression': LinearRegression(), 
	 'ElasticNet': ElasticNet(), 
	 'SVR': SVR(), 
	 'DecisionTree': DecisionTreeRegressor(), 
	 'RandomForest': RandomForestRegressor(), 
	 'ExtraTrees': ExtraTreesRegressor(), 
	 'AdaBoost': AdaBoostRegressor(), 
	 'GradientBoost': GradientBoostingRegressor()}

score = pd.DataFrame(columns=['mse_h_train', 'mse_h_test' , 'mse_s_train', 'mse_s_test', 
							  'r2_h_train', 'r2_h_test', 'r2_s_train', 'r2_s_test'])
for m in M.keys():
	print(m)
	M[m].fit(Xh_train, Yh_train)

	score.loc[m, 'mse_h_train'] = mean_squared_error(py.inverse_transform(Yh_train), 
													 py.inverse_transform(M[m].predict(Xh_train)))
	score.loc[m, 'mse_s_train'] = mean_squared_error(py.inverse_transform(Ys_train), 
													 py.inverse_transform(M[m].predict(Xs_train)))
	score.loc[m, 'mse_h_test'] = mean_squared_error(py.inverse_transform(Yh_test), 
													py.inverse_transform(M[m].predict(Xh_test)))
	score.loc[m, 'mse_s_test'] = mean_squared_error(py.inverse_transform(Ys_test), 
													py.inverse_transform(M[m].predict(Xs_test)))

	score.loc[m, 'r2_h_train'] = r2_score(py.inverse_transform(Yh_train), 
										  py.inverse_transform(M[m].predict(Xh_train)))
	score.loc[m, 'r2_s_train'] = r2_score(py.inverse_transform(Ys_train), 
										  py.inverse_transform(M[m].predict(Xs_train)))
	score.loc[m, 'r2_h_test'] = r2_score(py.inverse_transform(Yh_test), 
										 py.inverse_transform(M[m].predict(Xh_test)))
	score.loc[m, 'r2_s_test'] = r2_score(py.inverse_transform(Ys_test), 
										 py.inverse_transform(M[m].predict(Xs_test)))
	print(score.loc[m])
best_m = score['r2'].argmax()

plt.figure()
plt.plot(np.unique(Y_test), np.unique(Y_test)*np.corrcoef(Y_pred, Y_test)[0,1])
plt.scatter(Y_test, Y_pred, s=.1, alpha=.7)
plt.xlabel('True '+cols_glucose[0][0])
plt.ylabel('Predicted '+cols_glucose[0][0])
plt.show()

# Gaussian process

# ----------------------------- Non-timeseries models
# Standard

# Latent variable model
# Hidden markov?