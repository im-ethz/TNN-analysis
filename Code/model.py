# List of TODOS - overview
# - Model with and without imputation
# - Model for each athlete individually and combined
# - IDEA: don't use interpolation but GP/HMM to find true value of data
# - Try out different aggregation (time) intervals
# - Predict using diff of glucose levels
# - Use past glucose levels in prediction
# - Use past 15 minutes of cycling data in prediction
# - TODO: use tsextract for sliding windows
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import os
import gc
import matplotlib

from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.utils import shuffle

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score

import keras
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda, LSTM
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare

from plot import *
from helper import *

verbose = 1

K = 5

path = 'Data/TrainingPeaks+LibreView/'
savedir = 'Results/'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'1min/dropna/') if i.endswith('.csv')])

# read data
df = pd.DataFrame()
for i in athletes:
	df_i = pd.read_csv(path+'1min/dropna/'+str(i)+'.csv', header=[0,1], index_col=0)
	df_i.index = pd.to_datetime(df_i.index)

	# rename some columns
	df_i.rename(columns={'first':''}, inplace=True)

	cols_glucose = [('Historic Glucose mg/dL (filled)', 'mean'),
					('Scan Glucose mg/dL', 'mean')]

	cols_info 	 = [('time_training', ''), ('file_id', ''), ('glucose_id', ''), ('athlete', '')]

	cols_device	 = [('device_ELEMNTBOLT', ''), ('device_zwift', '')] # TODO: add elemntroam

	cols_feature = ['acceleration', 'altitude', 'cadence', 'distance', #'ascent',
					'heart_rate', 'left_pedal_smoothness', 'right_pedal_smoothness',
					'left_torque_effectiveness', 'right_torque_effectiveness', 'left_right_balance',
					'power', 'speed', 'temperature']
	cols_other = set(df_i.columns) - set(cols_info) - set(cols_glucose) - set(cols_device) \
	- set([(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']])

	# drop irrelevant cols
	df_i.drop(cols_other - set([('Historic Glucose mg/dL', 'mean')]), axis=1, inplace=True)

	# drop nan glucose
	# if glucose columns not in data, then remove data from that athlete
	try:
		df_i.dropna(how='all', subset=cols_glucose, inplace=True)
	except KeyError as e:
		print("KeyError: ", e)
		continue

	# select one device
	if ('device_zwift', '') in df_i.columns:
		df_i = df_i[df_i[('device_zwift', '')] == 0]
		df_i.drop(cols_device, axis=1, inplace=True)

	# obtain 

	# smooth historic glucose

	print(i, len(df_i))
	# add an athlete identifier
	df_i['athlete'] = i
	df = pd.concat([df, df_i])

df['training_id'] = df[[('athlete', ''), ('file_id', '')]]\
	.apply(lambda x: str(int(x[0])) + '_' + str(int(x[1])), axis=1)
df.loc[df.glucose_id.notna(), 'glucose_id_str'] = df.loc[df.glucose_id.notna(), [('athlete', ''), ('file_id', ''), ('glucose_id', '')]]\
.apply(lambda x: str(int(x[0])) + '_' + str(int(x[1])) + '_' + str(int(x[2])), axis=1)

# select athlete
df = df[df['athlete'] == 11]
# TODO: make model hierarchical?

# ----------------------------- Non-timeseries models
class TDNN: # time-delay neural network
	def __init__(self, n_input, n_output, n_hidden, act_hidden, 
		optimizer='Adam', loss='mse'):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.act_hidden = act_hidden
		self.n_output = n_output

		self.optimizer = optimizer
		self.loss = loss
		self.model = self.neural_network()

	def neural_network(self):
		inp = Input(shape = (self.n_input,))

		hid = Dense(self.n_hidden[0], activation = self.act_hidden[0])(inp)
		for h in range(1,len(self.n_hidden)):
			hid = Dense(self.n_hidden[h], activation = self.act_hidden[h])(hid)

		out = Dense(self.n_output, activation = 'linear')(hid)

		self.model = Model(inputs = inp, outputs = out)
		self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = RSquare(y_shape=(self.n_output,)))
		return self.model

class LSTM:
	def __init__(self, n_timesteps, n_features, n_output, n_hidden, act_hidden,
		optimizer='Adam', loss='mse'):
		self.n_timesteps = n_timesteps # window size
		self.n_features = n_features # features (so n_cols = n_timesteps * n_features)
		
		self.n_output = n_output
		self.n_hidden = n_hidden
		self.act_hidden = act_hidden

		self.optimizer = optimizer
		self.loss = loss
		self.model = lstm()

	def lstm(self):
		model = Sequential()
		model.add(LSTM(h_hidden[0], input_shape=(n_timesteps, n_features)))
		model.add(Dropout(0.5))
		model.add(Dense(100, activation='relu'))
		model.add(Dense(n_output, activation='linear'))
		model.compile(optimizer=self.optimizer, loss = self.loss, metrics = RSquare(y_shape=(self.n_output,)))
		return self.model

def glucose_avg(Yh_pred, perm):
	# average the predicted glucose id over the last 15 measurements,
	# as done in the LibreView Historic Glucose
	Yh_avg = df.loc[perm.values(), 'glucose_id_str'].to_frame()
	Yh_avg['Yh_pred'] = Yh_pred
	Yh_map = Yh_avg.groupby('glucose_id_str').mean().to_dict()['Yh_pred']
	return Yh_avg.apply(lambda x: Yh_map[x[0]], axis=1)

def plot_history(history, metric):
	plt.plot(history.history[metric], label='loss')
	plt.plot(history.history['val_'+metric], label='val_loss')
	plt.xlabel('Epoch')
	plt.ylabel(metric)
	plt.legend()
	plt.show()
	plt.close()

df.reset_index(inplace=True)

# train test split
# split up different file_ids, since we want to be able to predict for a new training
# this means that one file_id stays within one group (either train or test)
# this also means that the 15 outcomes of historic glucose stay within one group (train or test)
# and are not split between the two
# TODO: stratify by athlete
idx_train, idx_test = [], []
for idx in GroupKFold(n_splits = K).split(df, groups = df['training_id']):
	idx_train.append(shuffle(idx[0]))
	idx_test.append(shuffle(idx[1]))

for k in range(K):
	df_train = df.loc[idx_train[k]]
	df_test = df.loc[idx_test[k]]

	X_train = df_train[[(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']]]
	X_test = df_test[[(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']]]

	Yh_train = df_train[('Historic Glucose mg/dL (filled)', 'mean')]
	Yh_test = df_test[('Historic Glucose mg/dL (filled)', 'mean')]
	Ys_train = df_train[('Scan Glucose mg/dL', 'mean')]
	Ys_test = df_test[('Scan Glucose mg/dL', 'mean')]

	Xh_train = X_train[Yh_train.notna()]
	Xs_train = X_train[Ys_train.notna()]
	Xh_test = X_test[Yh_test.notna()]
	Xs_test = X_test[Ys_test.notna()]

	Yh_train = Yh_train.dropna().to_frame()
	Ys_train = Ys_train.dropna().to_frame()
	Yh_test = Yh_test.dropna().to_frame()
	Ys_test = Ys_test.dropna().to_frame()

	perm_h_train = pd.DataFrame(Yh_train.index).to_dict()[0]
	perm_s_train = pd.DataFrame(Ys_train.index).to_dict()[0]
	perm_h_test = pd.DataFrame(Yh_test.index).to_dict()[0]
	perm_s_test = pd.DataFrame(Ys_test.index).to_dict()[0]

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
	# TODO: custom loss function as well for this scoring
	# TODO: in some way for a nn combine the two outcomes for one loss function
	# TODO: fix cross-validation
	# TODO: error per time in training (maybe error is worse in beginning of training)
	# TODO: add training time as variable

	M = {'LinearRegression': LinearRegression(),
		 'Lasso': Lasso(alpha=1.), 
		 'ElasticNet': ElasticNet(alpha=1., l1_ratio=.5), 
		 #'SVR': SVR(), 
		 #'DecisionTree': DecisionTreeRegressor(), 
		 #'RandomForest': RandomForestRegressor(), 
		 #'ExtraTrees': ExtraTreesRegressor(), 
		 #'AdaBoost': AdaBoostRegressor(), 
		 #'GradientBoost': GradientBoostingRegressor()
		 }

	score = pd.DataFrame(columns=['mse_h_train', 'mse_h_avg_train', 'mse_s_train', 'mse_h_test' , 'mse_h_avg_test', 'mse_s_test', 
								  'r2_h_train', 'r2_h_avg_train', 'r2_s_train', 'r2_h_test', 'r2_h_avg_test', 'r2_s_test'])
	for m in M.keys():
		print(m)
		M[m].fit(Xh_train, Yh_train)

		# mse for historic and scan glucose
		score.loc[m, 'mse_h_train'] = mean_squared_error(Yh_train, M[m].predict(Xh_train))
		score.loc[m, 'mse_s_train'] = mean_squared_error(Ys_train, M[m].predict(Xs_train))
		score.loc[m, 'mse_h_test'] = mean_squared_error(Yh_test, M[m].predict(Xh_test))
		score.loc[m, 'mse_s_test'] = mean_squared_error(Ys_test, M[m].predict(Xs_test))

		# r2 for historic and scan glucose
		score.loc[m, 'r2_h_train'] = r2_score(Yh_train, M[m].predict(Xh_train))
		score.loc[m, 'r2_s_train'] = r2_score(Ys_train, M[m].predict(Xs_train))
		score.loc[m, 'r2_h_test'] = r2_score(Yh_test, M[m].predict(Xh_test))
		score.loc[m, 'r2_s_test'] = r2_score(Ys_test, M[m].predict(Xs_test))

		# mse and r2 for averaged historic glucose
		score.loc[m, 'mse_h_avg_train'] = mean_squared_error(Yh_train, glucose_avg(M[m].predict(Xh_train), perm_h_train))
		score.loc[m, 'mse_h_avg_test'] = mean_squared_error(Yh_test, glucose_avg(M[m].predict(Xh_test), perm_h_test))
		score.loc[m, 'r2_h_avg_train'] = r2_score(Yh_train, glucose_avg(M[m].predict(Xh_train), perm_h_train))
		score.loc[m, 'r2_h_avg_test'] = r2_score(Yh_test, glucose_avg(M[m].predict(Xh_test), perm_h_test))

		print(score.loc[m])

	# neural network
	# TODO: batch size
	callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=20)]

	N = {'NeuralNetwork': NN(Xh_train.shape[1], 1, n_hidden = [5], act_hidden = ['tanh']).model}

	for n in N.keys():
		print(n)
		N[n].fit(Xh_train, Yh_train, 
			epochs = 100,
			verbose = 1,
			callbacks = callbacks,
			validation_data = (Xh_test, Yh_test))

		plot_history(N[n].history, 'loss')
		plot_history(N[n].history, 'r_square')

	# lstm
	L = {'LSTM': LSTM()}
	for l in L.keys():
		print(l)
		L[l].fit(Xh_train, Yh_train, 
			epochs = epochs,
			verbose = 1,
			callbacks = callbacks,
			validation_data = (Xh_test, Yh_test))

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