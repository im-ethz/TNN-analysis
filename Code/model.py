# TOOD (shortterm):
# - have validation set
# - make all functions work (also with avg_glucose)
# - once solid model, start summarizing data steps
# - include more data?
# - shift glucose values that we are predicting 5-15 minutes back in time
# - train on data of one athlete
# - train on scan glucose

# TODOS (mid)
# - include dexcom data?
# - Error per time in training (maybe the error is worse in the beginning of training)
# - Visualize train-test split
# - Use past cycling data in prediction (summary of all features up until t)
# - Besides mean, use other statistics in prediction with sliding window
# - Figure out the cause of the similarity in performance between the linear and non-linear models
# - Custom loss function for NN
# - Try out custom nn where the scan glucose is also modelled
# - Add time training as a variable (?) - Should we do this because distance is also a variable
# - Imputation of variables
# - Try out different window sizes for the glucose and for the cycling data
# - Optimize models

# TODOS (longterm)
# - Include feature selection
# - Model with and without imputation
# - Model for each athlete individually and combined
# - IDEA: don't use interpolation but GP/HMM to find true value of data
# - Try out different aggregation (time) intervals
# - Check out how much history of cycling data is needed
# - Should we group athletes so some athletes are completely out-of-sample?
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
from sklearn.base import clone

import keras
from keras.layers import Input, Dense, Concatenate, Dropout, Lambda, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.metrics import RSquare

from plot import *
from helper import *

verbose = 1

K = 5

window_size_features = 15

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

	cols_glucose_h = [('Historic Glucose mg/dL (filled)', 'mean'),
					('Historic Glucose mg/dL (shift-15) (filled)', 'mean'),
					('Historic Glucose mg/dL (shift-30) (filled)', 'mean'),
					('Historic Glucose mg/dL (shift-45) (filled)', 'mean'),
					('Historic Glucose mg/dL (shift-60) (filled)', 'mean')]
	cols_glucose_s = [#('Scan Glucose mg/dL', 'mean'),
					#('Scan Glucose mg/dL (shift-3)', 'mean'),
					#('Scan Glucose mg/dL (shift-5)', 'mean'),
					#('Scan Glucose mg/dL (shift-10)', 'mean'),
					#('Scan Glucose mg/dL (shift-15)', 'mean')
					]

	cols_info 	 = [('time_training', ''), ('file_id', ''), ('athlete', ''),
					('glucose_id', ''), ('glucose_id (shift-15)', ''), ('glucose_id (shift-30)', ''), 
					('glucose_id (shift-45)', ''), ('glucose_id (shift-60)', '')]

	cols_device	 = [('device_ELEMNTBOLT', ''), ('device_zwift', '')] # TODO: add elemntroam

	cols_feature = ['acceleration', 'altitude', 'cadence', 'distance', #'ascent',
					'heart_rate', 'left_pedal_smoothness', 'right_pedal_smoothness',
					'left_torque_effectiveness', 'right_torque_effectiveness', 'left_right_balance',
					'power', 'speed', 'temperature']
	cols_other = set(df_i.columns) - set(cols_info) - set(cols_glucose_h) - set(cols_glucose_s) - set(cols_device) \
	- set([(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']])

	# drop irrelevant cols
	df_i.drop(cols_other - set([('Historic Glucose mg/dL', 'mean')]), axis=1, inplace=True)

	# drop nan glucose
	# if glucose columns not in data, then remove data from that athlete
	try:
		df_i.dropna(how='all', subset=set(cols_glucose_h) ^ set(cols_glucose_s), inplace=True)
	except KeyError as e:
		print("KeyError: ", e)
		continue

	# select one device
	if ('device_zwift', '') in df_i.columns:
		df_i = df_i[df_i[('device_zwift', '')] == 0]
		df_i.drop(cols_device, axis=1, inplace=True)

	# obtain 

	# smooth historic glucose

	# Apply sliding window approach (for each athlete for each training)
	df_i.columns = pd.MultiIndex.from_arrays([df_i.columns.get_level_values(0), df_i.columns.get_level_values(1), df_i.shape[1] * ['t']])
	for t in range(1,window_size_features+1):
		df_shift = df_i[cols_feature].xs('mean', axis=1, level=1, drop_level=False)\
			.xs('t', axis=1, level=2, drop_level=False).shift(periods=-t, freq='min')
		df_shift.rename(columns = {'t':'t-%s'%t}, inplace=True)

		df_i = pd.merge(df_i, df_shift, how='left', left_index=True, right_index=True, validate='one_to_one')

	# TODO: ideally we want to know of the times before the window size how variable it was
	# also: a lot of data is dropped with a window size of 15
	# also: do we want to know of other features besides mean what they were in the 15 min before?
	df_i.dropna(subset=df_i[cols_feature].columns, how='any', inplace=True)

	# drop any item with no glucose
	df_i.dropna(subset=df_i[[x[0] for x in cols_glucose_h]].columns, how='any', inplace=True)

	# add an athlete identifier
	df_i['athlete'] = i

	print(i, len(df_i))
	df = pd.concat([df, df_i])

df['training_id'] = df[[('athlete', '', ''), ('file_id', '', 't')]]\
	.apply(lambda x: str(int(x[0])) + '_' + str(int(x[1])), axis=1)
#df.loc[df.glucose_id.notna(), 'glucose_id_str'] = df.loc[df.glucose_id.notna(), [('athlete', ''), ('file_id', ''), ('glucose_id', '')]]\
#.apply(lambda x: str(int(x[0])) + '_' + str(int(x[1])) + '_' + str(int(x[2])), axis=1)

# select athlete
#df = df[df['athlete'] == 11]
# TODO: make model hierarchical?

df.columns = pd.MultiIndex.from_tuples([(x[0][:22]+' (filled)', x[1], x[2]+x[0][29:32]) if x[0][:30] == 'Historic Glucose mg/dL (shift-' else x for x in df.columns])

# TODO: we can also change this to shift-15
# Because there is a lag in the interstitial glucose (so interstitial glucose(t) ~ blood glucose(t-15))
# we are predicting in the glucose of the past if we predict the historic glucose at t
# Q: How long does it take before physical activity affects the blood glucose levels?
# Is there also some kind of lag in that?
df[('Y', 'mean', 't')] = df[('Historic Glucose mg/dL (filled)', 'mean', 't')] - df[('Historic Glucose mg/dL (filled)', 'mean', 't-15')]
cols_Y = [('Y', 'mean', 't')]
cols_X = df[cols_feature].columns.append(pd.MultiIndex.from_product([['Historic Glucose mg/dL (filled)'], ['mean'], ['t-15', 't-30', 't-45', 't-60']]))

df.reset_index(inplace=True)


# ----------------------------- Models
class NN: # time-delay neural network
	def __init__(self, n_input, n_output, n_hidden, act_hidden, do_hidden,
		optimizer='Adam', loss='mse'):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.act_hidden = act_hidden
		self.do_hidden = do_hidden
		self.n_output = n_output

		self.optimizer = optimizer
		self.loss = loss
		self.model = self.neural_network()

	def neural_network(self):
		inp = Input(shape = (self.n_input,))

		hid = Dense(self.n_hidden[0], activation = self.act_hidden[0])(inp)
		hid = Dropout(self.do_hidden[0])(hid)
		for h in range(1,len(self.n_hidden)):
			hid = Dense(self.n_hidden[h], activation = self.act_hidden[h])(hid)
			hid = Dropout(self.do_hidden[h])(hid)

		out = Dense(self.n_output, activation = 'linear')(hid)

		self.model = Model(inputs = inp, outputs = out)
		self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = RSquare(y_shape=(self.n_output,)))
		return self.model

class LSTM:
	def __init__(self, n_input, n_output, n_hidden, act_hidden, n_timesteps_p, n_features_p, n_timesteps_g, n_features_g, 
		optimizer='Adam', loss='mse'):
		self.n_timesteps_p = n_timesteps_p # window size
		self.n_features_p = n_features_p # features (so n_cols = n_timesteps * n_features)

		self.n_timesteps_g = n_timesteps_g # window size
		self.n_features_g = n_features_g # features (so n_cols = n_timesteps * n_features)
		
		self.n_output = n_output
		self.n_hidden = n_hidden
		self.act_hidden = act_hidden

		self.optimizer = optimizer
		self.loss = loss
		self.model = lstm()

	def lstm(self):
		inp_s = Input(shape = (self.n_input,))
		inp_p = Input(input_shape=(self.n_timesteps_p, self.n_features_p))
		inp_g = Input(input_shape=(self.n_timesteps_g, self.n_features_g))

		s = Dense(self.n_hidden[0], activation = self.act_hidden[0])(inp_s)
		s = Dropout(self.do_hidden[0])(s)
		for h in range(1,len(self.n_hidden)):
			s = Dense(self.n_hidden[h], activation = self.act_hidden[h])(s)
			s = Dropout(self.do_hidden[h])(s)
		s = Dense(10, activation = 'linear')(s)

		p = LSTM(10)(inp_p)
		#p = Dropout(self.do_hidden[0])(p)
		#p = Dense(100, activation='relu')(p)
		p = Dense(10, activation='linear')(p)

		g = LSTM(3)(inp_g)
		#g = Dropout(self.do_hidden[0])(g)
		#p = Dense(100, activation='relu')(g)
		g = Dense(3, activation='linear')(g)

		comb = Concatenate([s, p, g])
		comb = Dense(20, activation='relu')(comb)
		out = Dense(self.n_output, activation='linear')(comb)

		model = Model(inputs=[inp_s, inp_p, inp_g], outputs=out)
		model.compile(optimizer=self.optimizer, loss = self.loss, metrics = RSquare(y_shape=(self.n_output,)))
		return self.model

def get_data(df, cols_X, cols_Y, k, idx_train, idx_test):
	df_train = df.loc[idx_train[k]]
	df_test = df.loc[idx_test[k]]

	X_train = df_train[cols_X]
	X_test = df_test[cols_X]

	Yh_train = df_train[cols_Y]
	Yh_test = df_test[cols_Y]
	#Ys_train = df_train[('Scan Glucose mg/dL', 'mean')]
	#Ys_test = df_test[('Scan Glucose mg/dL', 'mean')]

	Xh_train = X_train[Yh_train.notna().values]
	#Xs_train = X_train[Ys_train.notna().values]
	Xh_test = X_test[Yh_test.notna().values]
	#Xs_test = X_test[Ys_test.notna().values]

	Yh_train = Yh_train.dropna()
	#Ys_train = Ys_train.dropna()
	Yh_test = Yh_test.dropna()
	#Ys_test = Ys_test.dropna()

	perm_h_train = pd.DataFrame(Yh_train.index).to_dict()[0]
	#perm_s_train = pd.DataFrame(Ys_train.index).to_dict()[0]
	perm_h_test = pd.DataFrame(Yh_test.index).to_dict()[0]
	#perm_s_test = pd.DataFrame(Ys_test.index).to_dict()[0]

	# standardize data
	#px = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)
	px = StandardScaler().fit(Xh_train)
	Xh_train = px.transform(Xh_train)
	Xh_test = px.transform(Xh_test)
	#Xs_train = px.transform(Xs_train)
	#Xs_test = px.transform(Xs_test)

	#py = PowerTransformer(method='yeo-johnson', standardize=True).fit(Yh_train)
	py = StandardScaler().fit(Yh_train)
	Yh_train = py.transform(Yh_train).ravel()
	Yh_test = py.transform(Yh_test).ravel()
	#Ys_train = py.transform(Ys_train).ravel()
	#Ys_test = py.transform(Ys_test).ravel()
	# TODO: check if we just standardize
	return Xh_train, Yh_train, Xh_test, Yh_test, perm_h_train, perm_h_test

def glucose_avg(Yh_pred, perm):
	# average the predicted glucose id over the last 15 measurements,
	# as done in the LibreView Historic Glucose
	Yh_avg = df.loc[perm.values(), ['athlete', 'file_id', 'glucose_id']].to_frame()
	Yh_avg['Yh_pred'] = Yh_pred
	Yh_map = Yh_avg.groupby(['athlete', 'file_id', 'glucose_id']).mean().to_dict()['Yh_pred']
	return Yh_avg.apply(lambda x: Yh_map[x[0]], axis=1)

def evaluate(model):
	s = pd.Series(dtype=float)
	
	# mse for historic and scan glucose
	s['mse_h_train'] = mean_squared_error(Yh_train, model.predict(Xh_train))
	#s['mse_s_train'] = mean_squared_error(Ys_train, model.predict(Xs_train))
	s['mse_h_test'] = mean_squared_error(Yh_test, model.predict(Xh_test))
	#s['mse_s_test'] = mean_squared_error(Ys_test, model.predict(Xs_test))

	# r2 for historic and scan glucose
	s['r2_h_train'] = r2_score(Yh_train, model.predict(Xh_train))
	#s['r2_s_train'] = r2_score(Ys_train, model.predict(Xs_train))
	s['r2_h_test'] = r2_score(Yh_test, model.predict(Xh_test))
	#s['r2_s_test'] = r2_score(Ys_test, model.predict(Xs_test))

	# mse and r2 for averaged historic glucose
	#s['mse_h_avg_train'] = mean_squared_error(Yh_train, glucose_avg(model.predict(Xh_train), perm_h_train))
	#s['mse_h_avg_test'] = mean_squared_error(Yh_test, glucose_avg(model.predict(Xh_test), perm_h_test))
	#s['r2_h_avg_train'] = r2_score(Yh_train, glucose_avg(model.predict(Xh_train), perm_h_train))
	#s['r2_h_avg_test'] = r2_score(Yh_test, glucose_avg(model.predict(Xh_test), perm_h_test))
	return s

def plot_history(history, metric):
	plt.plot(history[metric], label='loss')
	plt.plot(history['val_'+metric], label='val_loss')
	plt.xlabel('Epoch')
	plt.ylabel(metric)
	plt.legend()
	plt.show()
	plt.close()

def plot_avg_history(history, metric):
	sns.lineplot(data=history, x='epoch', y=metric, label='loss')
	sns.lineplot(data=history, x='epoch', y='val_'+metric, label='val_loss')
	plt.xlabel('Epoch')
	plt.ylabel(metric)
	plt.legend()
	plt.show()
	plt.close()

# train test split
# split up different file_ids, since we want to be able to predict for a new training
# this means that one file_id stays within one group (either train or test)
# this also means that the 15 outcomes of historic glucose stay within one group (train or test)
# and are not split between the two
# TODO: stratify by athlete
# TODO: also group athletes, so some athletes are completely out-of-sample
idx_train, idx_test = [], []
for idx in GroupKFold(n_splits = K).split(df, groups = df['training_id']):
	idx_train.append(shuffle(idx[0]))
	idx_test.append(shuffle(idx[1]))


# ------------------ Linear
M = {'LinearRegression': [LinearRegression() for _ in range(K)],
	 'Lasso': [Lasso(alpha=1e-3) for _ in range(K)], 
	 'ElasticNet': [ElasticNet(alpha=1e-3, l1_ratio=.5) for _ in range(K)], 
	 #'SVR': [SVR() for _ in range(K)], 
	 #'DecisionTree': [DecisionTreeRegressor(max_depth=30, min_samples_split=0.2) for _ in range(K)], 
	 #'RandomForest': [RandomForestRegressor() for _ in range(K)], 
	 #'ExtraTrees': [ExtraTreesRegressor() for _ in range(K)], 
	 #'AdaBoost': [AdaBoostRegressor() for _ in range(K)], 
	 #'GradientBoost': [GradientBoostingRegressor() for _ in range(K)]
	 }

#score = pd.DataFrame(columns=['mse_h_train', 'mse_h_avg_train', 'mse_s_train', 'mse_h_test' , 'mse_h_avg_test', 'mse_s_test', 
#							  'r2_h_train', 'r2_h_avg_train', 'r2_s_train', 'r2_h_test', 'r2_h_avg_test', 'r2_s_test'])
score = {m: pd.DataFrame(columns=['mse_h_train', 'mse_h_avg_train', 'mse_h_test' , 'mse_h_avg_test',
							  	  'r2_h_train', 'r2_h_avg_train', 'r2_h_test', 'r2_h_avg_test']) for m in M.keys()}
for m in M.keys():
	print(m)
	for k in range(K):
		print(k)
		Xh_train, Yh_train, Xh_test, Yh_test, perm_h_train, perm_h_test = get_data(df, cols_X, cols_Y, k, idx_train, idx_test)

		M[m][k].fit(Xh_train, Yh_train)

		score[m].loc[k] = evaluate(M[m][k])

	print(score[m].mean())

"""
try:
	coef = pd.DataFrame(M[m].coef_, index=cols_X)
except AttributeError:
	coef = pd.DataFrame(M[m].feature_importances_, index=cols_X)
coef['abs'] = coef[0].abs()
coef = coef.sort_values('abs', ascending=False).drop('abs', axis=1)

coef.iloc[:20].plot(kind='barh')
plt.title(m+' coefficients')
plt.axvline(x=0)
plt.subplots_adjust(left=.5)
plt.show()
"""

# ------------------ Neural network
N = {'NeuralNetwork': [NN(Xh_train.shape[1], 1, n_hidden = [50], act_hidden = ['relu'], do_hidden=[0.5, 0.5], optimizer=Adam(learning_rate=1e-3)).model for _ in range(K)]}

score = {n: pd.DataFrame(columns=['mse_h_train', 'mse_h_avg_train', 'mse_h_test' , 'mse_h_avg_test',
							  	  'r2_h_train', 'r2_h_avg_train', 'r2_h_test', 'r2_h_avg_test']) for n in N.keys()}
history = pd.DataFrame()

for n in N.keys():
	print(n)
	for k in range(K):
		print(k)
		Xh_train, Yh_train, Xh_test, Yh_test, perm_h_train, perm_h_test = get_data(df, cols_X, cols_Y, k, idx_train, idx_test)

		# TODO: batch size
		callbacks = [CSVLogger(savedir+'nn/history.log', separator=',', append=False),
					 EarlyStopping(monitor='val_loss', min_delta=0, patience=10),
					 ModelCheckpoint(savedir+'nn/weights.hdf5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)]

		N[n][k].fit(Xh_train, Yh_train, 
			epochs = 200,
			verbose = 1,
			callbacks = callbacks,
			validation_data = (Xh_test, Yh_test))

		N[n][k].load_weights(savedir+'nn/weights.hdf5')
		hist = pd.read_csv(savedir+'nn/history.log', sep=',', engine='python')
		hist['k'] = k
		history = pd.concat([history, hist])

		score[n].loc[k] = evaluate(N[n][k])

		#plot_history(history[history['k'] == k], 'loss')
		#plot_history(history[history['k'] == k], 'r_square')
	print(score[n].mean())

	plot_avg_history(history, 'loss')
	plot_avg_history(history, 'r_square')

"""
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
"""