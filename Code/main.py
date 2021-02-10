# TOOD (shortterm):
# - implement simple baseline
# - implement shap
# - plot coefficients
# - include scan glucose
# - once solid model, start summarizing data steps
# - include more data?
# - shift glucose values that we are predicting 5-15 minutes back in time
# - train on data of one athlete
# - train on scan glucose

# TODOS (mid)
# - include dexcom data?
# - Error per time in training (maybe the error is worse in the beginning of training)
# - Error per athlete
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

from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.utils import shuffle

from model import *
from plot import *
from helper import *

verbose = 1

K = 5

window_size_features = 15

path = 'Data/TrainingPeaks+LibreView/'
savedir = 'Results/'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'1min/dropna/') if i.endswith('.csv')])

cols_glucose_h = [('Historic Glucose mg/dL (filled)', 'mean')]
cols_glucose_s = [('Scan Glucose mg/dL', 'mean')]
cols_glucose_b = [('Bubble Glucose mg/dL', 'mean')]

cols_glucose = cols_glucose_h + cols_glucose_s + cols_glucose_b

cols_glucose_previous = [('Historic Glucose mg/dL (shift-15) (filled)', 'mean'),
				  		('Historic Glucose mg/dL (shift-30) (filled)', 'mean'),
				  		('Historic Glucose mg/dL (shift-45) (filled)', 'mean'),
				  		('Historic Glucose mg/dL (shift-60) (filled)', 'mean')]

cols_info 	 = [('time_training', ''), ('file_id', ''), ('athlete', ''),
				('glucose_id', ''), ('glucose_id (shift-15)', ''), ('glucose_id (shift-30)', ''), 
				('glucose_id (shift-45)', ''), ('glucose_id (shift-60)', '')]

cols_device	 = [('device_ELEMNTBOLT', ''), ('device_zwift', '')] # TODO: add elemntroam

cols_feature = ['acceleration', 'altitude', 'cadence', 'distance', #'ascent',
				'heart_rate', 'left_pedal_smoothness', 'right_pedal_smoothness',
				'left_torque_effectiveness', 'right_torque_effectiveness', 'left_right_balance',
				'power', 'speed', 'temperature']

# read data
df = pd.DataFrame()
for i in athletes:
	print("\n------------------------------- Athlete ", i)
	df_i = pd.read_csv(path+'1min/dropna/'+str(i)+'.csv', header=[0,1], index_col=0)
	df_i.index = pd.to_datetime(df_i.index)

	# rename some columns
	df_i.rename(columns={'first':''}, inplace=True)

	# drop irrelevant cols
	cols_other = set(df_i.columns) - set(cols_info) - set(cols_glucose) - set(cols_glucose_previous) - set(cols_device) \
	- set([(c,x) for c in cols_feature for x in ['iqr', 'mean', 'median', 'minmax', 'std', 'sum']])
	df_i.drop(cols_other - set([('Historic Glucose mg/dL', 'mean')]), axis=1, inplace=True)

	print("T=", len(df_i.file_id.unique()), "N=", len(df_i))

	# select one device
	if ('device_zwift', '') in df_i.columns:
		df_i = df_i[df_i[('device_zwift', '')] == 0]
		df_i.drop(cols_device, axis=1, inplace=True)
	print("Remove virtual races")
	print("T=", len(df_i.file_id.unique()), "N=", len(df_i))

	# drop nan glucose
	# if glucose columns not in data, then remove data from that athlete
	df_i.dropna(how='all', subset=set(df_i.columns) & set(cols_glucose), inplace=True)
	print("Drop samples without any glucose values")
	print("T=", len(df_i.file_id.unique()), "N=", len(df_i))

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
	print("Drop samples without any feature values")
	print("T=", len(df_i.file_id['t'].unique()), "N=", len(df_i))

	# TODO smooth historic glucose
	# drop any item with no shifted glucose
	df_i.dropna(subset=df_i[[x[0] for x in cols_glucose_previous]].columns, how='any', inplace=True)
	print("Drop samples without any historic (shifted) glucose values")
	print("T=", len(df_i.file_id['t'].unique()), "N=", len(df_i))

	# add an athlete identifier
	df_i['athlete'] = i

	print(i, len(df_i))
	df = pd.concat([df, df_i])

df['training_id'] = df[[('athlete', '', ''), ('file_id', '', 't')]]\
	.apply(lambda x: str(int(x[0])) + '_' + str(int(x[1])), axis=1)

# select athlete
#df = df[df['athlete'] == 11]
# TODO: make model hierarchical?

df.columns = pd.MultiIndex.from_tuples([(x[0][:22]+' (filled)', x[1], x[2]+x[0][29:32]) if x[0][:30] == 'Historic Glucose mg/dL (shift-' else x for x in df.columns])
#df.columns = pd.MultiIndex.from_tuples([(x[0][:18], x[1], x[2]+x[0][25:28]) if x[0][:25] == 'Scan Glucose mg/dL (shift' else x for x in df.columns])

# TODO: we can also change this to shift-15
# Because there is a lag in the interstitial glucose (so interstitial glucose(t) ~ blood glucose(t-15))
# we are predicting in the glucose of the past if we predict the historic glucose at t
# Q: How long does it take before physical activity affects the blood glucose levels?
# Is there also some kind of lag in that?
cols_X = df[cols_feature].columns.append(pd.MultiIndex.from_product([['Historic Glucose mg/dL (filled)'], ['mean'], ['t-15', 't-30', 't-45', 't-60']]))

#df[('Historic Glucose mg/dL (diff)', 'mean', 't')] = df[('Historic Glucose mg/dL (filled)', 'mean', 't')] - df[('Historic Glucose mg/dL (filled)', 'mean', 't-15')]
#df[('Scan Glucose mg/dL (diff)', 'mean', 't')] = df[('Scan Glucose mg/dL', 'mean', 't')] - df[('Scan Glucose mg/dL', 'mean', 't-15')]
cols_Y = [('Historic Glucose mg/dL (filled)', 'mean', 't'), ('Scan Glucose mg/dL', 'mean', 't'), ('Bubble Glucose mg/dL', 'mean', 't')]#[('Historic Glucose mg/dL (diff)', 'mean', 't')]
#cols_Ys = [('Scan Glucose mg/dL (diff)', 'mean', 't')]

types = [c[0].rstrip(" (filled)").rstrip("lucose mg/dL")[:-2] for c in cols_Y]

df.reset_index(inplace=True)

# plot histogram of glucose
PlotData('Descriptives/').plot_hist_glucose(df, cols_Y)

"""
# diff glucose at t and t-15
#ax = sns.histplot(df[('Scan Glucose mg/dL (diff)')], label='Scan', 
#	stat='density', kde=True,
#	alpha=0.3, line_kws={'lw':2.})
#ax = change_color_histplot(ax, 2)
sns.histplot(df[('Historic Glucose mg/dL (diff)', 'mean', 't')], label='Historic', 
	stat='density', kde=True, 
	alpha=0.3, line_kws={'lw':2.})
plt.xlabel('Glucose mg/dL (diff)')
plt.legend()
plt.show()
"""

# ----------------------------- Models
def get_data(df, cols_X, cols_Y, k, idx_train, idx_val):
	X_train =  df.loc[idx_train[k]][cols_X]
	Y_train =  df.loc[idx_train[k]][cols_Y]

	X_val = df.loc[idx_val[k]][cols_X]
	Y_val = df.loc[idx_val[k]][cols_Y]

	perm_train = pd.DataFrame(Y_train.index).to_dict()[0]
	perm_val = pd.DataFrame(Y_val.index).to_dict()[0]

	# standardize data
	#px = PowerTransformer(method='yeo-johnson', standardize=True).fit(X_train)
	px = StandardScaler().fit(X_train[Y_train[cols_Y[0]].notna()])
	X_train = px.transform(X_train)
	X_val = px.transform(X_val)

	#py = PowerTransformer(method='yeo-johnson', standardize=True).fit(Yh_train)
	py = StandardScaler().fit(Y_train[Y_train[cols_Y[0]].notna()])
	Y_train = py.transform(Y_train)
	Y_val = py.transform(Y_val)
	# TODO: check if we just standardize

	X_train_arr, Y_train_arr, X_val_arr, Y_val_arr, perm_train_arr, perm_val_arr = [], [], [], [], [], []
	for c in range(len(cols_Y)):
		X_train_arr.append(X_train[~np.isnan(Y_train[:,c])])
		Y_train_arr.append(Y_train[~np.isnan(Y_train[:,c])][:,c])
		perm_train_arr.append(dict(p for p in perm_train.items() if not np.isnan(Y_train[:,c])[p[0]]))

		X_val_arr.append(X_val[~np.isnan(Y_val[:,c])])
		Y_val_arr.append(Y_val[~np.isnan(Y_val[:,c])][:,c])
		perm_val_arr.append(dict(p for p in perm_val.items() if not np.isnan(Y_val[:,c])[p[0]]))

	return X_train_arr, Y_train_arr, X_val_arr, Y_val_arr, perm_train_arr, perm_val_arr

def glucose_avg(Yh_pred, perm):
	# average the predicted glucose id over the last 15 measurements,
	# as done in the LibreView Historic Glucose
	Yh_avg = df.loc[perm.values(), ['athlete', 'file_id', 'glucose_id']]
	Yh_avg.columns = Yh_avg.columns.get_level_values(0)
	Yh_avg['Yh_pred'] = Yh_pred
	Yh_map = Yh_avg.groupby(['athlete', 'file_id', 'glucose_id']).mean().to_dict()['Yh_pred']
	return Yh_avg.apply(lambda x: Yh_map[(x[0], x[1], x[2])], axis=1)

def calc_score(model, X, Y, perm=None, historic=False):
	Y_pred = model.predict(X)
	if historic:
		Y_pred = glucose_avg(Y_pred, perm)

	mse = mean_squared_error(Y, Y_pred)
	r2 = r2_score(Y, Y_pred)
	return mse, r2

def evaluate(model, data, cols_Y=cols_Y):
	# data of shape {'train': (X_train, Y_train, perm_train), 'val': (X_val, Y_val, perm_val)}
	train, val = tuple(data)
	(X_train, Y_train, perm_train), (X_val, Y_val, perm_val) = data[train], data[val]

	types = [c[0].rstrip(" (filled)").rstrip("lucose mg/dL")[:-2] for c in cols_Y]

	s = pd.Series(index=pd.MultiIndex.from_product([['mse', 'r2'], types+['Historic (avg)'], [train, val]]), dtype=float)
	
	for i, t in enumerate(types):
		s[('mse', t, train)], s[('r2', t, train)] = calc_score(model, X_train[i], Y_train[i])
		s[('mse', t, val)], s[('r2', t, val)] = calc_score(model, X_val[i], Y_val[i])
		if t == 'Historic':
			s[('mse', t+' (avg)', train)], s[('r2', t+' (avg)', train)] = calc_score(model, X_train[i], Y_train[i], perm_train[i], historic=True)
			s[('mse', t+' (avg)', val)], s[('r2', t+' (avg)', val)] = calc_score(model, X_val[i], Y_val[i], perm_val[i], historic=True)
	return s

# train-val-test split
# split up grouping on different file_ids, since we want to be able to predict for a new training
# this means that one file_id stays within one group (either train or test)
# this also means that the 15 outcomes of historic glucose stay within one group (train or test)
# and are not split between the two
# TODO: stratify by athlete
# TODO: also group athletes, so some athletes are completely out-of-sample
# TODO: outer-fold cv
idx_trainval, idx_test = list(GroupShuffleSplit(n_splits=1, train_size=.8).split(df, groups=df['training_id']))[0]

idx_train, idx_val = [], []
for idx in GroupKFold(n_splits = K).split(df.loc[idx_trainval], groups = df.loc[idx_trainval, 'training_id']):
	idx_train.append(shuffle(idx_trainval[idx[0]]))
	idx_val.append(shuffle(idx_trainval[idx[1]]))

# visualize data split
PlotData('Descriptives/').plot_data_split(df, idx_val, idx_test)

# ------------------ Basic baseline
# predicting based on previous glucose value
def basic_baseline(Y, perm, name):
	# TODO: not the best way to get glucose values
	# TODO: maybe if there's a nan, just take a mean?

	# combine:
	cols = ['athlete', 'file_id']
	if name == 'Historic':
		cols += ['glucose_id']
	Y_pred = df.loc[perm.values(), cols]
	Y_pred.columns = Y_pred.columns.get_level_values(0)
	Y_pred['Y_pred'] = Y
	if name == 'Historic':
		Y_map = Y_pred.groupby(cols).mean()
	else:
		Y_map = Y_pred.sort_values(cols).set_index(cols)
	Y_map['Y_pred (shift)'] = np.nan
	for a, f in zip(Y_map.index.get_level_values(0), Y_map.index.get_level_values(1)):
		try:
			Y_map.loc[a].loc[f]['Y_pred (shift)'] = Y_map.loc[a].loc[f]['Y_pred'].shift(1)
		except AttributeError:
			continue
	Y_map = Y_map.to_dict()['Y_pred (shift)']
	Y_pred = Y_pred.apply(lambda x: Y_map[tuple(x[:len(cols)])], axis=1)
	Y = Y[Y_pred.notna()]
	Y_pred = Y_pred.dropna().values
	return Y, Y_pred

X_trainval, Y_trainval, X_test, Y_test, perm_trainval, perm_test = get_data(df, cols_X, cols_Y, 0, [idx_trainval], [idx_test])

score = pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'], types+['Historic (avg)'], ['trainval', 'test']]))

score.loc['BasicBaseline'] = np.nan
for i, t in enumerate(types):
	score.loc['BasicBaseline'][('mse', t, 'trainval')] = mean_squared_error(*basic_baseline(Y_trainval[i], perm_trainval[i], t))
	score.loc['BasicBaseline'][('r2', t, 'trainval')] = r2_score(*basic_baseline(Y_trainval[i], perm_trainval[i], t))
	score.loc['BasicBaseline'][('mse', t, 'test')] = mean_squared_error(*basic_baseline(Y_test[i], perm_test[i], t))
	score.loc['BasicBaseline'][('r2', t, 'test')] = r2_score(*basic_baseline(Y_test[i], perm_test[i], t))

# ------------------ Linear
M = {'LinearRegression': [LinearRegression() for _ in range(K+1)],
	 #'Lasso': [Lasso(alpha=1e-3) for _ in range(K+1)], 
	 #'ElasticNet': [ElasticNet(alpha=1e-3, l1_ratio=.5) for _ in range(K+1)], 
	 #'SVR': [SVR() for _ in range(K+1)], 
	 #'DecisionTree': [DecisionTreeRegressor(max_depth=30, min_samples_split=0.2) for _ in range(K+1)], 
	 #'RandomForest': [RandomForestRegressor() for _ in range(K+1)], 
	 #'ExtraTrees': [ExtraTreesRegressor() for _ in range(K+1)], 
	 #'AdaBoost': [AdaBoostRegressor() for _ in range(K+1)], 
	 #'GradientBoost': [GradientBoostingRegressor() for _ in range(K+1)]
	 }

cv_score = {m: pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'], types+['Historic (avg)'], ['train', 'val']])) for m in M.keys()}
for m in M.keys():
	print(m)
	for k in range(K):
		print(k)
		X_train, Y_train, X_val, Y_val, perm_train, perm_val = get_data(df, cols_X, cols_Y, k, idx_train, idx_val)

		M[m][k].fit(X_train[0], Y_train[0])

		cv_score[m].loc[k] = evaluate(M[m][k], {'train': (X_train, Y_train, perm_train), 'val': (X_val, Y_val, perm_val)})

	print(cv_score[m].mean())

	M[m][k+1].fit(X_trainval[0], Y_trainval[0])
	score.loc[m] = evaluate(M[m][k+1], {'trainval': (X_trainval, Y_trainval, perm_trainval), 'test': (X_test, Y_test, perm_test)})
	print(score)

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


# ------------------ Neural network
N = {'NeuralNetwork': [NN(len(cols_X), 1, n_hidden = [50], act_hidden = ['relu'], do_hidden=[0.5, 0.5], optimizer=Adam(learning_rate=1e-3)).model for _ in range(K+1)]}

cv_score.update({n: pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'],  types+['Historic (avg)'], ['train', 'val']])) for n in N.keys()})
#score = pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'], ['hist', 'scan'], ['test']]))
history = pd.DataFrame()

for n in N.keys():
	print(n)
	if not os.path.exists(savedir+n):
		os.mkdir(savedir+n)
	for k in range(K):
		print(k)
		X_train, Y_train, X_val, Y_val, perm_train, perm_val = get_data(df, cols_X, cols_Y, k, idx_train, idx_val)

		# TODO: batch size
		callbacks = [CSVLogger(savedir+n+'/history.log', separator=',', append=False),
					 EarlyStopping(monitor='val_loss', min_delta=0, patience=10),
					 ModelCheckpoint(savedir+n+'/weights.hdf5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)]

		N[n][k].fit(X_train[0], Y_train[0], 
			epochs = 200,
			verbose = 1,
			callbacks = callbacks,
			validation_data = (X_val[0], Y_val[0]))

		N[n][k].load_weights(savedir+n+'/weights.hdf5')
		hist = pd.read_csv(savedir+n+'/history.log', sep=',', engine='python')
		hist['k'] = k
		history = pd.concat([history, hist])

		cv_score[n].loc[k] = evaluate(N[n][k], {'train': (X_train, Y_train, perm_train), 'val': (X_val, Y_val, perm_val)})

	print(cv_score[n].mean())

	PlotResults(savedir+n+'/', 'cv').plot_avg_metric_history(history, 'loss')
	PlotResults(savedir+n+'/', 'cv').plot_avg_metric_history(history, 'r_square')

	# TODO: there's definitely something wrong here as test is better than trainval
	# Maybe it has to do with the splitting? That some athletes in general are bad?
	N[n][k+1].fit(X_trainval[0], Y_trainval[0],
		epochs = 200,
		verbose = 1,
		callbacks = callbacks,
		validation_data = (X_test[0], Y_test[0]))

	N[n][k+1].load_weights(savedir+n+'/weights.hdf5')
	hist = pd.read_csv(savedir+n+'/history.log', sep=',', engine='python')
	hist['k'] = k+1
	history = pd.concat([history, hist])

	score.loc[n] = evaluate(N[n][k+1], {'trainval': (X_trainval, Y_trainval, perm_trainval), 'test': (X_test, Y_test, perm_test)})
	print(score)

	PlotResults(savedir+n+'/').plot_metric_history(history[history['k'] == k+1], 'loss')
	PlotResults(savedir+n+'/').plot_metric_history(history[history['k'] == k+1], 'r_square')

"""
	# lstm
	L = {'LSTM': LSTM()}
	for l in L.keys():
		print(l)
		L[l].fit(Xh_train, Yh_train, 
			epochs = epochs,
			verbose = 1,
			callbacks = callbacks,
			validation_data = (Xh_val, Yh_val))

best_m = score['r2'].argmax()

plt.figure()
plt.plot(np.unique(Y_val), np.unique(Y_val)*np.corrcoef(Y_pred, Y_val)[0,1])
plt.scatter(Y_val, Y_pred, s=.1, alpha=.7)
plt.xlabel('True '+cols_glucose[0][0])
plt.ylabel('Predicted '+cols_glucose[0][0])
plt.show()
"""