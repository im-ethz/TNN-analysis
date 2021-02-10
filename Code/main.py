# TOOD (shortterm):
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

from model import *
from plot import *
from helper import *

verbose = 1

sns.set()
sns.set_context('paper')
sns.set_style('white')

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

df.reset_index(inplace=True)

# plot histogram of glucose
# glucose at t
glucose_palette = sns.diverging_palette(10, 50, n=5)[:3] + sns.diverging_palette(10, 50, n=7)[4:6]
type_palette = sns.color_palette("viridis")
patch_count = [0]

fig, ax0 = plt.subplots()
# annotate glucose levels
for i, (g, l) in enumerate(glucose_levels.items()):
	ax0.axvspan(l[0], l[1], alpha=0.2, color=glucose_palette[i])

ax0.text(glucose_levels['hypo L2'][0]+25, 1.03, 'hypo', color=glucose_palette[0])
ax0.text(glucose_levels['hyper L1'][0]+35, 1.03, 'hyper', color=glucose_palette[4])
ax0.text(glucose_levels['normal'][0]+25, 1.03, 'normal', color=tuple([c*0.5 for c in glucose_palette[2]]))

ax0.annotate('L2', xy=(glucose_levels['hypo L2'][0]+25, .95), color=glucose_palette[0])
ax0.annotate('L1', xy=(glucose_levels['hypo L1'][0], .95), color=glucose_palette[1])
ax0.annotate('L1', xy=(glucose_levels['hyper L1'][0]+25, .95), color=glucose_palette[3])
ax0.annotate('L2', xy=(glucose_levels['hyper L2'][0]+80, .95), color=glucose_palette[4])

ax = ax0.twinx()

sns.histplot(df[('Scan Glucose mg/dL')], label='Scan', ax=ax,
	stat='density', kde=True,
	binwidth=10, alpha=0.3, line_kws={'lw':2.})
patch_count.append(len(ax.patches))

sns.histplot(df[('Bubble Glucose mg/dL')], label='Bubble',  ax=ax,
	stat='density', kde=True,
	binwidth=10, alpha=0.3, line_kws={'lw':2.})
patch_count.append(len(ax.patches))

sns.histplot(df[('Historic Glucose mg/dL (filled)', 'mean', 't')], label='Historic', ax=ax,
	stat='density', kde=True, 
	binwidth=10, alpha=0.3, line_kws={'lw':2.})
patch_count.append(len(ax.patches))

# somehow changing the color in the function does not work
for l in range(len(ax.lines)):
	ax.lines[l].set_color(type_palette[l*2])
	for p in ax.patches[patch_count[l]:patch_count[l+1]]:
		alpha = p.get_facecolor()[3]
		p.set_facecolor(type_palette[l*2])#[n_color])
		p.set_alpha(alpha)	

ax.set_xlim((20, df[['Scan Glucose mg/dL', 'Bubble Glucose mg/dL', 'Historic Glucose mg/dL (filled)']].max().max()+30))
ax0.set_xlabel('Glucose mg/dL')
ax0.set_ylabel('Probability')
ax.set_ylabel('')
plt.legend()
plt.savefig('Descriptives/hist_glucose.pdf', bbox_inches='tight')
plt.savefig('Descriptives/hist_glucose.png', bbox_inches='tight')
plt.show()


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
for k in range(K):
	df.loc[idx_val[k], ('split', '', '')] = k
df.loc[idx_test, ('split', '', '')] = K+1

df_split = df.groupby('training_id').first()
df_split = df_split[['local_timestamp', 'athlete', 'file_id', 'split']]
df_split.columns = df_split.columns.get_level_values(0)
df_split.sort_values(['athlete', 'file_id'])
for i in athletes:
	df_split.loc[df_split.athlete == i, 'file_id'] = np.arange(len(df_split[df_split.athlete == i]))
df_split = df_split.set_index(['athlete', 'file_id']).unstack()['split']
sns.heatmap(df_split, cmap=sns.color_palette('Greens', K+1))
plt.show()

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

cv_score = {m: pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'], [c[0].rstrip(" (filled)").rstrip("lucose mg/dL")[:-2] for c in cols_Y]+['Historic (avg)'], ['train', 'val']])) for m in M.keys()}
score = pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'], [c[0].rstrip(" (filled)").rstrip("lucose mg/dL")[:-2] for c in cols_Y]+['Historic (avg)'], ['trainval', 'test']]))
for m in M.keys():
	print(m)
	for k in range(K):
		print(k)
		X_train, Y_train, X_val, Y_val, perm_train, perm_val = get_data(df, cols_X, cols_Y, k, idx_train, idx_val)

		M[m][k].fit(X_train[0], Y_train[0])

		cv_score[m].loc[k] = evaluate(M[m][k], {'train': (X_train, Y_train, perm_train), 'val': (X_val, Y_val, perm_val)})

	print(cv_score[m].mean())

	X_trainval, Y_trainval, X_test, Y_test, perm_trainval, perm_test = get_data(df, cols_X, cols_Y, 0, [idx_trainval], [idx_test])
	M[m][k+1].fit(X_trainval[0], Y_trainval[0])
	score.loc[m] = evaluate(M[m][k+1], {'trainval': (X_trainval, Y_trainval, perm_trainval), 'test': (X_test, Y_test, perm_test)})
	print(score)
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
N = {'NeuralNetwork': [NN(len(cols_X), 1, n_hidden = [50], act_hidden = ['relu'], do_hidden=[0.5, 0.5], optimizer=Adam(learning_rate=1e-3)).model for _ in range(K+1)]}

cv_score.update({n: pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'],  [c[0].rstrip(" (filled)").rstrip("lucose mg/dL")[:-2] for c in cols_Y]+['Historic (avg)'], ['train', 'val']])) for n in N.keys()})
#score = pd.DataFrame(columns=pd.MultiIndex.from_product([['mse', 'r2'], ['hist', 'scan'], ['test']]))
history = pd.DataFrame()

for n in N.keys():
	print(n)
	for k in range(K):
		print(k)
		X_train, Y_train, X_val, Y_val, perm_train, perm_val = get_data(df, cols_X, cols_Y, k, idx_train, idx_val)

		# TODO: batch size
		callbacks = [CSVLogger(savedir+'nn/history.log', separator=',', append=False),
					 EarlyStopping(monitor='val_loss', min_delta=0, patience=10),
					 ModelCheckpoint(savedir+'nn/weights.hdf5', monitor='val_loss', verbose=1, mode='min', save_best_only=True)]

		N[n][k].fit(X_train[0], Y_train[0], 
			epochs = 200,
			verbose = 1,
			callbacks = callbacks,
			validation_data = (X_val[0], Y_val[0]))

		N[n][k].load_weights(savedir+'nn/weights.hdf5')
		hist = pd.read_csv(savedir+'nn/history.log', sep=',', engine='python')
		hist['k'] = k
		history = pd.concat([history, hist])

		cv_score[n].loc[k] = evaluate(N[n][k], {'train': (X_train, Y_train, perm_train), 'val': (X_val, Y_val, perm_val)})

		#plot_history(history[history['k'] == k], 'loss')
		#plot_history(history[history['k'] == k], 'r_square')
	print(cv_score[n].mean())

	plot_avg_history(history, 'loss')
	plot_avg_history(history, 'r_square')

	# TODO: there's definitely something wrong here as test is better than trainval
	# Maybe it has to do with the splitting? That some athletes in general are bad?
	X_trainval, Y_trainval, X_test, Y_test, perm_trainval, perm_test = get_data(df, cols_X, cols_Y, 0, [idx_trainval], [idx_test])
	N[n][k+1].fit(X_trainval[0], Y_trainval[0],
		epochs = 200,
		verbose = 1,
		callbacks = callbacks,
		validation_data = (X_test[0], Y_test[0]))

	N[n][k+1].load_weights(savedir+'nn/weights.hdf5')
	hist = pd.read_csv(savedir+'nn/history.log', sep=',', engine='python')
	hist['k'] = k+1
	history = pd.concat([history, hist])

	score.loc[n] = evaluate(N[n][k+1], {'trainval': (X_trainval, Y_trainval, perm_trainval), 'test': (X_test, Y_test, perm_test)})
	print(score)

	plot_history(history[history['k'] == k+1], 'loss')
	plot_history(history[history['k'] == k+1], 'r_square')


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