# TODO: descriptives for minute data
import numpy as np
import scipy as sp
import pandas as pd
import datetime
import os
import gc
import matplotlib

from plot import *
from helper import *

# remove if moved to plot
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from statsmodels.graphics.tsaplots import plot_acf

sns.set()
sns.set_context('paper')
sns.set_style('white')

path = 'Data/TrainingPeaks+LibreView/'
savedir = 'Descriptives/'
if not os.path.exists(savedir):
	os.mkdir(savedir)

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'1sec/') if i.endswith('.csv')])

def plot_feature_distr_subplots(df, cols, figsize, savetext=''):
	fig, axs = plt.subplots(*cols.shape)
	for i in range(cols.shape[0]):
		for j in range(cols.shape[1]):
			print(cols[i,j])
			df[cols[i,j]].hist(ax=axs[i,j])
			df[cols.flatten()].hist(column = cols[i,j], ax=axs[i,j], figsize=figsize)
			#ax[i,j].set_title('')
			#ax[i,j].set_xlabel(feature_array[i,j])
	plt.tight_layout()
	#plt.savefig(self.savedir+'feature_hist_'+savetext+'.pdf', bbox_inches='tight')
	#plt.savefig(self.savedir+'feature_hist_'+savetext+'.png', bbox_inches='tight')
	plt.show()

def plot_feature_timeseries_subplots(self, df, feature_array, figsize, savetext='', scatter=False):
	if scatter:
		plot_kwargs = dict(marker='o', markersize=.5, lw=0.)
		pname = 'scatter'
	else:
		plot_kwargs = dict(lw=.5)
		pname = 'plot'

	axs = df[feature_array.flatten()].plot(subplots=True, layout=feature_array.shape, figsize=figsize, **plot_kwargs)
	for i in range(feature_array.shape[0]):
		for j in range(feature_array.shape[1]):
			axs[i,j].set_ylabel(feature_array[i,j])
			axs[i,j].legend(loc='upper center')
			try:
				axs[i,j].plot(df[feature_array[i,j]+'_ewm'], color=axs[i,j].get_lines()[0].get_color())
			except:
				pass
	plt.tight_layout()
	plt.savefig(self.savedir+'feature_'+pname+'_'+savetext+'.pdf', bbox_inches='tight')
	plt.savefig(self.savedir+'feature_'+pname+'_'+savetext+'.png', bbox_inches='tight')
	plt.show()


def plot_timeseries_parasite(df, cols, ylabels, ylims, axlocs, lws, alphas, figsize=(15,4),
	colors = [(0.6, 0.6, 0.6)] + sns.color_palette("colorblind"), savetext='', legend=True):
	axlocs = ['left' if x == 'l' else 'right' for x in axlocs]
	plt.figure(figsize=figsize)
	ax = []
	ax.append(host_subplot(111, axes_class=AA.Axes))
	plt.subplots_adjust(right=0.75)

	# set up parasite axes
	lr_count = {'left':0, 'right':0}
	for i in range(1, len(cols)):
		ax.append(ax[0].twinx())
		if i > 1:
			offset = (lr_count[axlocs[i]]+1)*60
			if axlocs[i] == 'left':
				offset *= -1
				ax[i].axis['right'].toggle(all=False)
			new_fixed_axis = ax[i].get_grid_helper().new_fixed_axis
			ax[i].axis[axlocs[i]] = new_fixed_axis(loc=axlocs[i], axes=ax[i], offset=(offset, 0))
			lr_count[axlocs[i]] += 1
		ax[i].axis[axlocs[i]].toggle(all=True)
	
	# plot
	for i, c in enumerate(cols):
		#try:
		#	ax[i].plot(df[c+'_ewm'], label=c+'_ewm', lw=lws[i], c=colors[i], alpha=.7)
		#	px, = ax[i].plot(df[c], label=c, c=colors[i], alpha=alphas[i], marker='o', markersize=.5, lw=0.)
		#except:
		px, = ax[i].plot(df[c], label=c, lw=lws[i], c=colors[i], alpha=alphas[i])
		#	pass
		# plot altitude
		if c == 'altitude':
			try:
				px, = ax[i].fill_between(pd.Series(df.index).values, df[c].values, 
					label=c, lw=.5, color='gray', alpha=.1)
			except TypeError:
				pass
		# plot imputed glucose
		if c == 'Scan Glucose mg/dL':
			ax[i].scatter(df.index, df[c], color=px.get_color(), alpha=alphas[i])
		ax[i].axis[axlocs[i]].label.set_color(px.get_color())
		ax[i].set_ylim(ylims[i])
		ax[i].set_ylabel(ylabels[i])
		ax[i].axis[axlocs[i]].major_ticklabels.set_color(px.get_color())

	ax[0].set_xlim((df.index.min(), df.index.max()))

	ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax[0].set_xlabel("Time (h:m)")
	if legend: ax[0].legend()

	#plt.savefig(self.savedir+'feature_timeseries_'+savetext+'.pdf', bbox_inches='tight')
	#plt.savefig(self.savedir+'feature_timeseries_'+savetext+'.png', bbox_inches='tight')
	plt.show()


for i in athletes:
	df = pd.read_csv(path+'1sec/'+str(i)+'.csv', index_col=0)
	df.index = pd.to_datetime(df.index)

	# drop nancols #TODO: move to preprocess
	df.dropna(how='all', axis=1, inplace=True)

	cols_plot = np.array([['position_lat', 'position_long', 'distance_diff', 'speed', 'cadence'],
				 		  ['altitude', 'grade', 'ascent', 'power', 'heart_rate'],
						  ['left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 'left_torque_effectiveness', 'right_torque_effectiveness'],
				 		  ['glucose_BUBBLE', 'Historic Glucose mg/dL', 'Historic Glucose mg/dL (filled)', 'Scan Glucose mg/dL', 'calories'],
						  ['gps_accuracy', 'temperature', 'temperature_smooth', 'battery_soc', 'battery_soc_ilin']])

	# for one athlete, plot histogram subplots of all features
	axs = df[cols_plot.flatten()].hist(grid=False, figsize=(20,20)) #TODO: figsize
	for ax in axs.flatten():
		ax.set_yticks([])
	plt.savefig(savedir+'hist_'+str(i)+'.pdf', bbox_inches='tight')
	plt.savefig(savedir+'hist_'+str(i)+'.png', bbox_inches='tight')
	plt.show()

	# for one athlete, plot histogram subplots of all features + variance
	# EDIT: to computationally expensive with 25 features
	# sns.pairplot(df[cols_plot.flatten()]) #T
	# plt.show()

	# for one athlete, plot diagonal correlation matrix
	corr = df[cols_plot.flatten()].corr()
	sns.heatmap(corr, vmin=-1, vmax=1, center=0, #mask=np.triu(np.ones_like(corr, dtype=bool)),
		linewidths=.5, cmap=sns.diverging_palette(230,20,as_cmap=True), square=True)
	plt.savefig(savedir+'corr_'+str(i)+'.pdf', bbox_inches='tight')
	plt.savefig(savedir+'corr_'+str(i)+'.png', bbox_inches='tight')
	plt.show()

	# TODO: for one training print all features
	# remove 'grade'
	cols_ts = np.array(['altitude', 'ascent', 'distance_diff', 'cadence', 'speed', 'power', 'temperature_smooth',
						'heart_rate', 'Historic Glucose mg/dL (filled)', 'Scan Glucose mg/dL', 'calories'])#,
						#'left_right_balance', 'left_pedal_smoothness', 'right_pedal_smoothness', 
						#'left_torque_effectiveness', 'right_torque_effectiveness'])

	df_i = df[df.file_id == df.file_id[0]]

	plot_timeseries_parasite(df_i, cols_ts, cols_ts, 
		ylims = [(0,1000), (-0.05, 0.05), (0,100), (0,500), (0,50), (0,1200), (0, 30), (0, 250), (0,200), (0,200), (-0.05, 0.05)],
		axlocs = ['l', 'r', 'l', 'r', 'r', 'r', 'r', 'r', 'l', 'l', 'r'],#, 'r', 'r', 'r', 'r', 'r'],
		lws = [.7, .7, .7, .7, .7, .7, .7, .7, 1., 1., .7],#, .7, .7, .7, .7, .7],
		alphas = [.7, .7, .7, .7, .7, .7, .7, .7, .7, .7, .7],#, .7, .7, .7, .7, .7], 
		figsize=(20,5),	legend=False)

"""
	figsize = (15,4)
	axlocs = ["left", "right", "right", "right", "right", "right"]
	colors = [(0.6, 0.6, 0.6)] + sns.color_palette("colorblind")
	lws = [.7, .7, .7, .7, .7, .7]
	alphas = [.7, .7, .7, .2, .7, .7]
	legend = False

	plt.figure(figsize=figsize)
	ax = []
	ax.append(host_subplot(111, axes_class=AA.Axes))
	plt.subplots_adjust(right=0.75)

	# set up parasite axes
	lr_count = {'left':0, 'right':0}
	for i in range(1, len(cols_ts)):
		ax.append(ax[0].twinx())
		if i > 1:
			offset = (lr_count[axlocs[i]]+1)*60
			if axlocs[i] == 'left':
				offset *= -1
				ax[i].axis['right'].toggle(all=False)
			new_fixed_axis = ax[i].get_grid_helper().new_fixed_axis
			ax[i].axis[axlocs[i]] = new_fixed_axis(loc=axlocs[i], axes=ax[i], offset=(offset, 0))
			lr_count[axlocs[i]] += 1
		ax[i].axis[axlocs[i]].toggle(all=True)
	
	# plot
	for i, c in enumerate(cols_ts):
		try:
			ax[i].plot(df_i[c+'_ewm'], label=c+'_ewm', lw=lws[i], c=colors[i], alpha=.7)
			px, = ax[i].plot(df_i[c], label=c, c=colors[i], alpha=alphas[i], marker='o', markersize=.5, lw=0.)
		except:
			px, = ax[i].plot(df_i[c], label=c, lw=lws[i], c=colors[i], alpha=alphas[i])
			pass
		# plot altitude
		if c == 'altitude':
			try:
				px, = ax[i].fill_between(pd.Series(df_i.index).values, df_i[c].values, 
					label=c, lw=.5, color='gray', alpha=.2)
			except TypeError:
				pass
		# plot imputed glucose
		if c == 'glucose':
			ax[i].scatter(df_i.index, df_i['@'+c], color=px.get_color(), alpha=alphas[i])
		ax[i].axis[axlocs[i]].label.set_color(px.get_color())
		#ax[i].set_ylim(ylims[i])
		ax[i].set_ylabel(cols_ts[i])
		ax[i].axis[axlocs[i]].major_ticklabels.set_color(px.get_color())

	ax[0].set_xlim((df_i.index.min(), df_i.index.max()))

	ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	ax[0].set_xlabel("Time (h:m)")
	if legend: ax[0].legend()

	#plt.savefig(self.savedir+'feature_timeseries_'+savetext+'.pdf', bbox_inches='tight')
	#plt.savefig(self.savedir+'feature_timeseries_'+savetext+'.png', bbox_inches='tight')
	plt.show()
"""

	# TODO: for all trainings print median and std of all features