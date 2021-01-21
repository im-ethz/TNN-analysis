import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from statsmodels.graphics.tsaplots import plot_acf

# TODO: adjust plotsize plot interp_subplots
# TODO: ylabel interp plot
# TODO: legend interp individual plot

def custom_colormap(base, cmin, cmax, n):
	cmap_base = plt.cm.get_cmap(base)
	cmap_custom = cmap_base.from_list(base+str(n), cmap_base(np.linspace(cmin, cmax, n)), n)
	return cmap_custom

class PlotData:
	def __init__(self, savedir, savetext='', athlete='all'):
		sns.set()
		sns.set_context('paper')
		sns.set_style('white')

		self.savedir = savedir
		self.savetext = savetext
		self.athlete = athlete

		#plt.rcParams.update({"text.usetex": True, "text.latex.preamble":[r"\usepackage{amsmath}",r"\usepackage{siunitx}",],})

	def plot_glucose_availability_calendar(self, df, cbarticks, dtype='TrainingPeaks', **kwargs):
		plt.figure(figsize=(15,4))
		
		ax = sns.heatmap(df, **kwargs)

		cbar = ax.collections[0].colorbar
		cbar.set_ticks(list(cbarticks.values()))
		cbar.set_ticklabels(list(cbarticks.keys()))

		for text in ax.texts:
			if text.get_text() == '0':
				text.set_visible(False)

		plt.yticks(rotation=0)
		plt.xlabel('Day')
		plt.ylabel('Month')
		plt.title("Glucose availability %s - Athlete %s"%(dtype, self.athlete))
		plt.savefig(self.savedir+str(self.athlete)+self.savetext+'_glucose_availaibility.pdf', bbox_inches='tight')
		plt.close()

	def plot_feature_distr_subplots(self, df, i, cols, figsize, savetext=''):
		axs = df[cols].hist(grid=False, figsize=figsize)
		for ax in axs.flatten():
			ax.set_yticks([])
		plt.savefig(self.savedir+'hist_'+str(i)+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'hist_'+str(i)+'.png', bbox_inches='tight')
		plt.show()
		plt.close()

	def plot_feature_correlation(self, df, i, cols, ticks=None, ticklocs=None):
		if ticks is None:
			ticks = cols
		corr = df[cols].corr()
		ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, #mask=np.triu(np.ones_like(corr, dtype=bool)),
			linewidths=.5, cmap=sns.diverging_palette(230,20,as_cmap=True), square=True)
		if ticklocs is not None:
			plt.xticks(ticklocs+0.5, ticks)
			plt.yticks(ticklocs+0.5, ticks)
		plt.savefig(self.savedir+'corr_'+str(i)+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'corr_'+str(i)+'.png', bbox_inches='tight')
		plt.show()
		plt.close()

	def plot_feature_clustermap(self, df, i, cols, ticks=None, ticklocs=None):
		if ticks is None:
			ticks = cols
		corr = df[cols].corr()
		sns.clustermap(corr, vmin=-1, vmax=1, center=0, row_cluster=False, cbar_pos=(0.05, .35, .02, .4), #mask=np.triu(np.ones_like(corr, dtype=bool)),
			linewidths=.5, cmap=sns.diverging_palette(230,20,as_cmap=True), square=True)
		if ticklocs is not None:
			plt.xticks(ticklocs+0.5, ticks)
			plt.yticks(ticklocs+0.5, ticks)
		plt.savefig(self.savedir+'corrcluster_'+str(i)+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'corrcluster_'+str(i)+'.png', bbox_inches='tight')
		plt.show()
		plt.close()

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


	def plot_feature_timeseries_parasite(self, df, i, cols, ylabels, ylims, axlocs, lws, alphas, figsize=(15,4),
		colors = [(0.6, 0.6, 0.6)] + sns.color_palette("colorblind"), legend=True):
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
			px, = ax[i].plot(df[c], label=c, lw=lws[i], c=colors[i], alpha=alphas[i])
			# plot altitude
			if c[:8] == 'altitude':
				try:
					px, = ax[i].fill_between(pd.Series(df.index).values, df[c].values, 
						label=c, lw=.5, color='gray', alpha=.1)
				except TypeError:
					pass
			# plot imputed glucose
			if c[:18] == 'Scan Glucose mg/dL':
				ax[i].scatter(df.index, df[c], color=px.get_color(), alpha=alphas[i])
			ax[i].axis[axlocs[i]].label.set_color(px.get_color())
			ax[i].set_ylim(ylims[i])
			ax[i].set_ylabel(ylabels[i])
			ax[i].axis[axlocs[i]].major_ticklabels.set_color(px.get_color())

		ax[0].set_xlim((df.index.min(), df.index.max()))

		ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		ax[0].set_xlabel("Time (h:m)")
		if legend: ax[0].legend()

		plt.savefig(self.savedir+'feature_timeseries_'+str(i)+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'feature_timeseries_'+str(i)+'.png', bbox_inches='tight')
		plt.show()
		plt.close()

	def plot_feature_timeseries_agg_parasite(self, df, i, cols, ylabels, ylims, axlocs, lws, alphas, figsize=(15,4),
		colors = [(0.6, 0.6, 0.6)] + sns.color_palette("colorblind"), legend=True):
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
			px, = ax[i].plot(df[(c, 'mean')], label=c, lw=lws[i], c=colors[i], alpha=alphas[i])
			ax[i].fill_between(df[(c, 'mean')].values - df[(c, 'std')].values, df[(c, 'mean')].values + df[(c, 'std')].values, 
				label=c, lw=.5, color=colors[i], alpha=alphas[i]-0.3)

			# plot altitude
			if c[:8] == 'altitude':
				try:
					px, = ax[i].fill_between(pd.Series(df.index).values, df[(c, 'mean')].values, 
						label=c, lw=.5, color='gray', alpha=.1)
				except TypeError:
					pass
			# plot imputed glucose
			if c[:18] == 'Scan Glucose mg/dL':
				ax[i].scatter(df.index, df[(c, 'mean')], color=px.get_color(), alpha=alphas[i])
			ax[i].axis[axlocs[i]].label.set_color(px.get_color())
			ax[i].set_ylim(ylims[i])
			ax[i].set_ylabel(ylabels[i])
			ax[i].axis[axlocs[i]].major_ticklabels.set_color(px.get_color())

		ax[0].set_xlim((df.index.min(), 30000))

		#ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		ax[0].set_xlabel("Time (s)")
		if legend: ax[0].legend()

		plt.savefig(self.savedir+'feature_timeseries_'+str(i)+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'feature_timeseries_'+str(i)+'.png', bbox_inches='tight')
		plt.show()
		plt.close()

	def plot_interp_allinone(self, df, feature, interp_list, savetext=''):
		plt.figure()
		for i in interp_list:
			plt.plot(df.index, df[feature+'_'+i], label=i)
		plt.scatter(df.index, df['@'+feature])
		plt.legend()
		plt.savefig(self.savedir+'interp_allinone'+feature+'_'+savetext+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'interp_allinone'+feature+'_'+savetext+'.png', bbox_inches='tight')
		plt.show()

	def plot_interp_subplots(self, df, feature, interp_list, shape, savetext=''):
		fig, ax = plt.subplots(shape[0], shape[1], sharex=True, sharey=True)
		for i in range(shape[0]):
			for j in range(shape[1]):
				df[feature+'_'+interp_list[shape[1]*i+j]].plot(ax=ax[i,j], label=interp_list[shape[1]*i+j])
				ax[i,j].scatter(df.index, df['@'+feature], s=5., c='red')
				ax[i,j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
				ax[i,j].legend()
		plt.tight_layout()
		plt.savefig(self.savedir+'interp_'+feature+'_'+savetext+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'interp_'+feature+'_'+savetext+'.png', bbox_inches='tight')
		plt.show()

	def plot_interp_individual(self, df, feature, interp):
		plt.figure()
		df[feature+'_'+interp].plot()
		plt.scatter(df.index, df['@'+feature], label=interp)
		plt.legend()
		plt.show()

	def plot_smooth_individual(self, df, feature, smoothings):
		plt.figure()
		plt.scatter(df.index, df[feature], s=.3, label='data', alpha=.4, edgecolor=None)
		for s in smoothings:
			plt.plot(df.index, df[feature+'_'+s], label=s, lw=1.5, alpha=.7)
		plt.legend()
		plt.ylabel(feature)
		plt.show()

	def plot_smooth_subplots(self, df, feature_list, smoothings, shape, savetext=''):
		fig, ax = plt.subplots(shape[0], shape[1], sharex=True, figsize=(18,10))
		for i in range(shape[0]):
			for j in range(shape[1]):
				ax[i,j].scatter(df.index, df[feature_list[shape[1]*i+j]], s=.3, label='data', alpha=.4, edgecolor=None)
				for s in smoothings:
					ax[i,j].plot(df.index, df[feature_list[shape[1]*i+j]+'_'+s], color='red', label=s, lw=1.5, alpha=.7)
				ax[i,j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
				ax[i,j].set_ylabel(feature_list[shape[1]*i+j])
				ax[i,j].legend()
		plt.tight_layout()
		plt.savefig(self.savedir+'smooth_'+savetext+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'smooth_'+savetext+'.png', bbox_inches='tight')
		plt.show()