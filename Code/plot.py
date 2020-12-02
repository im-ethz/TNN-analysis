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

	def plot_glucose_availability_calendar(self, df, **kwargs):
		plt.figure(figsize=(15,4))
		
		ax = sns.heatmap(df, **kwargs)

		cbar = ax.collections[0].colorbar
		cbar.set_ticks([-1,0,1])
		cbar.set_ticklabels(['False', 'NA', 'True'])

		plt.yticks(rotation=0)
		plt.xlabel('Day')
		plt.ylabel('Month')
		plt.title("Glucose availability TrainingPeaks")
		plt.savefig(self.savedir+str(self.athlete)+'_glucose_availaibility.pdf')

	def plot_feature_distr_subplots(self, df, feature_array, figsize, savetext=''):
		fig, axs = plt.subplots(*feature_array.shape)
		for i in range(feature_array.shape[0]):
			for j in range(feature_array.shape[1]):
				df[feature_array.flatten()].hist(column = feature_array[i,j], ax=axs[i,j], figsize=figsize)
				#ax[i,j].set_title('')
				#ax[i,j].set_xlabel(feature_array[i,j])
		plt.tight_layout()
		plt.savefig(self.savedir+'feature_hist_'+savetext+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'feature_hist_'+savetext+'.png', bbox_inches='tight')
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

	def plot_timeseries_parasite(df, cols, ylabels, ylims, axlocs, lws, alphas, figsize=(15,4),
		colors = [(0.6, 0.6, 0.6)] + sns.color_palette("colorblind"), savetext='', legend=True):
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
			# plot data
			#px, = ax[i].plot(df[c], label=c, lw=lws[i], c=colors[i], alpha=alphas[i])
			# plot smoothed line (ewm)
			try:
				ax[i].plot(df[c+'_ewm'], label=c+'_ewm', lw=lws[i], c=colors[i], alpha=.7)
				px, = ax[i].plot(df[c], label=c, c=colors[i], alpha=alphas[i], marker='o', markersize=.5, lw=0.)
			except:
				px, = ax[i].plot(df[c], label=c, lw=lws[i], c=colors[i], alpha=alphas[i])
				pass
			# plot elevation
			if i == 0:
				try:
					px, = ax[i].fill_between(pd.Series(df.index).values, df[c].values, 
						label=c, lw=.5, color='gray', alpha=.3)
				except TypeError:
					pass
			# plot imputed glucose
			if c == 'glucose':
				ax[i].scatter(df.index, df['@'+c], color=px.get_color(), alpha=alphas[i])
			ax[i].axis[axlocs[i]].label.set_color(px.get_color())
			ax[i].set_ylim(ylims[i])
			ax[i].set_ylabel(ylabels[i])
		
		ax[0].set_xlim((df.index.min(), df.index.max()))

		ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		ax[0].set_xlabel("Time (h:m)")
		if legend: ax[0].legend()

		plt.savefig(self.savedir+'feature_timeseries_'+savetext+'.pdf', bbox_inches='tight')
		plt.savefig(self.savedir+'feature_timeseries_'+savetext+'.png', bbox_inches='tight')
		plt.show()