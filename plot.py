# TODO:
# Glycaemic control (HbA1C) over the course of a season per athlete
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from colorsys import rgb_to_hls

from calc import glucose_levels
from config import SAVE_PATH, rider_mapping_inv

sns.set()
sns.set_context('paper')
sns.set_style('white')

plt.rcParams.update({'font.family':'sans-serif',
					 'font.sans-serif':'Latin Modern Sans'})

color_sec = {'wake'	: sns.color_palette("Set1")[1],
			 'exercise': sns.color_palette("Set1")[4],
			 'recovery': sns.color_palette("Set1")[2],
			 'sleep': sns.color_palette("Set1")[3]}

color_race = {'train': sns.color_palette("Set1")[8],
			  'race':(0.8455062527192158, 0.21363575247920147, 0.4145075850498335)} #'#d8366a'

palette_ath = sns.color_palette('inferno', n_colors=7)[:6]+ sns.color_palette('YlGnBu', n_colors=7)[:6] # alternatives for YlGnBu: viridis_r, mako_r

def cut_cmap(cmap_left, cmap_right, cut=10, freq=100, grey=None):
	left = cm.get_cmap(cmap_left+'_r', freq)
	right = cm.get_cmap(cmap_right, freq)
	if grey:
		colors = np.vstack((left(np.linspace(0, 1-cut/freq/2, freq)),
							cut*[cm.get_cmap('Greys')(grey)],
							right(np.linspace(0+cut/freq/2, 1, freq))))
	else:
		colors = np.vstack((left(np.linspace(0, 1-cut/freq/2, freq)),
							right(np.linspace(0+cut/freq/2, 1, freq))))
	
	return ListedColormap(colors, name='BluesReds')

def savefig(path, i='', dtype='Dexcom', legend=None, title=None, xticks=None, yticks=None, **titlekwargs):
	if title is not None:
		plt.title(r'$\bf{Cyclist}$ '+r'$\bf{:d}$ - '.format(i)+title, **titlekwargs)
	if legend is not None:
		for text in legend:
			text.set_fontsize(6)
	
	plt.savefig(f'{SAVE_PATH}{dtype}/{path}_{i}.pdf', bbox_inches='tight')
	plt.savefig(f'{SAVE_PATH}{dtype}/{path}_{i}.png', dpi=300, bbox_inches='tight')
	
	if title is not None:
		plt.title(r'$\bf{:s}$ '.format(rider_mapping_inv[i])+title, **titlekwargs)
	if legend is not None:
		for leg in legend:
			text = leg.get_text().split()
			leg.set_text(rider_mapping_inv[int(text[0])]+' '+' '.join(text[1:]))
	if xticks is not None:
		xticks.set_xticklabels([rider_mapping_inv[int(j.get_text())] for j in xticks.get_xticklabels()], rotation=90)
	if yticks is not None:
		yticks.set_yticklabels([rider_mapping_inv[int(j.get_text())] for j in yticks.get_yticklabels()], rotation=0)
	
	if title is not None or legend is not None or xticks is not None or yticks is not None:
		plt.savefig(f'{SAVE_PATH}{dtype}/{path}_NAME_{i}.pdf', bbox_inches='tight')
		plt.savefig(f'{SAVE_PATH}{dtype}/{path}_NAME_{i}.png', dpi=600, bbox_inches='tight')
	plt.show()
	plt.close()

def plot_availability(df_avail, cmap='Blues', rot_months=0, itv_months=1, vmin=0, vmax=1, plot_total=False, plot_colorbar=True):
	fig, ax = plt.subplots(figsize=(15,6))
	ax = sns.heatmap(df_avail, cmap=cmap, vmin=vmin, vmax=vmax)

	if plot_total:
		# put total percentage on RHS
		ax2 = ax.secondary_yaxis("right")
		ax2.set_yticks(np.arange(len(df_avail.index))+0.5)
		if plot_total == 'perc':
			ax2.set_yticklabels([r"$\bf{:.0f}\%$".format(i) for i in df_avail.sum(axis=1)/df_avail.notna().sum(axis=1)*100])
			ax.text(0.99, 1.02, r'$\bf{:s}$'.format('Total (\%)'), ha='left', transform=ax.transAxes)
		if plot_total == 'count':
			ax2.set_yticklabels([r"$\bf{:.0f}$".format(i) for i in df_avail.count(axis=1)])
			ax.text(0.99, 1.02, r'$\bf{:s}$'.format('Days'), ha='left', transform=ax.transAxes)
		ax2.tick_params(axis='y', length=0)
		ax2.spines['right'].set_visible(False)

	if plot_colorbar:
		# adjust ticks colorbar
		cbar_ax = fig.get_axes()[1]
		cbar_ax.set_yticks([0, .2, .4, .6, .8, 1.])
		cbar_ax.set_yticklabels(["{:.0f}%".format(i*100) for i in [0, .2, .4, .6, .8, 1.]])
		cbar_ax.text(3., 0.5, 'Percentage of max. CGM readings per day', va='center', rotation=270)

	monthyear = df_avail.columns.strftime("%b '%y")
	ticksloc = np.where(monthyear.to_series().shift() != monthyear.to_series())[0][1::itv_months]
	plt.xticks(ticks=ticksloc, labels=monthyear[ticksloc], rotation=rot_months)
	plt.xlabel('date')
	plt.ylabel('rider')
	return ax

def plot_hist_glucose_settings(ax, ax0, col='Glucose Value (mg/dL)', xlim=(20,410), ylabel='Probability', loc_legend=(1., 0.96)):
	ax.set_xlim((20, 410))
	ax.set_xlabel(col)
	ax.xaxis.set_visible(True)
	ax.set_ylabel(ylabel)
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_label_position('left')
	ax0.yaxis.set_visible(False)
	ax0.set_ylabel('')
	plt.legend(loc='upper right', bbox_to_anchor=loc_legend, prop={'family': 'DejaVu Sans Mono', 'size':8})

def plot_glucose_levels(ax, color=True, orient='vertical', text=True, subtext=True):
	if color:
		glucose_palette = sns.diverging_palette(10, 50, n=5)[:3] + sns.diverging_palette(10, 50, n=7)[4:6]
	else:
		glucose_palette = sns.diverging_palette(10, 10, s=0, n=5)

	# annotate glucose levels
	for i, (g, l) in enumerate(glucose_levels.items()):
		if orient == 'vertical':
			ax.axvspan(l[0], l[1]+0.99, alpha=0.2, color=glucose_palette[i], lw=0)
		elif orient == 'horizontal':
			ax.axhspan(l[0], l[1]+0.99, alpha=0.2, color=glucose_palette[i], lw=0)

	# text: hypo - target - hyper
	if text:
		ax.text(glucose_levels['hypo L2'][0]+25, 1.03, 'hypo', color=glucose_palette[0])
		ax.text(glucose_levels['hyper L1'][0]+35, 1.03, 'hyper', color=glucose_palette[4])
		ax.text(glucose_levels['target'][0]+25, 1.03, 'target', color=tuple([c*0.5 for c in glucose_palette[2]]))

	# text: L2 L1
	if subtext:
		ax.annotate('L2', xy=(glucose_levels['hypo L2'][0]+25, .95), color=glucose_palette[0])
		ax.annotate('L1', xy=(glucose_levels['hypo L1'][0], .95), color=glucose_palette[1])
		ax.annotate('L1', xy=(glucose_levels['hyper L1'][0]+25, .95), color=glucose_palette[3])
		ax.annotate('L2', xy=(glucose_levels['hyper L2'][0]+80, .95), color=glucose_palette[4])
	return ax

def plot_bar(data, x, width=.8, colors=dict(h_neg=10, h_pos=10, s=0, l=50), ax=plt, plot_numbers=False, unit='', duration=None):
	hatch = ('\\\\', '\\\\', None, '//', '//')
	color_palette = sns.diverging_palette(**colors, n=5)
	bottom = 0
	for sec, (label, y) in enumerate(data.items()):
		c = ax.bar(x=x, height=y, width=width, bottom=bottom, color=color_palette[sec], hatch=hatch[sec])
		bottom += y
		if plot_numbers and y >= 4:
			if sec == 2:
				ax.bar_label(c, labels=['%.0f'%y+unit], label_type='center', color='gray')
			elif plot_numbers == 'full':
				ax.bar_label(c, labels=['%.0f'%y+unit], label_type='center', fontweight='black', color='white')
	if duration:
		ax.text(x, -8, duration, ha='center', color='gray')

class PlotResults:
	def __init__(self, regression):
		vars(self).update(vars(regression))

	def info_coefficients(self, x):
		if pd.isnull(x['Pr(>|z|)']):
			return None
		else:
			info = r" {:.2f}  ({:.2f} - {:.2f})  ".format(x['Estimate'], x['CI_lower'], x['CI_upper'])
			if type(x['Pr(>|z|)']) == float:
				info += r"   {:.3f}".format(x['Pr(>|z|)'])
			else:
				#info += r"$<$"+r"{:s}".format(x['Pr(>|z|)'].split('<')[1])
				info += r"   {:s}".format(x['Pr(>|z|)'].split('<')[1])
			info += r"{:s}".format(x['Sign'])
			return info

	def subplot_coefficients(self, df, fig, ax, title='', textlocs=(0, 0.7), cmax=0.6, xlim=None, leq_sep=0.9, categories=True):
		cmap = cm.get_cmap('RdBu_r')#cut_cmap('Blues', 'Reds', cut=0)
		if not cmax:
			cmax = np.log(df['Estimate']).abs().max()
		colors = ((np.log(df['Estimate']) / cmax)+1)/2

		x = np.arange(df.shape[0])

		#ax.axvline(1, color='black')
		ax.plot((1,1), (1, df.shape[0]), color='black')
		
		ax.scatter(df['Estimate'], x, marker='s', color=cmap(colors))

		# clip if outside of xlims (TODO: change this)
		for n in x:
			ax.plot((df['CI_lower'].iloc[n], df['CI_upper'].iloc[n]), (n,n), color=cmap(colors[n]))

		# ticks on LHS
		ax.set_yticks(x, df.index)
		if categories:
			cat_list = list(self.categories.keys())
			for n, tick in enumerate(ax.get_yticklabels()):
				if tick.get_text() in cat_list:
					#tick.set_fontstyle('italic')
					tick.set_fontsize(8)
					tick.set_fontweight('bold')
					cat_list.remove(tick.get_text())

		# ticks on RHS
		ax0 = ax.twinx()
		info_ticks = df.apply(self.info_coefficients, axis=1)
		info_ticks[0] = "Odds ratio (95% CI)  p-value"
		ax0.set_yticks(x, info_ticks)
		ax0.get_yticklabels()[0].set_fontweight('bold')

		# put < sign manually so that we can use latex text, and it outlines to the RHS
		xmax = xlim[1] if xlim else df.max().max()
		for n in x:
			if not pd.isnull(df['Pr(>|z|)'].iloc[n]):
				if str(df['Pr(>|z|)'].iloc[n]).startswith('<'):
					ax0.text(xmax+(xmax-1)*leq_sep, n+0.06, r"$<$", color='gray', va='center')

		# arrowheads
		ax.plot(1, 0, marker=9, color='black', markersize=5, transform=ax.transAxes, clip_on=False)
		ax.plot(0, 0, marker=8, color='black', markersize=5, transform=ax.transAxes, clip_on=False)

		# xlabel
		ax.set_xlabel(f'Odds ratio (95% CI) of {self.event}glycemia', labelpad=30)
		ax.text(textlocs[0], -1/ax.get_figure().get_size_inches()[1]*0.6+0.01, 
				f'Decreased risk of\n {self.event}glycemia', color=cmap(0.05), fontsize=8, transform=ax.transAxes)
		ax.text(textlocs[1], -1/ax.get_figure().get_size_inches()[1]*0.6+0.01, 
				f'Increased risk of\n {self.event}glycemia', color=cmap(0.95), fontsize=8, transform=ax.transAxes)

		ax.set_title(title.upper(), x=1, y=-1, transform=ax.transData)

		# layout
		sns.despine(ax=ax, left=True, right=True)
		sns.despine(ax=ax0, left=True, right=True)

		ax.tick_params(length=0)
		ax0.tick_params(length=0, pad=2, colors='gray')

		ax.set_ylim(x.max()+1, x.min())
		ax0.set_ylim(x.max()+1, x.min())

		if xlim:
			ax.set_xlim(xlim)
			ax0.set_xlim(xlim)

	def plot_coefficients(self, fe, transform, figsize=(12,5), savefig=True, **kws_sub):
		fig, axs = plt.subplots(1,3, figsize=figsize, sharey=True, sharex=True)
		for i, sec in enumerate(self.sections):
			self.subplot_coefficients(transform(fe)[sec], fig, axs[i], title=sec, **kws_sub)
		plt.tight_layout()

		if savefig:
			plt.savefig(f"{self.root}coefficients_{self.filename.lstrip('model_')}.pdf", bbox_inches='tight')
			plt.savefig(f"{self.root}coefficients_{self.filename.lstrip('model_')}.png", bbox_inches='tight', dpi=600)

	def plot_coefficients_per_sec(self, fe_hypo, fe_hyper, sec, transform, figsize=(8,5), savefig=True, **kws_sub):
		fig, axs = plt.subplots(1,2, figsize=figsize, sharey=True, sharex=True)
		self.event = 'hypo'
		self.subplot_coefficients(transform(fe_hypo)[sec], fig, axs[0], title=sec, **kws_sub)
		self.event = 'hyper'
		self.subplot_coefficients(transform(fe_hyper)[sec], fig, axs[1], title=sec, **kws_sub)
		plt.tight_layout()

		if savefig:
			plt.savefig(f"{self.root}coefficients_{self.filename.split('_')[2]}_{sec}.pdf", bbox_inches='tight')
			plt.savefig(f"{self.root}coefficients_{self.filename.split('_')[2]}_{sec}.png", bbox_inches='tight', dpi=600)
