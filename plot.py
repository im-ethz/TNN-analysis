# TODO:
# Glycaemic control (HbA1C) over the course of a season per athlete
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from colorsys import rgb_to_hls, hls_to_rgb

from calc import glucose_levels
from config import SAVE_PATH, ANON, rider_mapping_inv

ANON = True

plt.style.use('./diabetes_care.mplstyle')

color_sec = {'wake'	: sns.color_palette("Set1")[4],#[1],
			 'exercise': sns.color_palette("Set1")[2],#[4],
			 'recovery': sns.color_palette("Set1")[1],#[2],
			 'sleep': sns.color_palette("Set1")[3]}

color_race = {'train': sns.color_palette("Set1")[8],
			  'race':(0.8455062527192158, 0.21363575247920147, 0.4145075850498335)} #'#d8366a'

palette_ath = sns.color_palette('inferno', n_colors=7)[:6]+ sns.color_palette('YlGnBu', n_colors=7)[:6] # alternatives for YlGnBu: viridis_r, mako_r

def cut_cmap(cmap_left, cmap_right, cut=10, freq=100, grey=None):
	left = get_cmap(cmap_left+'_r', freq)
	right = get_cmap(cmap_right, freq)
	if grey:
		colors = np.vstack((left(np.linspace(0, 1-cut/freq/2, freq)),
							cut*[get_cmap('Greys')(grey)],
							right(np.linspace(0+cut/freq/2, 1, freq))))
	else:
		colors = np.vstack((left(np.linspace(0, 1-cut/freq/2, freq)),
							right(np.linspace(0+cut/freq/2, 1, freq))))
	
	return ListedColormap(colors, name='BluesReds')

def savefig(path, i='', legend=None, title=None, xticks=None, yticks=None, **titlekwargs):
	if title is not None:
		plt.title(r'$\bf{Participant}$ '+r'$\bf{:d}$ - '.format(i)+title, **titlekwargs)
	if legend is not None:
		for text in legend:
			text.set_fontsize(8)
	
	plt.savefig(f'{SAVE_PATH}/{path}_{i}.pdf')#, bbox_inches='tight')
	plt.savefig(f'{SAVE_PATH}/{path}_{i}.png', dpi=1000)#, bbox_inches='tight')
	
	if not ANON:
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
			plt.savefig(f'{SAVE_PATH}/{path}_NAME_{i}.pdf')#, bbox_inches='tight')
			plt.savefig(f'{SAVE_PATH}/{path}_NAME_{i}.png', dpi=1000)#, bbox_inches='tight')
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
	plt.ylabel('participant')
	return ax

def plot_hist_glucose_settings(ax, ax0, col='Glucose Value (mg/dL)', xlim=(20,410), ylabel='Probability', loc_legend=(1., 0.96)):
	ax.set_xlim((20, 410))
	ax.set_xlabel(col)
	ax.xaxis.set_visible(True)
	ax.set_ylabel(ylabel)
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_label_position('left')
	ax0.yaxis.set_visible(False)
	sns.despine(ax=ax0, bottom=True, top=True, left=True, right=True)
	ax0.set_ylabel('')
	plt.legend(loc='upper right', bbox_to_anchor=loc_legend)

def plot_glucose_levels(ax, orient='vertical', shade=False, text=False, subtext=False):
	assert orient in ('vertical', 'horizontal'), "Please pass either horizontal or vertical"

	if shade:
		fn = ax.axvspan if orient == 'vertical' else ax.axhspan
		for i, (g, l) in enumerate(glucose_levels.items()):
			fn(l[0], l[1]+0.99, alpha=.2, color=sns.diverging_palette(10, 10, s=0, n=5)[i], lw=0)
	else:
		fn = ax.axvline if orient == 'vertical' else ax.axhline
		for g, l in list(glucose_levels.items())[1:]:
			fn(l[0], color='k', linewidth=.5, zorder=1)

	# text: hypo - target - hyper
	if text:
		ax.text(glucose_levels['hypo L2'][0]+25, 1.03, 'hypo')
		ax.text(glucose_levels['hyper L1'][0]+35, 1.03, 'hyper')
		ax.text(glucose_levels['target'][0]+25, 1.03, 'target')

	# text: L2 L1
	if subtext:
		ax.annotate('L2', xy=(glucose_levels['hypo L2'][0]+25, .95), fontsize=8)
		ax.annotate('L1', xy=(glucose_levels['hypo L1'][0], .95), fontsize=8)
		ax.annotate('L1', xy=(glucose_levels['hyper L1'][0]+25, .95), fontsize=8)
		ax.annotate('L2', xy=(glucose_levels['hyper L2'][0]+80, .95), fontsize=8)
	return ax

def plot_bar(data, x, width=.8, colors=dict(h_neg=10, h_pos=10, s=0, l=50), ax=plt, plot_numbers=False, labelsize=10, unit='', duration=None):
	hatch = ('\\\\', '\\\\', None, '//', '//')
	color_palette = sns.diverging_palette(**colors, n=5)
	bottom = 0
	for sec, (label, y) in enumerate(data.items()):
		c = ax.bar(x=x, height=y, width=width, bottom=bottom, color=color_palette[sec], hatch=hatch[sec])
		bottom += y
		if plot_numbers and y >= 4:
			if sec == 2:
				ax.bar_label(c, labels=['%.0f'%y+unit], label_type='center', fontsize=labelsize, color='black')
			elif plot_numbers == 'full':
				ax.bar_label(c, labels=['%.0f'%y+unit], label_type='center', fontsize=labelsize, fontweight='bold', color='white')
	if duration:
		ax.text(x, -8, duration, ha='center', color='black')

class PlotResults():
	def __init__(self, regression):
		vars(self).update(vars(regression))
		self.regression = regression

	def info_coefficients(self, x):
		if pd.isnull(x['Pr(>|z|)']):
			return None
		else:
			info = r" {:.2f}  [{:.2f}$-${:.2f}]  ".format(x['Estimate'], x['CI_lower'], x['CI_upper'])
			if type(x['Pr(>|z|)']) == float:
				info += r"   {:.3f}".format(x['Pr(>|z|)'])
			else:
				#info += r"$<$"+r"{:s}".format(x['Pr(>|z|)'].split('<')[1])
				info += r"   {:s}".format(x['Pr(>|z|)'].split('<')[1])
			info += r"{:s}".format(x['Sign'])
			return info

	def subplot_coefficients(self, df, fig, ax, sec, 
			textx=(-0.5, 0.7), texty=1.1, xlim=None, leq=1.9, 
			tickcolor = 'gray',
			cmax=0.5, cmap_cut=30, categories=True):
		#cmap = cut_cmap('Blues', 'Reds', cut=cmap_cut)#get_cmap('RdBu_r')#
		palette = sns.color_palette("RdBu_r", n_colors=11)
		cmap = LinearSegmentedColormap.from_list("", [palette[0], '#CCCCCC', palette[-1]])
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
		info_ticks[0] = "Odds ratio [95%CI] $p$-value"
		ax0.set_yticks(x, info_ticks)
		ax0.get_yticklabels()[0].set_fontweight('bold')

		# put < sign manually so that we can use latex text, and it outlines to the RHS
		xmax = xlim[1] if xlim else df.max().max()
		for n in x:
			if not pd.isnull(df['Pr(>|z|)'].iloc[n]):
				if str(df['Pr(>|z|)'].iloc[n]).startswith('<'):
					ax0.text(xmax+(xmax-1)*leq, n+0.06, r"$<$", color=tickcolor, va='center', fontsize=8)

		# arrowheads
		ax.plot(1, 0, marker=9, color='black', markersize=5, transform=ax.transAxes, clip_on=False)
		ax.plot(0, 0, marker=8, color='black', markersize=5, transform=ax.transAxes, clip_on=False)

		# xlabel
		ax.set_xlabel(f'Odds ratio of {self.event}glycemia \nduring {sec}', labelpad=15)#30)
		ax.text(textx[0], -1/ax.get_figure().get_size_inches()[1]*texty+0.15, 
				f'Decreased odds',#\nof {self.event}glycemia', 
				color=cmap(0.01), fontsize=8, transform=ax.transAxes)
		ax.text(textx[1], -1/ax.get_figure().get_size_inches()[1]*texty+0.15, 
				f'Increased odds',#\nof {self.event}glycemia', 
				color=cmap(0.99), fontsize=8, transform=ax.transAxes)

		#ax.set_title(title.title(), x=1, y=-1, transform=ax.transData, fontsize=9)

		# layout
		sns.despine(ax=ax, left=True, right=True)
		sns.despine(ax=ax0, left=True, right=True)

		ax.tick_params(length=0)
		ax0.tick_params(length=0, pad=2, colors=tickcolor)

		ax.set_ylim(x.max()+1, x.min())
		ax0.set_ylim(x.max()+1, x.min())

		if xlim:
			ax.set_xlim(xlim)
			ax0.set_xlim(xlim)

		ax.set_xscale('log')
		ax.xaxis.set_major_locator(LogLocator(base=2, numticks=6))
		ax.xaxis.set_minor_locator(LogLocator(base=100, numticks=1)) #we need this because somehow they are not turned off
		ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

	def plot_coefficients(self, co, figsize=(6.8,5), wspace=2, xlim=(0.45, 2.2), leq=11.5, savefig=True, **kws_sub):
		fig, axs = plt.subplots(1,3, figsize=figsize, sharey=True, sharex=True, gridspec_kw=dict(wspace=wspace))
		for i, sec in enumerate(self.sections):
			self.subplot_coefficients(self.regression.transform_co(co)[sec], fig, axs[i], 
				sec=sec, xlim=xlim, leq=leq, **kws_sub)

		if savefig:
			plt.savefig(f"{self.root}coef_{self.filename[6:]}.pdf", bbox_inches='tight')
			plt.savefig(f"{self.root}coef_{self.filename[6:]}.png", dpi=1000, bbox_inches='tight')
		plt.show()
		plt.close()

	def plot_coefficients_env(self, fe, figsize=(10,20), wspace=1.2, xlim=(0.1, 10), textx=(-0.05, 0.6), leq=21, drop_minor_ticks=False, savefig=True, **kws_sub):
		cols = fe.index.get_level_values(0).unique()
		fig, axs = plt.subplots(len(cols),3, figsize=figsize, sharey='row', sharex=True, gridspec_kw=dict(wspace=wspace))
		for n, col in enumerate(cols):
			for i, sec in enumerate(self.sections):
				self.subplot_coefficients(self.regression.transform_fe(fe.loc[col])[sec], fig, axs[n, i], 
					sec=sec, xlim=xlim, leq=leq, textx=textx, **kws_sub)
				for text in axs[n,i].texts:
					if n != len(cols)-1:
						text.set_visible(False)
					else:
						text.set_position((text.get_position()[0], text.get_position()[1]-0.4))
				if n != 0:
					axs[n,i].set_title('')
				if n != len(cols)-1:
					axs[n,i].set_xlabel('')
				elif drop_minor_ticks:
					axs[n,i].xaxis.set_tick_params(which='minor', labelbottom=False)

		if savefig:
			plt.savefig(f"{self.root}coef_env_{self.filename[6:]}.pdf", bbox_inches='tight')
			plt.savefig(f"{self.root}coef_env_{self.filename[6:]}.png", dpi=1000, bbox_inches='tight')
		plt.show()
		plt.close()

	def plot_coefficients_per_sec(self, co_hypo, co_hyper, sec, figsize=(6.8,4.5), wspace=0.8, xlim=(0.45, 2.2), 
		textx=(0, 0.6), texty=1, leq=2.2, savefig=True, **kws_sub):
		fig, axs = plt.subplots(1,2, figsize=figsize, sharey=True, sharex=True, gridspec_kw=dict(wspace=wspace))
		self.event = 'hypo'
		self.subplot_coefficients(self.regression.transform_co(co_hypo)[sec], fig, axs[0], 
			sec=sec, xlim=xlim, leq=leq, textx=textx, texty=texty, **kws_sub)
		self.event = 'hyper'
		self.subplot_coefficients(self.regression.transform_co(co_hyper)[sec], fig, axs[1], 
			sec=sec, xlim=xlim, leq=leq, textx=textx, texty=texty, **kws_sub)

		if savefig:
			plt.savefig(f"{self.root}coef_{self.filename.split('_')[2]}_{sec}.pdf", bbox_inches='tight')
			plt.savefig(f"{self.root}coef_{self.filename.split('_')[2]}_{sec}.png", dpi=1000, bbox_inches='tight')
		plt.show()
		plt.close()