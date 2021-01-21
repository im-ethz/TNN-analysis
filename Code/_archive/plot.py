def	plot_feature_distr_suplots():
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

def plot_feature_timeseries_parasite(df, cols, ylabels, ylims, axlocs, lws, alphas, figsize=(15,4),
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