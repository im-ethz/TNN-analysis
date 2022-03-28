from matplotlib import dates as mdates

from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist as AA

class PlotPreprocess:
    def __init__(self, savedir, savetext='', athlete='all'):
        sns.set()
        sns.set_context('paper')
        sns.set_style('white')

        self.savedir = savedir
        self.savetext = savetext
        self.athlete = str(athlete)

    def plot_hist(self, x, xname, bins=40):
        x.hist(bins=bins)
        plt.xlabel(xname)
        plt.savefig(self.savedir+xname+'_'+self.athlete+'.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+xname+'_'+self.athlete+'.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

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

    def plot_training_calendar(self, df):
        df_avail = df['athlete'].copy().to_frame()
        df_avail.index = pd.to_datetime(df_avail.index).date
        df_avail['month'] = pd.to_datetime(df_avail.index).month
        df_avail = df_avail.reset_index().drop_duplicates()
        df_avail = df_avail.groupby(['athlete', 'month']).count().unstack()['index']
        df_avail = df_avail.rename(columns=month_mapping)
        sns.heatmap(df_avail, annot=True, linewidth=.5, cmap='Greens')
        plt.savefig(self.savedir+'glucose_availability_all.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+'glucose_availability_all.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_hist_feature_subplots(self, df, athlete, cols, figsize, kde=False, plot_smooth=False, layout=None, savetext=''):
        if isinstance(cols, pd.MultiIndex):
            colnames = [c[0] for c in cols]
        else:
            colnames = cols
        
        if kde == True:
            if layout is not None:
                cols = cols.to_numpy().reshape(layout)
                fig, axs = plt.subplots(*layout, figsize=figsize)
                for i in range(layout[0]):
                    for j in range(layout[1]):
                        sns.histplot(df[cols[i,j]], ax=axs[i,j], kde=True)
                        axs[i,j].set_ylabel('')
                if plot_smooth:
                    sns.histplot(df[('temperature_smooth', 'mean', 't')], ax=axs[layout[0]-1, layout[1]-1], kde=True, label='smooth', color='green')
                    axs[layout[0]-1, layout[1]-1].legend()
            else:
                print("Please give layout with KDE")
                return
        else:
            axs = df[cols].hist(grid=False, figsize=figsize, layout=layout)

        for j, ax in enumerate(axs.flatten()):
            ax.set_yticks([])
            ax.set_title('')
            ax.set_xlabel(colnames[j])

        plt.tight_layout()
        plt.savefig(self.savedir+'hist_'+str(athlete)+'.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+'hist_'+str(athlete)+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_hist_glucose(self, df, cols_Y, binwidth=10):
        types = [c[0].rstrip(" (filled)").rstrip("lucose mg/dL")[:-2] for c in cols_Y]
        type_palette = sns.color_palette("viridis")
        
        patch_count = [0]
        fig, ax0 = plt.subplots()
        ax0 = plot_glucose_levels(ax0)
        ax = ax0.twinx()

        for col, name in zip(cols_Y, types):
            sns.histplot(df[col], label=name, ax=ax,
                stat='density', kde=True,
                binwidth=binwidth, alpha=0.3, line_kws={'lw':2.})
            patch_count.append(len(ax.patches))

        # TODO: remove
        # somehow changing the color in the function does not work
        for l in range(len(ax.lines)):
            ax.lines[l].set_color(type_palette[l*2])
            for p in ax.patches[patch_count[l]:patch_count[l+1]]:
                alpha = p.get_facecolor()[3]
                p.set_facecolor(type_palette[l*2])#[n_color])
                p.set_alpha(alpha)  

        ax.set_xlim((20, df[cols_Y].max().max()+30))
        ax0.set_xlabel('Glucose mg/dL')
        ax0.set_ylabel('Probability')
        ax.set_ylabel('')
        plt.legend()
        plt.savefig(self.savedir+'hist_glucose.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+'hist_glucose.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_feature_correlation(self, df, cols, ticks=None, ticklocs=None):
        if ticks is None:
            ticks = cols
        corr = df[cols].corr()
        ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, #mask=np.triu(np.ones_like(corr, dtype=bool)),
            linewidths=.5, cmap=sns.diverging_palette(230,20,as_cmap=True), square=True)
        if ticklocs is not None:
            plt.xticks(ticklocs+0.5, ticks, rotation='vertical')
            plt.yticks(ticklocs+0.5, ticks)
        ax.xaxis.tick_top()
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(self.savedir+'corr_'+self.savetext+'_'+str(self.athlete)+'.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+'corr_'+self.savetext+'_'+str(self.athlete)+'.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.savedir+'corrcluster_'+str(i)+'.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.savedir+'feature_'+pname+'_'+savetext+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

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
        plt.savefig(self.savedir+'feature_timeseries_'+str(i)+'.png', dpi=300, bbox_inches='tight')
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
        plt.savefig(self.savedir+'feature_timeseries_'+str(i)+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_interp_allinone(self, df, feature, interp_list, savetext=''):
        plt.figure()
        for i in interp_list:
            plt.plot(df.index, df[feature+'_'+i], label=i)
        plt.scatter(df.index, df['@'+feature])
        plt.legend()
        plt.savefig(self.savedir+'interp_allinone'+feature+'_'+savetext+'.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+'interp_allinone'+feature+'_'+savetext+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

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
        plt.savefig(self.savedir+'interp_'+feature+'_'+savetext+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_interp_individual(self, df, feature, interp):
        plt.figure()
        df[feature+'_'+interp].plot()
        plt.scatter(df.index, df['@'+feature], label=interp)
        plt.legend()
        plt.show()
        plt.close()

    def plot_smooth_individual(self, df, feature, smoothings):
        plt.figure()
        plt.scatter(df.index, df[feature], s=.3, label='data', alpha=.4, edgecolor=None)
        for s in smoothings:
            plt.plot(df.index, df[feature+'_'+s], label=s, lw=1.5, alpha=.7)
        plt.legend()
        plt.ylabel(feature)
        plt.show()
        plt.close()

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
        plt.savefig(self.savedir+'smooth_'+savetext+'.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def plot_data_split(self, df, idx_val, idx_test):
        K = len(idx_val)
        for k in range(K):
            df.loc[idx_val[k], ('split', '', '')] = k
        df.loc[idx_test, ('split', '', '')] = K+1

        df_split = df.groupby('training_id').first()
        df_split = df_split[['local_timestamp', 'athlete', 'file_id', 'split']]
        df_split.columns = df_split.columns.get_level_values(0)
        df_split.sort_values(['athlete', 'file_id'])
        for i in df.athlete.unique():
            df_split.loc[df_split.athlete == i, 'file_id'] = np.arange(len(df_split[df_split.athlete == i]))
        df_split = df_split.set_index(['athlete', 'file_id']).unstack()['split']
        sns.heatmap(df_split, cmap=sns.color_palette('Greens', K+1))
        plt.savefig(self.savedir+'data_split.pdf', bbox_inches='tight')
        plt.savefig(self.savedir+'data_split.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()