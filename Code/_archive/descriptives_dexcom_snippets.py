
df_training = pd.concat({idx: df[(df.RIDER == i) & (df.timestamp >= ts_min) & (df.timestamp <= ts_max)] 
	for idx, (i, ts_min, ts_max, _) in df_tp.iterrows()})

df_training = df_training.reset_index().rename(columns={'level_0':'tid', 'level_1':'index'}).set_index('index')
df_training.to_csv(path+'dexcom_clean_training.csv')

# ------------- glucose 4h after training session
df_after = pd.concat({idx: df[(df.RIDER == i) & (df.timestamp >= ts_max) & (df.timestamp <= ts_max + pd.to_timedelta('6h'))] 
	for idx, (i, ts_min, ts_max, _) in df_tp.iterrows()})

df_after = df_after.reset_index().rename(columns={'level_0':'tid', 'level_1':'index'}).set_index('index')

# remove timestamps that are in training sessions
df_after = df_after[~df_after.index.isin(df_training.index)]

# drop duplicates
df_after.drop_duplicates(subset=df_after.columns.drop('tid'), keep='last', inplace=True)

df_after.to_csv(path+'dexcom_clean_after.csv')

# ------------- glucose during the wake part of the day
# note: we don't select days with training sessions anymore
df_wake = df[df.local_timestamp.dt.time >= datetime.time(6)]
df_wake.to_csv(path+'dexcom_clean_wake.csv')

# ------------- glucose during the sleep part of the day
# note: we don't select days with training sessions anymore
df_sleep = df[df.local_timestamp.dt.time < datetime.time(6)]
df_sleep.to_csv(path+'dexcom_clean_sleep.csv')



df_training = {}
for i in athletes:
	df =  pd.read_csv(path_trainingpeaks+str(i)+'/'+str(i)+'_data.csv', index_col=0)
	df_training[i] = df.groupby('file_id')['local_timestamp'].agg(['first', 'last'])
df_training = pd.concat(df_training)
df_training = df_training.reset_index().rename(columns={'level_0':'athlete'})


ts_training = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=first, end=last, freq='s')]).to_frame(index=False, name=['RIDER', 'timestamp']) \
	for _, (athlete, first, last, _) in df_training.iterrows()])


df_glucose_training = pd.merge(df_glucose, ts_training, how='inner', on=['RIDER','timestamp'], validate='one_to_one')
df_glucose_training.to_csv('Data/Dexcom/dexcom_clean_training.csv')



ts_after = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=last, periods=4*3600, freq='s')]).to_frame(index=False, name=['RIDER', 'timestamp']) \
	for _, (athlete, _, first, last) in df_training.iterrows()])

# TODO: ask if this is the right approach to this division
# remove timestamps that are included in training sessions
ts_training['one'] = 1
ts_after = ts_after.merge(ts_training, how='left', on=['RIDER', 'timestamp'])
ts_after = ts_after[ts_after.one.isna()]
ts_training.drop('one', axis=1, inplace=True)
ts_after.drop('one', axis=1, inplace=True)

# drop duplicates
ts_after.drop_duplicates(inplace=True)

df_glucose_after = pd.merge(df_glucose, ts_after, how='inner', on=['RIDER','timestamp'], validate='one_to_one')
df_glucose_after.to_csv('Data/Dexcom/dexcom_clean_after.csv')

# ------------- glucose during the wake part of the day of the training sessions
df_training['first'] = pd.to_datetime(df_training['first'])
df_training['last'] = pd.to_datetime(df_training['last'])

ts_wake = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=datetime.datetime.combine(first.date(), datetime.time(6,0)), 
				  end=datetime.datetime.combine(first.date(), datetime.time(23,59,59)), freq='s')]).to_frame(index=False, name=['RIDER', 'timestamp']) \
	for _, (athlete, first, last, _) in df_training.iterrows()])

ts_wake.drop_duplicates(inplace=True)

df_glucose_wake = pd.merge(df_glucose, ts_wake, how='inner', on=['RIDER','timestamp'], validate='one_to_one')
df_glucose_wake.to_csv('Data/Dexcom/dexcom_clean_wake.csv')

# ------------- glucose during the sleep part of the day of the training sessions
ts_sleep = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=datetime.datetime.combine(first.date()+datetime.timedelta(days=1), datetime.time(0)), 
				  end=datetime.datetime.combine(first.date()+datetime.timedelta(days=1), datetime.time(5,59,59)), freq='s')]).to_frame(index=False, name=['RIDER', 'timestamp']) \
	for _, (athlete, first, last, _) in df_training.iterrows()])

ts_sleep.drop_duplicates(inplace=True)

df_glucose_sleep = pd.merge(df_glucose, ts_sleep, how='inner', on=['RIDER','timestamp'], validate='one_to_one')
df_glucose_sleep.to_csv('Data/Dexcom/dexcom_clean_sleep.csv')

# ------------- glucose on the entire day of the training session
ts_day = pd.concat([pd.MultiIndex.from_product([[athlete], 
	pd.date_range(start=datetime.datetime.combine(first.date(), datetime.time(0)), 
				  end=datetime.datetime.combine(first.date(), datetime.time(23,59,59)), freq='s')]).to_frame(index=False, name=['RIDER', 'timestamp']) \
	for _, (athlete, first, last, _) in df_training.iterrows()])

ts_day.drop_duplicates(inplace=True)
df_glucose_day = pd.merge(df_glucose, ts_day, how='inner', on=['RIDER','timestamp'], validate='one_to_one')
df_glucose_day.to_csv('Data/Dexcom/dexcom_clean_day.csv')


# not ANON
for i in df_glucose.RIDER.unique():
	type_palette = sns.color_palette("Set1")

	fig, ax0 = plt.subplots()
	ax0 = plot_glucose_levels(ax0, color=False)
	ax = ax0.twinx()
	
	for k, sec in enumerate(df_glucose_.keys()):
		sns.kdeplot(df_glucose_[sec][df_glucose_[sec].RIDER == i][col], ax=ax, linewidth=2, color=type_palette[k],
			label=sec+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
			.format(df_glucose_[sec][df_glucose_[sec].RIDER == i][col].mean(), 
					df_glucose_[sec][df_glucose_[sec].RIDER == i][col].std()/df_glucose_[sec][df_glucose_[sec].RIDER == i][col].mean()*100))
	plot_hist_glucose_settings(ax, ax0, col)
	plt.title(r'$\bf{:s}$ - completeness $= {:.0f}\%$ $\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$'\
		.format(rider_mapping_inv[i], len(df_glucose[df_glucose.RIDER == i][col])/(365*24*60/5)*100,
				df_glucose[df_glucose.RIDER == i][col].mean(),
				df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100), y=1.06)
	plt.savefig(savepath+'hist_glucose_sec_NAME_%s.pdf'%i, bbox_inches='tight')
	plt.savefig(savepath+'hist_glucose_sec_NAME_%s.png'%i, dpi=300, bbox_inches='tight')
	plt.show()
	plt.close()


# not ANON
type_palette = sns.color_palette("viridis_r", n_colors=11)

fig, ax0 = plt.subplots()
ax0 = plot_glucose_levels(ax0, color=False)
ax = ax0.twinx()
for c, i in enumerate(athletes):
	sns.kdeplot(df_glucose[df_glucose.RIDER == i][col], ax=ax, 
		linewidth=1.5, color=type_palette[c], alpha=.8,
		label=rider_mapping_inv[i]+r' ($\mu = {:.1f}$ $\sigma/\mu = {:.0f}\%$)'\
		.format(df_glucose[df_glucose.RIDER == i][col].mean(), 
				df_glucose[df_glucose.RIDER == i][col].std()/df_glucose[df_glucose.RIDER == i][col].mean()*100))
plot_hist_glucose_settings(ax, ax0, col)
plt.savefig(savepath+'hist_glucose_riders_NAME.pdf', bbox_inches='tight')
plt.savefig(savepath+'hist_glucose_riders_NAME.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()