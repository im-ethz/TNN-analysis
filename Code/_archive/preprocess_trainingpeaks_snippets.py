

	idx_zero_power = df[df.power == 0].index
	idx_nan_power = df[df.power.isna()].index

	idx_first_zero_power = idx_zero_power[idx_zero_power.to_series().diff() != 1]
	idx_first_nan_power = idx_nan_power[idx_nan_power.to_series().diff() != 1]

	df.loc[idx_first_nan_power-1, 'power']
	df['timediff'] = df.local_timestamp.diff() 
	df['power_shift'] = df.power.shift(1)
	df.loc[idx_first_nan_power, ['timediff', 'device_ELEMNT', 'device_ELEMNTBOLT', 'device_GARMIN', 'device_ZWIFT']]
	df.groupby(['device_ELEMNT', 'device_ELEMNTBOLT', 'device_GARMIN', 'device_ZWIFT'])['timediff'].mean()

	df.loc[idx_first_nan_power].loc[(df.loc[idx_first_nan_power, 'timediff'] != '1s') & \
									(df.loc[idx_first_nan_power, 'power_shift'] == 0)]

	for i in idx_first_nan_power:
		print(df.loc[i-20:i+10, ['local_timestamp', 'power', 'file_id']])

	for f in df.file_id.unique():
		df_f = df[df.file_id == f]
		idx_zero_power = df_f[df_f.power == 0].index
		idx_nan_power = df_f[df_f.power == 0].index




	"""
	OLD: use this when we don't remove nans for local timestamps, or when we don't combine both timestamps
	nan_ts = df['timestamp'].notna()
	nan_lts = df['local_timestamp'].notna()
	nan_ltsl = df['local_timestamp_loc'].notna()

	print_times_dates("duplicated timestamps", 
		df[nan_ts], df[nan_ts]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts], df[nan_lts]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl], df[nan_ltsl]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')

	dupl_timestamp = df[nan_ts][df[nan_ts]['timestamp'].duplicated()]
	dupl_timestamp_first = df[nan_ts]['timestamp'].duplicated(keep='first')
	dupl_timestamp_last = df[nan_ts]['timestamp'].duplicated(keep='last')
	dupl_timestamps_keep = df[nan_ts][df[nan_ts]['timestamp'].duplicated(keep=False)].sort_values('timestamp')

	# TODO: find out a way to combine this info. 
	print("Are there glucose values in the duplicate data that is not dropped? ",
		not df[nan_ts][df[nan_ts]['timestamp'].duplicated(keep=False)].sort_values('timestamp').glucose.dropna().empty)
	print("Are there glucose values in the duplicate that that will be dropped? ",
		not df[nan_ts][df[nan_ts]['timestamp'].duplicated()].sort_values('timestamp').glucose.dropna().empty)

	dupl_devices = []
	for dev in devices:
		print("Duplicate timestamps in %s: "%dev, df[nan_ts][dupl_timestamp_first][dev].sum())
		if df[nan_ts][dupl_timestamp_first][dev].sum() != 0:
			dupl_devices.append(dev)
	# TODO: make sure that this happens in the right order, and that not the original duplicate is removed
	# TODO: find out why garmin is double and if data is the same then (latter: not)

	df['drop_dupl_timestamp'] = df[dupl_devices].sum(axis=1).astype(bool)
	"""

	"""
	print("\n----------------- SELECT: device_0 == ELEMNT")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & df['device_ELEMNTBOLT']], 
		df[nan_ts & df['device_ELEMNTBOLT']]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & df['device_ELEMNTBOLT']], 
		df[nan_lts & df['device_ELEMNTBOLT']]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & df['device_ELEMNTBOLT']], 
		df[nan_ltsl & df['device_ELEMNTBOLT']]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	# TODO: check if duplicated local timestamps were in error_timestamps_fileid

	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'])], 
		df[nan_ltsl | (df['device_ELEMNTBOLT'] | df['device_zwift'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	
	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift or Rouvy")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])], 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')

	print("\n----------------- SELECT: device_0 == ELEMNT BOLT or zwift or Rouvy or Garmin")
	print_times_dates("duplicated timestamps", 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_ts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['timestamp'].duplicated())
	print_times_dates("duplicated local timestamps", 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_lts & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['local_timestamp'].duplicated(), ts='local_timestamp')
	print_times_dates("duplicated local timestamps from location", 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])], 
		df[nan_ltsl & (df['device_ELEMNTBOLT'] | df['device_zwift'] | df['device_Rouvy'] | df['device_garmin'])]['local_timestamp_loc'].duplicated(), ts='local_timestamp_loc')
	"""

	"""
	df_dupl = df[df['timestamp'].duplicated(keep=False)]
	if not df_dupl.empty:
		df_dupl['date'] = df_dupl['timestamp'].dt.date
		df_dupl[df_dupl['timestamp'] == df_dupl.iloc[0]['timestamp']]

		print("Dates with multiple files but not duplicate timestamps:\n",
			*tuple(set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist()) 
				- set(df_dupl['date'].unique())))
		print("Dates with duplicate timestamps but not multiple files:\n",
			*tuple(set(df_dupl['date'].unique()) 
				- set(pd.to_datetime(date_filename[date_filename['N'] > 1].index).date.tolist())))
		# conclusion: duplicate timestamps must be because there are duplicate files
	"""
