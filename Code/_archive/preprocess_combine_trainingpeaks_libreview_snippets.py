	## CHECK IF THIS SHOULD BE LOCAL TIMESTAMP
	"""
	dates_tp_glucose = df_tp.timestamp[df_tp.glucose_BUBBLE.notna()].dt.date.unique()

	for date in dates_tp_glucose:
		df_tp_example = df_tp[df_tp.timestamp.dt.date == date][['timestamp','local_timestamp', 'glucose_BUBBLE']]
		df_lv_example = df_lv[df_lv.timestamp.dt.date == date][['Device Timestamp','Historic Glucose mg/dL', 'Scan Glucose mg/dL', 'Strip Glucose mg/dL']]
		df_example = df_merge[df_merge.index.date == date][['local_timestamp', 'glucose_BUBBLE', 'Historic Glucose mg/dL', 'Scan Glucose mg/dL', 'Strip Glucose mg/dL']]

		plt.scatter(df_tp_example['timestamp'], df_tp_example['glucose_BUBBLE'], label='tp')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Historic Glucose mg/dL'], label='lv_hist')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Scan Glucose mg/dL'], label='lv_scan')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Strip Glucose mg/dL'], label='lv_strip')
		plt.legend()
		plt.show()

		plt.scatter(df_tp_example['local_timestamp'], df_tp_example['glucose_BUBBLE'], label='tp')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Historic Glucose mg/dL'], label='lv_hist')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Scan Glucose mg/dL'], label='lv_scan')
		plt.scatter(df_lv_example['Device Timestamp'], df_lv_example['Strip Glucose mg/dL'], label='lv_strip')
		plt.legend()
		plt.show()
	"""
