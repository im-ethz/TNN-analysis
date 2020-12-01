import numpy as np
import pandas as pd

import datetime
import os

tp_path = 'Data/TrainingPeaks/'
if not os.path.exists(tp_path+'clean/'):
	os.mkdir(tp_path+'clean/')

tp_athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(tp_path+'csv/')])

for i in tp_athletes:
	df = pd.DataFrame()
	files = os.listdir(tp_path+'csv/'+str(i)+'/data')
	for f in files:
		df_tmp = pd.read_csv(tp_path+'csv/'+str(i)+'/data/'+f)

		# open all other files and check for missing info

		df = df.append(df_tmp, ignore_index=True, verify_integrity=True)
	df.to_csv(tp_path+'clean/'+str(i)+'_data.csv'