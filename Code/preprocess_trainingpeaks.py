import numpy as np
import pandas as pd
import datetime
import os

from plot import *
from helper import *

path = 'Data/TrainingPeaks/'
if not os.path.exists(path+'clean/'):
	os.mkdir(path+'clean/')

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'csv/')])

dict_files = {}
for i in athletes:
	no_glucose=False
	df = pd.DataFrame()

	files = sorted(os.listdir(path+'csv/'+str(i)+'/data'))
	for f in files:
		df_tmp = pd.read_csv(path+'csv/'+str(i)+'/data/'+f)

		# open all other files and check for missing info

		df = df.append(df_tmp, ignore_index=True, verify_integrity=True)

	df['Zwift'] = False

	files_virtual = sorted(os.listdir(path+'csv/'+str(i)+'/Zwift/data'))
	for f in files_virtual:
		df_tmp = pd.read_csv(path+'csv/'+str(i)+'/Zwift/data/'+f)

		# open all other files and check for missing info

		df = df.append(df_tmp, ignore_index=True, verify_integrity=True)

	df['Zwift'].fillna(True, inplace=True)

	# save df to file
	df.to_csv(path+'clean/'+str(i)+'_data.csv', index_label=False)