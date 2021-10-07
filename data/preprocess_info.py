import numpy as np
import pandas as pd
import datetime
import os
import gc

import sys
sys.path.append(os.path.abspath('../'))

from calc import calc_hr_zones, calc_power_zones

from config import DATA_PATH

SAVE_PATH = './'

# ----------------------- age + diagnosis
age = pd.read_csv(DATA_PATH+'age_diagnosis.csv', index_col=0)

# ----------------------- fit
# read in fit variables
fit = pd.read_csv(DATA_PATH+'fitness.csv', index_col=[0,1], header=[0,1])
fit.reset_index(inplace=True)

# take average for beginning, mid and end of season
fit = fit.groupby('RIDER').mean()
fit.reset_index(inplace=True)

cols = {('RIDER', '')						: 'RIDER',
		('ID and ANTROPOMETRY', 'Age')		: 'age', #yr
		('ID and ANTROPOMETRY', 'Height')	: 'height', #cm
		('ID and ANTROPOMETRY', 'Weight')	: 'weight', #kg
		('ID and ANTROPOMETRY', 'bf(%)')	: 'bf(%)', #%
		('ID and ANTROPOMETRY', 'HbA1C')	: 'HbA1c', #% (?)
		('VT2 (RCP)', 'W')					: 'FTP',#W
		('VT2 (RCP)', 'HR')					: 'LTHR', #bpm
		('VO2peak', 'HR')					: 'HRmax', #bpm
		('VO2peak', 'VO2/Kg')				: 'VO2max', #mL/min/kg
		}

fit = fit[cols.keys()]
cols = {'_'.join(k):v for k,v in cols.items()}
fit.columns = ['_'.join(c) for c in fit.columns]
fit.rename(columns=cols, inplace=True)

# note: age peron in the fitness file is incorrect
info = pd.merge(age, fit.drop('age', axis=1), how='outer', on='RIDER')
info = info[info.RIDER.isin([1,2,3,4,5,6,10,12,13,14,15])]
info.to_csv(SAVE_PATH+'info.csv', index_label=False)