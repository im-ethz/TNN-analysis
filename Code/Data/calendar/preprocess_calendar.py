import os
import sys
sys.path.append(os.path.abspath('../../'))

import numpy as np
import pandas as pd

from plot import *
from helper import *
from config import rider_mapping

path = './'
rider_mapping = {k.upper() : v for k, v in rider_mapping.items()}

df = pd.read_excel(path+'2019_WorkoutsSummary.xls', sheet_name=None)
df = pd.concat(df)

# anonymize
print("RIDER nan: ", df[df.RIDER.isna()].dropna(axis=1))
df.dropna(subset=['RIDER'], inplace=True)
df['RIDER'] = df['RIDER'].map(rider_mapping)

# reindex
df = df.reset_index()
df.drop('level_0', axis=1, inplace=True)
df.rename(columns={'level_1':'TrainingIndex'}, inplace=True)
df = df.set_index(['RIDER', 'TrainingIndex'])

df.to_csv(path+'calendar_2019_anonymous.csv')

df[['WorkoutType','WorkoutDay', 'Event']].to_csv(path+'calendar-events_2019_anonymous')