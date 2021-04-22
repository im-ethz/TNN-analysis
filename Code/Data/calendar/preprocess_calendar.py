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

# reindex, create training index and create calendar index
df.reset_index(inplace=True)
df.drop('level_0', axis=1, inplace=True)
df.rename(columns={'level_1':'TrainingIndex'}, inplace=True)
df.sort_values(['RIDER', 'TrainingIndex'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.reset_index(inplace=True)
df.rename(columns={'index':'CalendarIndex'}, inplace=True)
df.set_index(['RIDER', 'TrainingIndex'], drop=True, inplace=True)

df = df[['WorkoutDay', 'WorkoutType', 'Event', 'Title', 'WorkoutDescription',
       'PlannedDuration', 'PlannedDistanceInMeters',
       'CoachComments', 'DistanceInMeters', 'PowerAverage', 'PowerMax',
       'Energy', 'AthleteComments', 'TimeTotalInHours', 'VelocityAverage',
       'VelocityMax', 'CadenceAverage', 'CadenceMax', 'HeartRateAverage',
       'HeartRateMax', 'TorqueAverage', 'TorqueMax', 'IF', 'TSS',
       'HRZone1Minutes', 'HRZone2Minutes', 'HRZone3Minutes', 'HRZone4Minutes',
       'HRZone5Minutes', 'HRZone6Minutes', 'HRZone7Minutes', 'HRZone8Minutes',
       'HRZone9Minutes', 'HRZone10Minutes', 'PWRZone1Minutes',
       'PWRZone2Minutes', 'PWRZone3Minutes', 'PWRZone4Minutes',
       'PWRZone5Minutes', 'PWRZone6Minutes', 'PWRZone7Minutes',
       'PWRZone8Minutes', 'PWRZone9Minutes', 'PWRZone10Minutes', 'Rpe',
       'Feeling', 'CalendarIndex']]

df[df.Event == 'Race'].to_csv('calendar_2019_anon_RACE.csv')
df[df.Event == 'Camp'].to_csv('calendar_2019_anon_CAMP.csv')

df.to_csv(path+'calendar_2019_anonymous.csv')