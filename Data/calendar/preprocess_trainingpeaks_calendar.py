import os

import numpy as np
import pandas as pd

import datetime

import gc

path = '../TrainingPeaks/2019/'

athletes = sorted([int(i.rstrip('.csv')) for i in os.listdir(path+'csv/')])

# TODO: improve by adding n_workout that says how many workouts there are per day instead of a counter
# match trainingpeaks data with calendar
# goal: check whether there are training sessions missing, training sessions that should be merged,
# or training sessions that are not in the calendar
# -------------------- Read df_training
df_training = {}
for i in athletes:
	print("\n------------------------------- Athlete ", i)

	df = pd.read_csv(path+'clean2/'+str(i)+'/'+str(i)+'_data.csv', index_col=0)

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])

	# create training array 
	df_i = df.copy()
	df_i['date_x'] = pd.to_datetime(df_i.local_timestamp.dt.date)
	df_i.rename(columns={'distance':'distance_x'}, inplace=True)
	df_i = df_i[['date_x', 'file_id', 'distance_x']].drop_duplicates(subset=['date_x', 'file_id'], keep='last').reset_index(drop=True)
	
	# round distance to hm
	df_i['distance_km_x'] = df_i['distance_x'] / 1e3
	df_i['distance_km_x'] = df_i['distance_km_x'].round()
	df_i.drop('distance_x', axis=1, inplace=True)

	# and identify workout number in df
	df_i['n_workout_x'] = df_i.groupby('date_x').apply(lambda x: pd.Series(np.arange(len(x)))).reset_index(drop=True)

	df_training[i] = df_i

df_training = pd.concat(df_training).reset_index().drop('level_1', axis=1).rename(columns={'level_0':'athlete'})

# -------------------- Read calendar
# read workout calendar
calendar = pd.read_csv('./calendar_2019_anonymous.csv')
calendar.WorkoutDay = pd.to_datetime(calendar.WorkoutDay)

# read training calendar
calendar = calendar.loc[calendar.WorkoutType == 'Bike', 
			['RIDER', 'WorkoutDay', 'CalendarIndex', 'DistanceInMeters']].reset_index(drop=True)
calendar.rename(columns={'WorkoutDay':'date_y', 
						 'DistanceInMeters':'distance_y', 
						 'CalendarIndex':'cal_id'}, inplace=True)

# round distance to hm
calendar['distance_y'].replace({0:np.nan}, inplace=True)
calendar['distance_km_y'] = calendar['distance_y'] / 1e3
calendar['distance_km_y'] = calendar['distance_km_y'].round()
calendar.drop('distance_y', axis=1, inplace=True)

# identify workout number in calendar
calendar['n_workout_y'] = calendar.groupby(['RIDER', 'date_y']).apply(lambda x: pd.Series(np.arange(len(x)))).reset_index(drop=True)

# -------------------- First merge
# check if keys are unique (keep duplicates as residue)
x_dupl = df_training.duplicated(subset=['athlete', 'date_x', 'distance_km_x'], keep='first')
y_dupl = calendar.duplicated(subset=['RIDER', 'date_y', 'distance_km_y'], keep='first')

x = df_training[~x_dupl] ; res_x = df_training[x_dupl]
y = calendar[~y_dupl] 	 ; res_y = calendar[y_dupl]

# first merge
df_merge = pd.merge(x, y, how='outer', 
	left_on=['athlete', 'date_x', 'distance_km_x'], right_on=['RIDER', 'date_y', 'distance_km_y'], validate="one_to_one")

# get residue
res_x = res_x.append(df_merge[df_merge.date_y.isna()]\
	.drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge[df_merge.date_x.isna()]\
	.drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

df_merge = df_merge[df_merge.date_x.notna() & df_merge.date_y.notna()]

# -------------------- Second merge
# ensure 1-1 merge
x_dupl = res_x.duplicated(subset=['athlete', 'date_x', 'distance_km_x'], keep='first')
y_dupl = res_y.duplicated(subset=['RIDER', 'date_y', 'distance_km_y'], keep='first')

x = res_x[~x_dupl] ; res_x = res_x[x_dupl]
y = res_y[~y_dupl] ; res_y = res_y[y_dupl]

# second merge (repeat but then with the residues)
df_merge2 = pd.merge(x, y, how='outer', 
	left_on=['athlete', 'date_x', 'distance_km_x'], right_on=['RIDER', 'date_y', 'distance_km_y'], validate="one_to_one")

# get residue
res_x = res_x.append(df_merge2[df_merge2.date_y.isna()]\
	.drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge2[df_merge2.date_x.isna()]\
	.drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

df_merge2 = df_merge2[df_merge2.date_x.notna() & df_merge2.date_y.notna()]

# -------------------- Third merge
# get not nan distances
x_dist = res_x.distance_km_x.notna() & (res_x.distance_km_x != 0)
y_dist = res_y.distance_km_y.notna() & (res_y.distance_km_y != 0)

x = res_x[x_dist] ; res_x = res_x[~x_dist]
y = res_y[y_dist] ; res_y = res_y[~y_dist]

x['distance_km1_x'] = (x['distance_km_x'] / 10).round()
y['distance_km1_y'] = (y['distance_km_y'] / 10).round()

# ensure 1-1 merge
x_dupl = x.duplicated(['athlete', 'date_x', 'distance_km1_x'], keep=False)
y_dupl = y.duplicated(['RIDER', 'date_y', 'distance_km1_y'], keep=False)

x = x[~x_dupl] ; res_x = res_x.append(x[x_dupl].drop('distance_km1_x', axis=1))
y = y[~y_dupl] ; res_y = res_y.append(y[y_dupl].drop('distance_km1_y', axis=1))

# third: merge not nan distances with lower accuracy
df_merge3 = pd.merge(x, y, how='outer', 
	left_on=['athlete', 'date_x', 'distance_km1_x'], right_on=['RIDER', 'date_y', 'distance_km1_y'], validate="one_to_one")

# get residue
res_x = res_x.append(df_merge3[df_merge3.date_y.isna()]\
	.drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'distance_km1_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge3[df_merge3.date_x.isna()]\
	.drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'distance_km1_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

res_x.drop('distance_km1_x', axis=1, inplace=True)
res_y.drop('distance_km1_y', axis=1, inplace=True)

df_merge3 = df_merge3[df_merge3.date_x.notna() & df_merge3.date_y.notna()]
df_merge3.drop(['distance_km1_x', 'distance_km1_y'], axis=1, inplace=True)

# -------------------- Fourth merge
# get not nan distances
x_dist = res_x.distance_km_x.notna() & (res_x.distance_km_x != 0)
y_dist = res_y.distance_km_y.notna() & (res_y.distance_km_y != 0)

x = res_x[x_dist] ; res_x = res_x[~x_dist]
y = res_y[y_dist] ; res_y = res_y[~y_dist]

# ensure 1-1 merge
x_dupl = x.duplicated(['athlete', 'date_x'], keep=False)
y_dupl = y.duplicated(['RIDER', 'date_y'], keep=False)

x = x[~x_dupl] ; res_x = res_x.append(x[x_dupl])
y = y[~y_dupl] ; res_y = res_y.append(y[y_dupl])

# fourth merge: merge only on athlete, date for training sessions with not nan distance
df_merge4 = pd.merge(x, y, how='outer',
	left_on=['athlete', 'date_x'], right_on=['RIDER', 'date_y'], validate='one_to_one')

# get residue
res_x = res_x.append(df_merge4[df_merge4.date_y.isna()]\
	.drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge4[df_merge4.date_x.isna()]\
	.drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

df_merge4 = df_merge4[df_merge4.date_x.notna() & df_merge4.date_y.notna()]

# check if on the same dates, there are more files
for _, (athlete, date) in df_merge4[['athlete', 'date_x']].iterrows():
	res_dates = res_x[(res_x.athlete == athlete) & (res_x.date_x == date)]
	if not res_dates.empty: 
		print(res_dates)

# manually check dates that seem to be incorrectly matched
res_x = res_x.append(df_merge4.loc[[1,4,7,29]].drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge4.loc[[1,4,7,29]].drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

df_merge4.drop([1,4,7,29], inplace=True)

# -------------------- Fifth merge
# get nan distances
x_dist = res_x.distance_km_x.isna() | (res_x.distance_km_x == 0)
y_dist = res_y.distance_km_y.isna() | (res_y.distance_km_y == 0)

x = res_x[x_dist] ; res_x = res_x[~x_dist]
y = res_y[y_dist] ; res_y = res_y[~y_dist]

# ensure 1-1 merge
x_dupl = x.duplicated(['athlete', 'date_x'], keep=False)
y_dupl = y.duplicated(['RIDER', 'date_y'], keep=False)

x = x[~x_dupl] ; res_x = res_x.append(x[x_dupl])
y = y[~y_dupl] ; res_y = res_y.append(y[y_dupl])

# fifth: merge only on athlete, date for training sessions with nan distance
df_merge5 = pd.merge(x, y, how='outer',
	left_on=['athlete', 'date_x'], right_on=['RIDER', 'date_y'], validate='one_to_one')

# get residue
res_x = res_x.append(df_merge5[df_merge5.date_y.isna()]\
	.drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge5[df_merge5.date_x.isna()]\
	.drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

df_merge5 = df_merge5[df_merge5.date_x.notna() & df_merge5.date_y.notna()]

# -------------------- Sixth merge
# get nan distances
x_dist = res_x.distance_km_x.isna() | (res_x.distance_km_x == 0)
y_dist = res_y.distance_km_y.isna() | (res_y.distance_km_y == 0)

x = res_x[x_dist] ; res_x = res_x[~x_dist]
y = res_y[y_dist] ; res_y = res_y[~y_dist]

# ensure 1-1 merge
x_dupl = x.duplicated(['athlete', 'date_x'], keep=False)
y_dupl = y.duplicated(['RIDER', 'date_y'], keep=False)

x = x[~x_dupl] ; res_x = res_x.append(x[x_dupl])
y = y[~y_dupl] ; res_y = res_y.append(y[y_dupl])

# sixth: repeat merge only on athlete, date for training sessions with nan distance
df_merge6 = pd.merge(x, y, how='outer',
	left_on=['athlete', 'date_x'], right_on=['RIDER', 'date_y'], validate='one_to_one')

# get residue
res_x = res_x.append(df_merge6[df_merge6.date_y.isna()]\
	.drop(['RIDER', 'date_y', 'cal_id', 'distance_km_y', 'n_workout_y'], axis=1))
res_y = res_y.append(df_merge6[df_merge6.date_x.isna()]\
	.drop(['athlete', 'date_x', 'file_id', 'distance_km_x', 'n_workout_x'], axis=1))

res_x = res_x.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
res_y = res_y.sort_values(['RIDER', 'date_y', 'n_workout_y']).reset_index(drop=True)

df_merge6 = df_merge6[df_merge6.date_x.notna() & df_merge6.date_y.notna()]

# -------------------- Combine
df_merge = pd.concat([df_merge, df_merge2, df_merge3, df_merge4, df_merge5, df_merge6])
df_merge = df_merge.sort_values(['athlete', 'date_x', 'n_workout_x']).reset_index(drop=True)
df_merge = df_merge[['athlete', 'date_x', 'file_id', 'cal_id', 'distance_km_x', 'distance_km_y', 'n_workout_x', 'n_workout_y']]

df_merge.to_csv('./merge.csv', index_label=False)
res_x.to_csv('./res_x.csv', index_label=False)
res_y.to_csv('./res_y.csv', index_label=False)