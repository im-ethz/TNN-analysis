import numpy as numpy
import pandas as pd

df = pd.read_excel('2019_RaceCalendar_createdEva.xls')

df.set_index(['name', 'country', 'start_date', 'end_date'], inplace=True)

df = df.T

race_list = df.apply(lambda x: df.columns.get_level_values(0)[x == 1].values, axis=1)
countries_list = df.apply(lambda x: df.columns.get_level_values(1)[x == 1].values, axis=1)

for i in race_list.index:
	print("-------------- RIDER ", i)
	for race in race_list[i]:
		print(race)

for i in countries_list.index:
	print("-------------- RIDER ", i)
	for country in countries_list[i]:
		print(country)