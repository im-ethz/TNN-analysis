import numpy as np
import pandas as pd

month_mapping = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july',
				 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}

def create_calendar_array(df, datacol):
	"""
	Create array in the form of calendar (month x days)
	df 		- pandas DataFrame with dates as index (for only one year!)
	"""
	df['month'] = pd.to_datetime(df.index).month
	df['day'] = pd.to_datetime(df.index).day
	df_calendar = df.pivot('month','day', datacol)
	df_calendar.index = pd.CategoricalIndex(df_calendar.index.map(month_mapping), 
		ordered=True, categories=month_mapping.values())
	return df_calendar