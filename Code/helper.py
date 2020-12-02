import numpy as np
import pandas as pd

def create_calendar_array(df, datacol):
	"""
	Create array in the form of calendar (month x days)
	df 		- pandas DataFrame with dates as index (for only one year!)
	"""
	df['month'] = pd.to_datetime(df.index).month
	df['day'] = pd.to_datetime(df.index).day
	return df.pivot('month','day', datacol)