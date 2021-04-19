import numpy as np
import pandas as pd

# glucose levels in mg/dL
glucose_levels = {'hypo L2': (0,53),
				  'hypo L1': (54,69),
				  'normal' : (70,180),
				  'hyper L1': (181,250),
				  'hyper L2': (251,10000)}
mmoll_mgdl = 18#.018
mgdl_mmoll = 0.0555

def elevation_gain(altitude: pd.Series):
	# calculate the total elevation gain during a workout
	return altitude.diff()[altitude.diff() > 0].sum()

def elevation_loss(altitude: pd.Series):
	# calculate the total elevation loss during a workout
	return altitude.diff()[altitude.diff() < 0].sum()

def grade(altitude, distance):
	# TODO
	# calculate the difference between distance travelled vertically and horizontally expressed as a percentage
	return

def velocity_ascended_in_mph(elevation_gain, t, grade):
	# calculate velocity ascended in m/h (VAM), which measures how fast you are climbing a hill
	# using elevation_gain (meters ascended/hour), t (duration in seconds) and grade (% increase)
	return (elevation_gain/(t/3600))/(200+10*(grade*100))

def work(power: pd.Series):
	# W = integral P dt
	# TODO: make sure there are no missing values here
	return power.sum()

def normalised_power(power: pd.Series) -> pd.Series:
	# TODO: should I ignore 0s?
	# calculate normalised power (NP) for each training individually
	# make sure power is a pandas series with a monotonic datetimeindex
	return power.rolling('30s', min_periods=30).mean().pow(4).mean()**(1/4)

def intensity_factor(NP, FTP):
	# calculate intensity factor (IF) as ratio between normalised power (NP) and functional threshold power (FTP)
	return NP/FTP

def training_stress_score(T, IF):
	# calculate training stress score (TSS) using duration of workout in seconds (T) and intensity factor (IF)
	return 100 * (T/3600) * IF**2 

def chronic_training_load(TSS: pd.Series):
	# calculate chronic training load (CTL) (fitness)
	return TSS.ewm(span=42).mean()

def acute_training_load(TSS: pd.Series):
	# calculate acute training load (ATL) (fatigue)
	return TSS.ewm(span=7).mean()

def training_stress_balance(CTL, ATL):
	# calculate the training stress balance (TSB) (form) from chronic training load (CTL) and acute training load (ATL)
	return CTL - ATL

def variability_index(NP, power: pd.Series):
	# calculate variability index (VI) using normalised power (NP) and power
	# for each training individually
	return NP / power.mean()

def efficiency_factor(NP, heart_rate: pd.Series):
	# calculate efficiency factor (EF) using normalised power (NP) and heart rate
	# for each training individually
	return NP / heart_rate.mean()
