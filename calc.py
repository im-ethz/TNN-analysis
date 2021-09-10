import numpy as np
import pandas as pd

from flirt.stats.common import FUNCTIONS as flirt_functions

# glucose levels in mg/dL
glucose_levels = {'hypo L2': (0,53),
				  'hypo L1': (54,69),
				  'normal' : (70,180),
				  'hyper L1': (181,250),
				  'hyper L2': (251,10000)}

mmoll_mgdl = 18#.018
mgdl_mmoll = 0.0555

def calc_hr_zones(LTHR:float) -> list:
	# Coggan power zones
	# https://www.trainingpeaks.com/blog/power-training-levels/
	# 1 - Active Recovery
	# 2 - Endurance
	# 3 - Tempo
	# 4 - Lactate Threshold
	# 5 - VO2Max
	LTHR = LTHR[0]
	return [0.69*LTHR, 0.84*LTHR, 0.95*LTHR, 1.06*LTHR]

def calc_power_zones(FTP:float) -> list:
	# Coggan power zones
	# https://www.trainingpeaks.com/blog/power-training-levels/
	# 1 - Active Recovery
	# 2 - Endurance
	# 3 - Tempo
	# 4 - Lactate Threshold
	# 5 - VO2Max
	# 6 - Anaerobic Capacity
	FTP = FTP[0]
	return [0.56*FTP, 0.76*FTP, 0.91*FTP, 1.06*FTP, 1.21*FTP]

def time_in_zone(X:pd.Series, zones:list) -> list:
	# For each training session, calculate heart rate zones
	time_in_zone = np.zeros(len(zones)+1)
	for n, (z_min, z_max) in enumerate(zip([0]+zones, zones+[1e6])):
		time_in_zone[n] = ((X >= z_min) & (X < z_max)).sum()
	return time_in_zone

def elevation_gain(altitude: pd.Series):
	# calculate the total elevation gain during a workout
	return altitude.diff()[altitude.diff() > 0].sum()

def elevation_loss(altitude: pd.Series):
	# calculate the total elevation loss during a workout
	return altitude.diff()[altitude.diff() < 0].sum()

"""
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
"""

def normalised_power(power: pd.Series) -> float:
	# TODO: should I ignore 0s?
	# calculate normalised power (NP) for each training individually
	# make sure power is a pandas series with a monotonic datetimeindex
	return power.rolling('30s', min_periods=30).mean().pow(4).mean()**(1/4)

def intensity_factor(NP, FTP) -> float:
	# calculate intensity factor (IF) as ratio between normalised power (NP) and functional threshold power (FTP)
	return NP/FTP

def training_stress_score(T, IF) -> float:
	# calculate training stress score (TSS) using duration of workout in seconds (T) and intensity factor (IF)
	return 100 * (T/3600) * IF**2 

def variability_index(NP:float, power: pd.Series) -> float:
	# calculate variability index (VI) using normalised power (NP) and power
	# for each training individually
	return NP / power.mean()

def efficiency_factor(NP:float, heart_rate: pd.Series) -> float:
	# calculate efficiency factor (EF) using normalised power (NP) and heart rate
	# for each training individually
	return NP / heart_rate.mean()

def chronic_training_load(TSS: pd.Series):
	# calculate chronic training load (CTL) (fitness)
	return TSS.ewm(span=42).mean()

def acute_training_load(TSS: pd.Series):
	# calculate acute training load (ATL) (fatigue)
	return TSS.ewm(span=7).mean()

def training_stress_balance(CTL, ATL):
	# calculate the training stress balance (TSB) (form) from chronic training load (CTL) and acute training load (ATL)
	return CTL - ATL

def agg_power(X:pd.DataFrame, FTP):
	# For each training session, calculate power characteristics
	T = len(X)
	NP = normalised_power(X['power'])
	IF = intensity_factor(NP, FTP)
	TSS = training_stress_score(T, IF)
	VI = variability_index(NP, X['power'])
	EF = efficiency_factor(NP, X['heart_rate'])

	return pd.Series({'normalised_power'		: NP,
					  'intensity_factor'		: IF,
					  'training_stress_score'	: TSS,
					  'variability_index'		: VI,
					  'efficiency_factor'		: EF})

def agg_zones(X:pd.DataFrame, hr_zones:list, power_zones:list):
	time_in_hr_zone = {'time_in_hr_zone%s'%(n+1):t for n, t in enumerate(time_in_zone(X['heart_rate'], hr_zones))}
	time_in_power_zone = {'time_in_power_zone%s'%(n+1):t for n, t in enumerate(time_in_zone(X['power'], power_zones))}
	return pd.Series({**time_in_hr_zone, **time_in_power_zone})

def agg_stats(X:pd.DataFrame):
	# apply flirt statistics to every column of data
	results = {}
	for col in X:
		res = {}
		for name, func in flirt_functions.items():
			res[name] = func(X[col])
		results[col] = pd.Series(res)
	return pd.concat(results)