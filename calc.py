import numpy as np
import pandas as pd
from copy import copy

from flirt.stats.common import FUNCTIONS as flirt_functions

# keep only a selected number of functions to reduce data size
del flirt_functions['ptp'] # remove because of multicollinearity
del flirt_functions['skewness'], flirt_functions['kurtosis'] # redundant
del flirt_functions['lineintegral'] # mostly nan
del flirt_functions['n_above_mean'], flirt_functions['n_below_mean'],\
	flirt_functions['n_sign_changes'], flirt_functions['iqr_5_95'], \
	flirt_functions['pct_5'], flirt_functions['pct_95'] # redundant

# the following functions should be left: mean, std, min, max, sum, energy, iqr, line_integral

def timestamp_to_seconds(timestamp:pd.Series):
	return (pd.to_datetime(timestamp) - pd.to_datetime(pd.to_datetime(timestamp).dt.date)).dt.seconds

# glucose levels in mg/dL
glucose_levels = {'hypo L2': (0,53),
				  'hypo L1': (54,69),
				  'target' : (70,180),
				  'hyper L1': (181,250),
				  'hyper L2': (251,10000)}
# perform some adjustments to the glucose levels, so that floats don't fall in between glucose levels
# e.g. 69.5 falls in between 69 and 70
glucose_levels_ = {level: (lmin-(1-1e-8), lmax) if level.startswith('hyper') else (
						  (lmin, lmax+(1-1e-8)) if level.startswith('hypo') else (
						  (lmin, lmax))) for level, (lmin, lmax) in glucose_levels.items()}
glucose_levels_ext = copy(glucose_levels_)
glucose_levels_ext['hypo'] = (glucose_levels_['hypo L2'][0], glucose_levels_['hypo L1'][1])
glucose_levels_ext['hyper'] = (glucose_levels_['hyper L1'][0], glucose_levels_['hyper L2'][1])

mmoll_mgdl = 18
mgdl_mmoll = 1/mmoll_mgdl

def hypo(X:pd.Series):
	"""
	Calculate hypo according to definition of https://doi.org/10.2337/dc17-1600
	Note: make sure your data has the following shape
	- data is sorted by timestamp
	- data occurs at a frequency of 5 minutes (e.g. from Dexcom or Medtronic devices)
	- missing data should still have a timestamp entry, but its associated glucose value should be NaN
	"""
	res = (X < glucose_levels_['target'][0]) \
		& (X.shift(1) < glucose_levels_['target'][0]) \
		& (X.shift(2) < glucose_levels_['target'][0])
	res[X.isna() | X.shift(1).isna() | X.shift(2).isna()] = np.nan
	return res

def hyper(X:pd.Series):
	"""
	Calculate hyper according to definition of https://doi.org/10.2337/dc17-1600
	Note: make sure your data has the following shape
	- data is sorted by timestamp
	- data occurs at a frequency of 5 minutes (e.g. from Dexcom or Medtronic devices)
	- missing data should still have a timestamp entry, but its associated glucose value should be NaN
	"""
	res = (X > glucose_levels_['target'][1]) \
		& (X.shift(1) > glucose_levels_['target'][1]) \
		& (X.shift(2) > glucose_levels_['target'][1])
	res[X.isna() | X.shift(1).isna() | X.shift(2).isna()] = np.nan
	return res

def symmetric_scale(X, unit='mgdl'):
	# symmetric scaling for blood glucose
	if unit == 'mgdl':
		return 1.509*(np.log(X)**1.084 - 5.381)
	elif unit == 'mmoll':
		return 1.794*(np.log(X)**1.026 - 1.861)

def LBGI(X):
	# https://doi.org/10.2337/diacare.21.11.1870
	return symmetric_scale(X).apply(lambda x: 10*x**2 if x < 0 else 0).mean()

def HBGI(X):
	# https://doi.org/10.2337/dc06-1085
	return symmetric_scale(X).apply(lambda x: 10*x**2 if x >= 0 else 0).mean()

def time_in_level(x, l, extend=True):
	if extend:
		levels = glucose_levels_ext
	else:
		levels = glucose_levels_
	return ((x >= levels[l][0]) & (x <= levels[l][1])).sum()

def perc_in_level(x, l, extend=True):
	if extend:
		levels = glucose_levels_ext
	else:
		levels = glucose_levels_
	return ((x >= levels[l][0]) & (x <= levels[l][1])).sum() / x.count() * 100

def calc_hr_zones(LTHR:float) -> list:
	# Coggan HR zones
	# https://www.trainingpeaks.com/blog/power-training-levels/
	# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.111.3820&rep=rep1&type=pdf
	# 1 - Active Recovery
	# 2 - Endurance
	# 3 - Tempo
	# 4 - Lactate Threshold
	# 5 - VO2Max
	return [0.69*LTHR, 0.84*LTHR, 0.95*LTHR, 1.06*LTHR]

def calc_power_zones(FTP:float) -> list:
	# Coggan power zones
	# https://www.trainingpeaks.com/blog/power-training-levels/
	# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.111.3820&rep=rep1&type=pdf
	# 1 - Active Recovery
	# 2 - Endurance
	# 3 - Tempo
	# 4 - Lactate Threshold
	# 5 - VO2Max
	# 6 - Anaerobic Capacity
	# Note: the numbers below define the starts of the zones
	return [0.56*FTP, 0.76*FTP, 0.91*FTP, 1.06*FTP, 1.21*FTP]

def time_in_zone(X:pd.Series, zones:list) -> list:
	# For each training session, calculate heart rate zones
	time_in_zone = np.zeros(len(zones)+1)
	for n, (z_min, z_max) in enumerate(zip([0]+zones, zones+[1e6])):
		time_in_zone[n] = ((X >= z_min) & (X < z_max)).sum()
	return time_in_zone

def combine_pedal_smoothness(left, right, balance):
	return left*(balance.clip(0,100)/100) + right*(1-balance.clip(0,100)/100)

def elevation_gain(altitude: pd.Series):
	# calculate the total elevation gain during a workout
	return altitude.diff()[altitude.diff() > 0].sum()

def elevation_loss(altitude: pd.Series):
	# calculate the total elevation loss during a workout
	return altitude.diff()[altitude.diff() < 0].sum()

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

def stats_cgm(x, sec='', col='Glucose Value (mg/dL)'):
	return {'time_in_hypo_'+sec 	: time_in_level(x[col], 'hypo'),
			'time_in_hypoL2_'+sec 	: time_in_level(x[col], 'hypo L2'),
			'time_in_hypoL1_'+sec 	: time_in_level(x[col], 'hypo L1'),
			'time_in_target_'+sec 	: time_in_level(x[col], 'target'),
			'time_in_hyper_'+sec 	: time_in_level(x[col], 'hyper'),
			'time_in_hyperL1_'+sec 	: time_in_level(x[col], 'hyper L1'),
			'time_in_hyperL2_'+sec 	: time_in_level(x[col], 'hyper L2'),
			'perc_in_hypo_'+sec 	: perc_in_level(x[col], 'hypo'),
			'perc_in_hypoL2_'+sec 	: perc_in_level(x[col], 'hypo L2'),
			'perc_in_hypoL1_'+sec 	: perc_in_level(x[col], 'hypo L1'),
			'perc_in_target_'+sec 	: perc_in_level(x[col], 'target'),
			'perc_in_hyper_'+sec 	: perc_in_level(x[col], 'hyper'),
			'perc_in_hyperL1_'+sec 	: perc_in_level(x[col], 'hyper L1'),
			'perc_in_hyperL2_'+sec 	: perc_in_level(x[col], 'hyper L2'),
			'glucose_mean_'+sec 	: x[col].mean(),
			'glucose_std_'+sec 		: x[col].std(),
			'glucose_cv_'+sec 		: x[col].std() / x[col].mean() * 100,
			'glucose_rate_'+sec		: x['glucose_rate'].mean(),
			'completeness_'+sec 	: x[col].count() / x['timestamp'].count(),
			'count_'+sec 			: x[col].count(),
			'LBGI_'+sec 			: LBGI(x[col]),
			'HBGI_'+sec 			: HBGI(x[col]),
			'AUC_'+sec 				: np.trapz(y=x[col], x=x['timestamp']) / np.timedelta64(5, 'm'),
			'hypo_'+sec 			: x['hypo'].any(),
			'hyper_'+sec 			: x['hyper'].any()}

def agg_stats(X:pd.DataFrame):
	# apply flirt statistics to every column of data
	results = {}
	for col in X:
		res = {}
		for name, func in flirt_functions.items():
			res[name] = func(X[col])
		results[col] = pd.Series(res)
	return pd.concat(results)