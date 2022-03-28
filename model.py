from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

from itertools import combinations

from plot import *

class GLMM:
	def __init__(self, family='binomial', interactions=False, random_effects=None, verbose=0):
		"""
		Generalized Linear Mixed Model using lme4 in R

		Arguments
			family - model family (link function)
			interactions - whether to include interactions for all variables
			random effects - list of variables for which to include a random effect
			verbose - degree of which to display intermediate results
		"""
		self.family = family
		self.interactions = interactions
		self.random_effects = random_effects
		self.verbose = verbose

	def get_formula(self, X, y):
		"""
		Get regression formula based on data given
		Note: current implementation only allows for interactions of all variables

		Arguments
			X - independent variables
			y - dependent variable

		Note:
			formula_y - dependent part of formula
			formula_fix - fixed effects formula
			formula_int - interactions formula
			formula_rnd - random effects formula
		"""
		cols_X = X.columns if type(X) == pd.DataFrame else [X.name]
		groups = list(X.index.names)[0]
			
		# get formula
		formula_y = y.columns[0] + '~'
		formula_fix = '1+' + '+'.join(cols_X)
		formula_int = ''.join([f'+{col_A}*{col_B}' if self.interactions else '' for col_A, col_B in combinations(cols_X, 2)])
		if self.random_effects:
			formula_rnd = '+(1+' + '+'.join(self.random_effects)+f'|{groups})'
		else:
			formula_rnd = f'+(1|{groups})'

		self.formula = ''.join((formula_y, formula_fix, formula_int, formula_rnd))

		if self.verbose > 0:
			print(self.formula)

	def fit(self, X, y, path):
		"""
		Arguments
			X - independent variables
			y - dependent variable
		"""
		base = importr('base')
		stats = importr('stats')
		lme4 = importr('lme4')

		self.get_formula(X, y)

		data = pd.concat([X, y], axis=1).reset_index()
		with localconverter(ro.default_converter + pandas2ri.converter):
			r_df = ro.conversion.py2rpy(data)
			
		self.model = lme4.glmer(self.formula, data=r_df, family=self.family)
		
		base.sink(path)
		print(base.summary(self.model))
		base.sink()  # returns output to the console

class Regress:
	def __init__(self, event, family='binomial', experiment='', categories={}, sorting=[], root='', verbose=0):
		"""
		Arguments
			event - event of dependent variable (i.e. either hypo or hyper)
			family - model family used in the regression
			experiment - name of the experiment we are running
			categories - column categories
			root - where to save everything
			verbose - degree of which to display intermediate results
		"""
		self.event = event
		self.family = family
		self.experiment = experiment
		self.categories = categories
		self.sorting = sorting
		self.root = root
		self.verbose = verbose

		self.filename = f"model_{self.experiment}_{self.family}_{self.event}"
		self.sections = ('exercise', 'recovery', 'sleep')

	def fit(self, data, x, name, **kwargs):
		"""
		Run regression and read and save results

		Arguments
			data - data used for regression
			x - (list) independent variables
			name - name of the regression (often just independent variables, e.g. "duration")
			kwargs - any arguments passed to GLMM (e.g. whether to use interactions or random effects)

		Returns
			fe - fixed effects results of three sections
			re - random effects results of three sections
			score - evaluation of model of three sections
			res - residuals of three sections
		"""
		self.name = name

		if self.verbose > 1:
			print(self.event.upper())
		
		for sec in self.sections:
			y = f'GLUCOSE_{self.event}_{sec}'
			
			self.model = GLMM(family=self.family, verbose=self.verbose, **kwargs)
			self.model.fit(X=data[x], y=data[[y]],
				path=f'{self.root}{self.filename}_{self.name}_{sec}.txt')

	def plot_hist(self, data, x):
		sns.kdeplot(data=data, x=x[0], hue='RIDER', palette=palette_ath)
		plt.show()

	def plot_box(self, data, x):
		for sec in self.sections:
			y = f'GLUCOSE_{self.event}_{sec}'
			fig, ax = plt.subplots(figsize=(8,4))
			sns.boxplot(ax=ax, data=data.reset_index(), x='RIDER', y=x[0], hue=y)
			plt.show()

	def read(self, filename):
		"""
		Read regression results from R to pandas table in python
		
		Arguments
			filename - full filename where regression results are stored
		
		Returns
			fe - fixed effects results
			re - random effects results
			score - evaluation of model
			res - residuals
		"""
		with open(filename) as f:
			lines = f.readlines()

		# identify start and end
		start_score = np.where([l.lstrip().startswith('AIC') for l in lines])[0][0]
		start_res = np.where([l.startswith('Scaled residuals') for l in lines])[0][0]
			
		start = np.where([l.startswith('Random effects') for l in lines])[0][0]
		mid = np.where([l.startswith('Fixed effects') for l in lines])[0][0]
		end = np.where([l.startswith(('Signif. codes', 'Correlation of Fixed Effects')) for l in lines])[0][0]

		# read csv
		fe = pd.read_csv(filename, skiprows=mid+1, skipfooter=len(lines)-end+1, 
						 delimiter="\s+(?!<)", engine='python')
		re = pd.read_csv(filename, skiprows=start+1, skipfooter=len(lines)-mid+2, 
						 delimiter="\s+(?!<)", engine='python')
		fe.columns = ['Feature', 'Estimate', 'Std. Error', 'z value', 'Pr(>|z|)', 'Sign']
		
		score = pd.read_csv(filename, skiprows=start_score, skipfooter=len(lines)-start_score-3, 
						delimiter="\s+(?!<)", engine='python')
		res = pd.read_csv(filename, skiprows=start_res+1, skipfooter=len(lines)-start_res-4, 
						delimiter="\s+(?!<)", engine='python')
		
		return fe, re, score, res

	def get_results(self, name_map, cols_env):
		"""
		Combine the results of the regressions on three sections: exercise, recovery and sleep
		Read the regression results, calculate 95% CI, and convert the results to the appropriate format
					
		Returns
			fe - fixed effects results of three sections
			co - fixed effects results of three sections for the controlling variables
			re - random effects results of three sections
			score - evaluation of model of three sections
			res - residuals of three sections
		"""
		# read model results from files
		fe, re, score, res = {}, {}, {}, {}
		for sec in self.sections:
			fe[sec], re[sec], score[sec], res[sec] = self.read(f'{self.root}{self.filename}_{self.name}_{sec}.txt')
			fe[sec] = fe[sec].set_index('Feature')
			
			re[sec] = re[sec].T.reset_index()
			for col in re[sec].columns.drop(['index', 0]):
				re[sec][col] = re[sec][col].shift()
				re[sec][col] = re[sec][col].fillna(re[sec].loc[0,0])
			re[sec] = re[sec].set_index('index').T
			
		fe = pd.concat(fe)
		re = pd.concat(re)
		score = pd.concat(score, axis=1)
		res = pd.concat(res, axis=1)

		# calculate upper and lower 95% CI boundary
		fe['CI_lower'] = fe['Estimate'] - 1.96*fe['Std. Error']
		fe['CI_upper'] = fe['Estimate'] + 1.96*fe['Std. Error']
		
		# transform everything from log-odds to odds
		fe['CI_lower'] = np.exp(fe['CI_lower'])
		fe['CI_upper'] = np.exp(fe['CI_upper'])
		fe['Estimate'] = np.exp(fe['Estimate'])
		
		# round off
		cols_fe = ['Estimate', 'CI_lower', 'CI_upper']
		fe[cols_fe] = fe[cols_fe].round(2)
		fe['Pr(>|z|)'] = fe['Pr(>|z|)'].round(3)
		fe['Pr(>|z|)'] = fe['Pr(>|z|)'].replace({0.000: '<0.001'})
		cols_fe += ['Pr(>|z|)']
		
		# convert format
		cols_fe += ['Sign']
		fe = fe[cols_fe].reset_index().pivot(index='Feature', columns='level_0')
		fe.columns = fe.columns.swaplevel(0,1)
		fe = fe[pd.MultiIndex.from_product([fe.columns.get_level_values(0).unique(), cols_fe])]
		fe.columns.names = [None, None]

		cols_0 = {**{"(Intercept)": "Intercept"}, **name_map}
		cols_re = re.columns.drop(['Groups', 'Name'])
		re = re.reset_index().pivot(index=['Groups', 'Name'], columns=['level_0'], values=cols_re)
		re.columns = re.columns.swaplevel(0,1)
		re = re[pd.MultiIndex.from_product([re.columns.get_level_values(0).unique(), cols_re])]
		re.columns.names = [None, None]
		re = re.round(2)
		re = re.rename(index=cols_0)
		
		cols_0.update(cols_env)
		fe = fe.sort_index(key = lambda x: x.map({key:k+1 for k, key in enumerate(cols_0.keys())}))
		fe = fe.rename(index=cols_0)
		fe = fe.replace({None:''})
		fe = fe.replace({'.':''})
		
		co = fe.loc[['Intercept']+list(cols_env.values())]
		fe = fe.drop(co.index)
		return fe, co, re, score, res

	def transform_fe(self, fe, output=''):
		fe_new = pd.DataFrame(columns=fe.columns)
		for cat, idx in self.categories.items():
			if output == 'table':
				row = pd.DataFrame(index=[r'\multicolumn{2}{@{} l}{\textit{'+cat+'}}'], columns=fe.columns)
			else:
				row = pd.DataFrame(index=[' ', cat], columns=fe.columns)
			fe_new = pd.concat([fe_new, row, fe.loc[idx]])
		#fe_new = fe_new.iloc[1:]
		return fe_new

	def inv_transform_fe(self, fe):
		fe = fe.dropna(how='all')
		return fe

	def rename_time(self, x):
		if x.startswith('Time in'):
			return '('+' '.join(x.split(' (')[::-1])
		else:
			return x

	def inv_rename_time(self, x):
		if 'Time in' in x:
			return ' '.join(x.split(') ')[::-1])+')'
		else:
			return x

	def transform_cat(self):
		self.categories = pd.Series({v:key for key,values in self.categories.items() for v in values}).rename(index=self.rename_time).reset_index()
		self.categories = self.categories.groupby(0)['index'].apply(list)
		self.categories = self.categories.sort_index(key=lambda x: x.map({col:i for i, col in enumerate(self.sorting)})).to_dict()

	def inv_transform_cat(self):
		self.categories = pd.Series({v:key for key,values in self.categories.items() for v in values}).rename(index=inv_rename_time).reset_index()
		self.categories = self.categories.groupby(0)['index'].apply(list).to_dict()

	def read_tables(self):
		fe = pd.read_csv(f"{self.root}{self.filename}_fe.csv", index_col=0, header=[0,1])
		fe = self.inv_transform_fe(fe)
		for col in np.unique(fe.columns.get_level_values(0)):
			fe.loc[:, (col,'Pr(>|z|)' )] = fe.loc[:, (col,'Pr(>|z|)' )].apply(lambda x: float(x) if not str(x).startswith('<') else x)
		fe = fe.replace({np.nan: ''})

		re = pd.read_csv(f"{self.root}{self.filename}_re.csv", index_col=[0,1,2], header=[0,1])

		# rename fe and categories for plotting
		self.transform_cat()
		fe = fe.rename(index=self.rename_time)
		return fe, re

	def save_tables(self, fe, re):
		with open(f"{self.root}{self.filename}.tex", 'w') as file:
			file.write(self.transform_fe(fe, output='table').to_latex(column_format='c', escape=False))
			file.write(re.to_latex(column_format='c', escape=False))
		self.transform_fe(fe).to_csv(f"{self.root}{self.filename}_fe.csv")
		re.to_csv(f"{self.root}{self.filename}_re.csv")

		# rename fe and categories for plotting
		self.transform_cat()
		fe = fe.rename(index=self.rename_time)
		return fe