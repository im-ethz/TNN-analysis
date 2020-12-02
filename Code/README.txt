How to get the data:
0. find login information in config.py
1. scrape TrainingPeaks using scrape_trainingpeaks.py
2. login at LibreView and download athlete data manually
3. convert TrainingPeaks fit.gz files to csv using parse_fit_to_csv.sh:
	a) extract every .fit.gz file to .fit
	b) convert fit file to csv with parse_fit_to_csv.py
4. merge TrainingPeaks (cycling) data with LibreView (glucose) data