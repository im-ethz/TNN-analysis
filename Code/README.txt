How to get the data:

0. find login information in config.py

1. scrape TrainingPeaks using scrape_trainingpeaks.py (select a date-range and workout-type bike)
	a) for each athlete, move the training files to a subdirectory with their name in the "raw" folder
	b) make sure everything is within the right time-range and remove duplicates by searching for (1) (2), etc.
	c) anonymize the data of each athlete by mapping their name to a number and put this data in the "raw_anonymous" folder. The mapping can be uploaded to "mapping.xls"

2. convert TrainingPeaks fit.gz files to csv using parse_fit_to_csv.sh:
	a) extract every .fit.gz file to .fit
	b) convert fit file to csv with parse_fit_to_csv.py

3. preprocess TODO

4. login at LibreView and download athlete data manually

5. merge TrainingPeaks (cycling) data with LibreView (glucose) data
