#!/bin/bash

path='Data/TrainingPeaks/'
athletes=("_test")"1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15")

mkdir "${path}fit/"
mkdir "${path}csv/"
for i in $athletes
do
	mkdir "${path}fit/${i}"
	mkdir "${path}csv/${i}"
	for filepath in "${path}RAW/${i}/"*.gz
	do
		filename=$(basename "${filepath}" .fit.gz)
		echo "${filename}"
		gzip -dk "${filepath}"
		mv "${path}RAW/${i}/${filename}.fit" "${path}fit/${i}/${filename}.fit"
		python parse_fit_to_csv.py "${filename}.fit" -i "${path}fit/${i}/" -o "${path}csv/${i}/"
	done
	mkdir "${path}fit/${i}/Zwift"
	mkdir "${path}csv/${i}/Zwift"
	for filepath in "${path}RAW/${i}/Zwift/"*.gz
	do
		filename=$(basename "${filepath}" .fit.gz)
		echo "${filename}"
		gzip -dk "${filepath}"
		mv "${path}RAW/${i}/Zwift/${filename}.fit" "${path}fit/${i}/Zwift/${filename}.fit"
		python parse_fit_to_csv.py "${filename}.fit" -i "${path}fit/${i}/Zwift/" -o "${path}csv/${i}/Zwift/"
	done	
done