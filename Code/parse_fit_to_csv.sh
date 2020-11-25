#!/bin/bash

path='Data/TrainingPeaks/'
athletes=("_test")"1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15")

mkdir "${path}fit/"
for i in $athletes
do
	mkdir "${path}fit/${i}"
	for filepath in "${path}RAW/${i}/"*.gz
	do
		filename=$(basename "${filepath}" .fit.gz)
		echo "${filename}"
		gzip -dk "${filepath}"
		mv "${path}RAW/${i}/${filename}.fit" "${path}fit/${i}/${filename}.fit"
	done
	mkdir "${path}fit/${i}/Zwift"
	for filepath in "${path}RAW/${i}/Zwift/"*.gz
	do
		filename=$(basename "${filepath}" .fit.gz)
		echo "${filename}"
		gzip -dk "${filepath}"
		mv "${path}RAW/${i}/Zwift/${filename}.fit" "${path}fit/${i}/Zwift/${filename}.fit"
	done
done
