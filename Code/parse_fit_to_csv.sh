#!/bin/bash
path='Data/TrainingPeaks/2019/'
path_raw='raw_anonymous/'
athletes=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18")

mkdir "${path}fit/"
mkdir "${path}csv/"
for i in ${athletes[*]}
do
	mkdir "${path}fit/${i}"
	mkdir "${path}csv/${i}"
	echo "Processing athlete ${i}"
	for filepath in "${path}${path_raw}${i}/"*.gz
	do
		filename=$(basename "${filepath}" .fit.gz)
		echo "${filename}"
		gzip -dk "${filepath}"
		mv "${path}${path_raw}${i}/${filename}.fit" "${path}fit/${i}/${filename}.fit"
		python3 parse_fit_to_csv.py "${filename}.fit" -i "${path}fit/${i}/" -o "${path}csv/${i}/"
	done
done