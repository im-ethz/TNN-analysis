#!/bin/bash
path='2019/2 - Peron/'

mkdir "${path}fit/"
mkdir "${path}csv/"

for filepath in "${path}raw/"*.gz
do
	filename=$(basename "${filepath}" .fit.gz)
	echo "${filename}"
	gzip -dk "${filepath}"
	mv "${path}raw/${filename}.fit" "${path}fit/${filename}.fit"
	python3 parse_fit_to_csv.py "${filename}.fit" -i "${path}fit/${i}/" -o "${path}csv/${i}/"
done