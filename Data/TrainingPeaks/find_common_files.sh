#!/bin/bash
path='2019/'
athletes=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18")

for i in ${athletes[*]}
do
	for j in ${athletes[*]}
	do
		if [ "${i}" != "${j}" ]
		then
			echo "${i}" "${j}"
			comm -12 <(ls "${path}/raw_anonymous/${i}") <(ls "${path}/raw_anonymous/${j}")
			#comm -12 <(ls "${path}/raw_anonymous/${i}") <(ls "${path}/raw_anonymous/${j}") | while read file
			#do
			#	diff "${path}/csv/${i}/data/${file::-7}_data.csv" "${path}/csv/${j}/data/${file::-7}_data.csv"
			#	#awk 'BEGIN{FS=","};FNR==NR{a[$1];next};!($1 in a)' file1 file2
			#done
		fi
	done
done