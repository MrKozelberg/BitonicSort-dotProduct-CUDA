#!/bin/bash
ns=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
Experiments=(1 2 3)
tmp=float_main.cu
nvcc $tmp -o a.out
for Experiment in ${Experiments[@]}
do
	echo "Experiment $Experiment"
	fn=../data/$Experiment.txt
	echo "" > $fn
	for n in ${ns[@]}
	do
		echo "n $n"
		./a.out $n >> $fn
	done
done
