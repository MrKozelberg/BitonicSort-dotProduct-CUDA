#!/bin/bash
cudacode=main.cu
cpucode=main.cpp
cudaout=main_cuda.out
cpuout=main_cpu.out
nvcc $cudacode -o $cudaout
g++ $cpucode -o $cpuout
arraySizes=(1024 4096 16384 65536 262144 1048576 4194304)
cudadf=../data/cuda_experiments.csv
cpudf=../data/cpu_experiments.csv
echo "" > $cudadf
echo "" > $cpudf
experiments=(1 2 3 4 5)
for arraySize in ${arraySizes[@]}
do
	for experiment in ${experiments[@]}
	do
		echo "$arraySize,$experiment"
		./$cudaout $arraySize >> $cudadf
		./$cpuout $arraySize >> $cpudf
	done
done
rm *.out