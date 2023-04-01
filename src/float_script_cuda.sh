#!/bin/bash
ns=(128 256 512 1024 2048 4096 8192 16384 32768 65536)
blockSizeBubbleSorts=(128 256 512 1024)
blockSizeDotProds=(128 256 512 1024)
Experiments=(1 2 3)
tmp=float_main.tmp.cu
for Experiment in ${Experiments[@]}
do
	echo "Experiment $Experiment"
	for blockSizeBubbleSort in ${blockSizeBubbleSorts[@]}
	do
		echo "blockSizeBubbleSort $blockSizeBubbleSort"
		for blockSizeDotProd in ${blockSizeDotProds[@]}
		do
			echo "blockSizeDotProd $blockSizeDotProd"
			fn=../data/$Experiment.$blockSizeBubbleSort.$blockSizeDotProd.txt
			echo "" > $fn
			echo "" > $tmp
			echo "#define blockSizeBubbleSort $blockSizeBubbleSort" >> $tmp
			echo "#define blockSizeDotProd $blockSizeDotProd" >> $tmp
			cat float_main.cu >> $tmp
			nvcc $tmp -o a.out
			for n in ${ns[@]}
			do
				echo "n $n"
				./a.out $n >> $fn				
			done
			
		done
	done
done
