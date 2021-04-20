#!/bin/sh
CUDA=0
EPOCH=50


for MODEL in  CustomMLP_6 CustomMLP_5 CustomMLP_4 LeNet5
do
        python main.py --mode=train --model=$MODEL -e=$EPOCH --cuda=$CUDA
        python main.py --mode=test --model=$MODEL -e=$EPOCH --cuda=$CUDA
        python main.py --mode=graph --model=$MODEL -e=$EPOCH --cuda=$CUDA -s=True -v=False
        python main.py --mode=graph --model=$MODEL -e=$EPOCH --cuda=$CUDA -s=True -v=True

	for VAR in 0.0001  0.001 0.01
	do
        python main.py --mode=train --model=$MODEL -e=$EPOCH -o=weight_decay -w=$VAR --cuda=$CUDA
        python main.py --mode=test --model=$MODEL -e=$EPOCH -o=weight_decay -w=$VAR --cuda=$CUDA
        python main.py --mode=graph --model=$MODEL -e=$EPOCH -o=weight_decay -w=$VAR --cuda=$CUDA -s=True -v=False
        python main.py --mode=graph --model=$MODEL -e=$EPOCH -o=weight_decay -w=$VAR --cuda=$CUDA -s=True -v=True
    done

    for STD in 0.3  0.2 0.1
	do
        python main.py --mode=train --model=$MODEL -e=$EPOCH -o=insert_noise -w=$VAR --cuda=$CUDA
        python main.py --mode=test --model=$MODEL -e=$EPOCH -o=insert_noise -w=$VAR --cuda=$CUDA
        python main.py --mode=graph --model=$MODEL -e=$EPOCH -o=insert_noise -w=$VAR --cuda=$CUDA -s=True -v=False
        python main.py --mode=graph --model=$MODEL -e=$EPOCH -o=insert_noise -w=$VAR --cuda=$CUDA -s=True -v=True
    done
    
done


