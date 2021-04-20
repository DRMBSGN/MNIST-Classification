#!/bin/sh
CUDA=0
EPOCH=50


python main.py --mode=graph_compare -e=50 --model=ReLet5 -z=True -s=False
python main.py --mode=graph_compare -e=50 --model=CustomMLP_6 -z=True -s=False
python main.py --mode=graph_compare -e=50 --model=all -z=True -s=False


