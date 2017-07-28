#!/bin/bash
#numshards=${1:-8}

CNNmodel=model3
freqor=2D
pyprog=`which ./esc50_us8K_classification.py`

n_epochs=1
batchsize=4

l1channels=4
l2channels=8
l3channels=16
fcsize=24

numLabels=2 #no.of classes
filesPerFold=2 #no. of samples per fold


python $pyprog pfile --fold=1 --freqorientation=$freqor --model=$CNNmodel batchsize=$batchsize n_epochs=$n_epochs l1channels=$l1channels l2channels=$l2channels l3channels=$l3channels fcsize=$fcsize filesPerFold=$filesPerFold numLabels=$numLabels

