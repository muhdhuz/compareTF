#!/bin/bash

echo "Enter the TFRecords folder basename"
read folder
echo "Choose the model: [model1, model3]"
read CNNmodel
echo "Choose [1D or 2D] convolution. If 1D, freq bins treated as channels, if 2D freq bins is the height of input"
read freqor
echo "Will now run classification on ESC50 with k-fold cross validation..."
pyprog=`which ./esc50_us8K_classification.py`

#CNNmodel=model3
#freqorientation=2D
#datafolder=stft_png

n_epochs=10
batchsize=100

l1channels=180
l2channels=8
l3channels=16
fcsize=800

numLabels=50 #no.of classes
filesPerFold=400 #no. of samples per fold

freqbins=103 
numFrames=43

python $pyprog --datafolder=$folder --fold=1 --freqorientation=$freqor --model=$CNNmodel --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels
python $pyprog --datafolder=$folder --fold=2 --freqorientation=$freqor --model=$CNNmodel --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels
python $pyprog --datafolder=$folder --fold=3 --freqorientation=$freqor --model=$CNNmodel --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels
python $pyprog --datafolder=$folder --fold=4 --freqorientation=$freqor --model=$CNNmodel --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels
python $pyprog --datafolder=$folder --fold=5 --freqorientation=$freqor --model=$CNNmodel --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels
