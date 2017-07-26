#!/bin/bash
#numshards=${1:-8}
echo "Choose the model: [model1, model3]"
read CNNmodel
echo "Choose [1D or 2D] convolution. If 1D, freq bins treated as channels, if 2D freq bins is the height of input"
read freqor
echo "Will now run classification on ESC50 with k-fold cross validation..."
pyprog=`which ./esc50_us8K_classification.py`

python $pyprog --fold=1 --freqorientation=$freqor --model=$CNNmodel
python $pyprog --fold=2 --freqorientation=$freqor --model=$CNNmodel
python $pyprog --fold=3 --freqorientation=$freqor --model=$CNNmodel
python $pyprog --fold=4 --freqorientation=$freqor --model=$CNNmodel
python $pyprog --fold=5 --freqorientation=$freqor --model=$CNNmodel

