#!/bin/bash

# cases:
# small -
# 1. linear 2D model1
# 2. linear 1D model1
# 3. cqt 2D model1
# 4. cqt 1D model1
# 5. linear 2D model3
# 6. linear 1D model3
# 7. cqt 2D model3
# 8. cqt 1D model3


pyprog=`which ./esc50_us8K_classification.py`
lin_datafolder=stft_small_png
cqt_datafolder=constQ_small_png

n_epochs=200
batchsize=100

numLabels=50 #no.of classes
filesPerFold=400 #no. of samples per fold

freqbins=103 
numFrames=43

# model1
l1channels=180
l2channels=0
l3channels=0
fcsize=800

# model3
m3_l1channels=24
m3_l2channels=48
m3_l3channels=96

python $pyprog --fold=1 --freqorientation=2D --model=model1 --datafolder=$lin_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallLinear2Dmodel1'

python $pyprog --fold=1 --freqorientation=1D --model=model1 --datafolder=$lin_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallLinear1Dmodel1'

python $pyprog --fold=1 --freqorientation=2D --model=model1 --datafolder=$cqt_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallCQT2Dmodel1'

python $pyprog --fold=1 --freqorientation=1D --model=model1 --datafolder=$cqt_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$l1channels --l2channels=$l2channels --l3channels=$l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallCQT1Dmodel1'

python $pyprog --fold=1 --freqorientation=2D --model=model3 --datafolder=$lin_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$m3_l1channels --l2channels=$m3_l2channels --l3channels=$m3_l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallLinear2Dmodel3'

python $pyprog --fold=1 --freqorientation=1D --model=model3 --datafolder=$lin_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$m3_l1channels --l2channels=$m3_l2channels --l3channels=$m3_l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallLinear1Dmodel3'

python $pyprog --fold=1 --freqorientation=2D --model=model3 --datafolder=$cqt_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$m3_l1channels --l2channels=$m3_l2channels --l3channels=$m3_l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallCQT2Dmodel3'

python $pyprog --fold=1 --freqorientation=1D --model=model3 --datafolder=$cqt_datafolder --freqbins=$freqbins --numFrames=$numFrames --batchsize=$batchsize --n_epochs=$n_epochs --l1channels=$m3_l1channels --l2channels=$m3_l2channels --l3channels=$m3_l3channels --fcsize=$fcsize --filesPerFold=$filesPerFold --numLabels=$numLabels --save_path='../Results/smallCQT1Dmodel3'

