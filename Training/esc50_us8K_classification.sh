#!/bin/bash
#numshards=${1:-8}
echo "Enter the directory that hold all the folds"
read main_dir
echo "Enter number of training shards for each fold"
read numshards
echo "will run spect2TFRecords with " $numshards " shards"
pyprog=`which ./spect2TFRecords.py`
echo "will execute " $pyprog
python $pyprog --main_dir=$main_dir --labels_file=labels.txt --train_shards=$numshards --num_threads=2
#--fold1_directory=./1 --fold2_directory=./2 --fold3_directory=./3 --fold4_directory=./4 --fold5_directory=./5 --output_directory=./
