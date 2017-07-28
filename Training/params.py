import argparse
import os

# pass some user input as flags
FLAGS = None
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datafolder', type=str, help='basename of folder where TFRecords are kept', default='stft_small_png')
parser.add_argument('--fold', type=int, help='fold used as test set for k-fold cross validation', default=1)
parser.add_argument('--freqorientation', type=str, help='convolution over 1D or 2D. If 1D freq bins treated as channels, if 2D freq bins is the height of input', default='2D')
parser.add_argument('--model', type=str, help='load the model to train', default='model1') 

parser.add_argument('--batchsize', type=int, help='number of data records per training batch', default=20) #default for testing
parser.add_argument('--n_epochs', type=int, help='number of epochs to use for training', default=8) #default for testing

parser.add_argument('--l1channels', type=int, help='Number of channels in the first convolutional layer', default=24) #default for testing
parser.add_argument('--l2channels', type=int, help='Number of channels in the second convolutional layer (ignored if numconvlayers is 1)', default=48) #default for testing
parser.add_argument('--l3channels', type=int, help='Number of channels in the second convolutional layer (ignored if numconvlayers is 1)', default=96) #default for testing
parser.add_argument('--fcsize', type=int, help='Dimension of the final fully-connected layer', default=400) #default for testing

parser.add_argument('--numLabels', type=int, help='number of classes in data', choices=[2,50], default=50) 
parser.add_argument('--filesPerFold', type=int, help='number of classes in data', choices=[2,400], default=400) #default for testing


parser.add_argument('--save_path', type=str, help='output root directory for logging',  default='../Results') 

FLAGS, unparsed = parser.parse_known_args(CMD_LINE)
print('\n FLAGS parsed :  {0}'.format(FLAGS))

#*****************************************************************
# Data Location
dataset_name = "ESC50" #supports ESC50 and US8K
TRAINING_FOLDS = 4

STFT_dataset_path = "../DataPrep/" + FLAGS.datafolder

#STFT_dataset_path = "C:/Users/Huz/Documents/python_scripts/comparing_TF_representations/compare_TF_rep/DataPrep/stft"
#CQT_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/US8K/data/2/cqt"
#MEL_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/US8K/data/2/mel"
#WAVE_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/wavelet"
#MFCC_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/mfcc"

INDIR = STFT_dataset_path

save_path = FLAGS.save_path #path to save output
if not os.path.isdir(save_path): os.mkdir(save_path)

text_file_name = "/test_small1D" #name of textfile to output with results


# Image/Data Parameters
K_NUMFRAMES = 43  #pixel width ie. time bins
K_FREQBINS = 103 #pixel height ie. frequency bins
NUM_CHANNELS = 1 #no of image channels
N_LABELS = FLAGS.numLabels #no.of classes

FRE_ORIENTATION = FLAGS.freqorientation #supports 2D and 1D
if FRE_ORIENTATION in ["2D","1D"]:
    pass
else:
    raise ValueError("please only enter '1D' or '2D'")

#see threading and queueing info: https://www.tensorflow.org/programmers_guide/reading_data
files_per_fold = FLAGS.filesPerFold #no. of samples per fold
NUM_THREADS = 4 #threads to read in TFRecords; dont want more threads than there are 



# Model Parameters

L1_CHANNELS = FLAGS.l1channels
L2_CHANNELS = FLAGS.l2channels
L3_CHANNELS = FLAGS.l3channels
FC_SIZE = FLAGS.fcsize


# Learning Parameters
BATCH_SIZE = FLAGS.batchsize
EPOCHS = FLAGS.n_epochs

TOTAL_RUNS = 1 #no. of rounds of k-fold cross validation done

test_batches_per_epoch = max(1, int(files_per_fold/BATCH_SIZE)) #include check for batch_size > files_per_fold
train_batches_per_epoch = max(1, int(files_per_fold*TRAINING_FOLDS/BATCH_SIZE)) #equivalent to steps per epoch
testNSteps = train_batches_per_epoch # test every n steps
print("Batch_size = " + str(BATCH_SIZE) + ", and files_per_fold is " + str(files_per_fold))
print("Will test every " + str(testNSteps) + " batches.")


# Network Parameters
epsilon = 1e-08 #epsilon value for Adam optimizer
dropout = .5 # Dropout, probability to keep units
l2reg = True #if want l2 regularization 
l2regfull = False #if want l2 regularization only on dense layers, else l2 regularization on all weight layers
beta = 0.001 # L2-regularization


#Tensorboard and Checkpoint Parameters
display_step = 4 # How often we want to write the tf.summary data to disk. each step denotes 1 mini-batch
checkpoint_epoch = 250 #checkpoint and save model every checkpoint_epoch 


# Train/Test holdout split Parameters
# can ignore if not using holdout
hold_prop = 0.4 #proportion of data used for testing
rand_seed = 14