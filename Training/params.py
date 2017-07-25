# Data Location
dataset_name = "ESC50" #supports ESC50 and US8K

STFT_dataset_path = "../DataPrep/stft_png"

#STFT_dataset_path = "C:/Users/Huz/Documents/python_scripts/comparing_TF_representations/compare_TF_rep/DataPrep/stft"
#CQT_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/US8K/data/2/cqt"
#MEL_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/US8K/data/2/mel"
#WAVE_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/wavelet"
#MFCC_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/mfcc"

INDIR = STFT_dataset_path
save_path = "../Results" #path to save output


# Image/Data Parameters
K_NUMFRAMES = 214  #pixel width ie. time bins
K_FREQBINS = 513 #pixel height ie. frequency bins
NUM_CHANNELS = 1 #no of image channels
N_LABELS = 50 #no.of classes
FRE_ORIENTATION = "2D" #supports 2D and 1D

NUM_THREADS = 4 #threads to read in TFRecords
files_per_fold = 400 #no. of samples per fold


# Model Parameters
L1_CHANNELS = 20 #180
FC_SIZE = 100 # 800


# Learning Parameters
BATCH_SIZE = 20
EPOCHS = 5
TOTAL_RUNS = 1 #no. of rounds of k-fold cross validation done

test_batches_per_epoch = int(files_per_fold/BATCH_SIZE)
train_batches_per_epoch = int(files_per_fold*4/BATCH_SIZE) #equivalent to steps per epoch
testNSteps = train_batches_per_epoch # test every n steps


# Network Parameters
epsilon = 1e-08 #epsilon value for Adam optimizer
dropout = .5 # Dropout, probability to keep units
l2reg = True #if want l2 regularization 
l2regfull = False #if want l2 regularization only on dense layers, else l2 regularization on all weight layers
beta = 0.001 # L2-regularization


#Tensorboard and Checkpoint Parameters
display_step = 2 # How often we want to write the tf.summary data to disk. each step denotes 1 mini-batch
checkpoint_epoch = 20 #checkpoint and save model every checkpoint_epoch 


# Train/Test holdout split Parameters
# can ignore if not using holdout
hold_prop = 0.4 #proportion of data used for testing
rand_seed = 14

#"C:/Users/Huz/Documents/python_scripts/comparing_TF_representations/compare_TF_rep/DataPrep/stft"
#"C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/stft"