# Data Location
dataset_name = "ESC50" #supports ESC50 and US8K

STFT_dataset_path = "../DataPrep/stft_png"

#STFT_dataset_path = "C:/Users/Huz/Documents/python_scripts/comparing_TF_representations/compare_TF_rep/DataPrep/stft"
#CQT_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/US8K/data/2/cqt"
#MEL_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/US8K/data/2/mel"
#WAVE_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/wavelet"
#MFCC_dataset_path = "C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/mfcc"

INDIR=STFT_dataset_path

save_path = "../Results" #path to save output


k_freqbins = 513
k_numFrames = 214

# Image Parameters
IMAGE_WIDTH= k_numFrames  #pixel width ie. time bins
IMAGE_HEIGHT = k_freqbins #pixel height ie. frequency bins
NUM_CHANNELS = 1 #no of image channels
N_LABELS = 2 #no.of classes

files_per_fold = 4

##################################################
# model params
L1_CHANNELS = 4 #180
FC_SIZE = 100 # 800

##################################################

# Learning Parameters
batch_size = 2
epochs = 2
TOTAL_RUNS = 1 #no. of rounds of k-fold cross validation done

testNSteps = 2 # test every n steps
test_batches_per_epoch = int(files_per_fold/batch_size)

# Network Parameters
epsilon = 1e-08 #epsilon value for Adam optimizer
dropout = .5 # Dropout, probability to keep units
l2reg = True #if want l2 regularization 
l2regfull = False #if want l2 regularization only on dense layers, else l2 regularization on all weight layers
beta = 0.001 # L2-regularization


#Tensorboard and Checkpoint Parameters
display_step = 2 # How often we want to write the tf.summary data to disk. each step denotes 1 mini-batch
checkpoint_epoch = 2 #checkpoint and save model every checkpoint_epoch 

# Train/Test holdout split Parameters
# can ignore if not using holdout
hold_prop = 0.4 #proportion of data used for testing
rand_seed = 14

#"C:/Users/Huz/Documents/python_scripts/comparing_TF_representations/compare_TF_rep/DataPrep/stft"
#"C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/stft"