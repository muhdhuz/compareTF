# Data Location
dataset_name = "ESC50" #supports ESC50 and US8K

STFT_dataset_path = "../DataPrep/stft"

save_path = "../Results" #path to save output

# Image Parameters
IMAGE_WIDTH= 214  #pixel width ie. time bins
IMAGE_HEIGHT = 513 #pixel height ie. frequency bins
NUM_CHANNELS = 1 #no of image channels
N_LABELS = 2 #no.of classes

# Learning Parameters
batch_size = 1
epochs = 1
TOTAL_RUNS = 1 #no. of rounds of k-fold cross validation done

# Network Parameters
epsilon = 1e-08 #epsilon value for Adam optimizer
dropout = .5 # Dropout, probability to keep units
l2reg = True #if want l2 regularization 
l2regfull = False #if want l2 regularization only on dense layers, else l2 regularization on all weight layers
beta = 0.001 # L2-regularization


#Tensorboard and Checkpoint Parameters
display_step = 1 #2 # How often we want to write the tf.summary data to disk. each step denotes 1 mini-batch
checkpoint_epoch = 1 # 10 #checkpoint and save model every checkpoint_epoch 

# Train/Test holdout split Parameters
# can ignore if not using holdout
hold_prop = 0.4 #proportion of data used for testing
rand_seed = 14

#"C:/Users/Huz/Documents/python_scripts/comparing_TF_representations/compare_TF_rep/DataPrep/stft"
#"C:/Users/Huz/Documents/python_scripts/Comparing_TF_representations/ESC50/data/1/stft"