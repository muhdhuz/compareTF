import numpy as np
import os
import re
import tensorflow as tf
import time
from datetime import datetime
import matplotlib.pyplot as plt
import importlib

from params import *

import utils.pickledModel as pickledModel
import utils.spectreader as spectreader
m = importlib.import_module(FLAGS.model) #import CNN model


#some utility functions
#*************************************
def time_taken(elapsed):
    """To format time taken in hh:mm:ss. Use with time.monotic()"""
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


# Create list of paramters for serializing so that network can be properly reconstructed, and for documentation purposes
##*************************************
parameters={
    'k_height' : m.k_height, 
    'k_numFrames' : K_NUMFRAMES, 
    'k_inputChannels' : m.k_inputChannels, 
    'K_NUMCONVLAYERS' : m.K_NUMCONVLAYERS, 
    'L1_CHANNELS' : L1_CHANNELS, 
    'L2_CHANNELS' : L2_CHANNELS,
    'L3_CHANNELS' : L3_CHANNELS, 
    'FC_SIZE' : FC_SIZE, 
    'K_ConvRows' : m.k_ConvRows, 
    'K_ConvCols' : m.k_ConvCols, 
    'k_ConvStrideRows' : m.k_ConvStrideRows, 
    'k_ConvStrideCols' : m.k_ConvStrideCols, 
    'k_poolRows' : m.k_poolRows, 
    'k_poolStrideRows' : m.k_poolStrideRows, 
    'k_downsampledHeight' : m.k_downsampledHeight, 
    'k_downsampledWidth' : m.k_downsampledWidth,
    'freqorientation' : FRE_ORIENTATION,
    'transform' : 
}


# read in the TFRecords
#*************************************
def getImage(fnames, fre_orientation, nepochs=None) :
    """ Reads data from the prepaired *list* of files in fnames of TFRecords, does some preprocessing 
    params:
    fnames - list of filenames to read data from
    fre_orientation - 2D or 1D defined as variable FRE_ORIENTATION
    nepochs - An integer (optional). Just fed to tf.string_input_producer().  Reads through all data num_epochs times before generating an OutOfRange error. None means read forever.
    """
    label, image = spectreader.getImage(fnames, nepochs)
    
    #no need to flatten - must just be explicit about shape so that shuffle_batch will work
    print("reshaping with K_HEIGHT = " + str(K_HEIGHT) + ", K_NUMFRAMES = " + str(K_NUMFRAMES) + ", and NUM_CHANNELS = " + str(NUM_CHANNELS) )
    image = tf.reshape(image,[K_HEIGHT,K_NUMFRAMES,NUM_CHANNELS])

    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, N_LABELS))
    print ("getImage returning")
    return label, image

def get_TFR_folds(a_dir, foldnumlist):
    """ Returns a list of files names in a_dir that start with foldX where X is from the foldnumlist"""
    lis = []
    for num in foldnumlist : 
        lis.extend([a_dir + '/' + name for name in os.listdir(a_dir)
            if name.startswith("fold"+str(num))])
    return lis


foldlist = [1,2,3,4,5]
fold = FLAGS.fold

datanumlist=[x for x in foldlist if x != fold]
validatenumlist=[fold]

datafnames=get_TFR_folds(INDIR, datanumlist)
target, data = getImage(datafnames, FRE_ORIENTATION, nepochs=EPOCHS)

validatefnames=get_TFR_folds(INDIR, validatenumlist)
vtarget, vdata = getImage(validatefnames, FRE_ORIENTATION, nepochs=None) #no need to test on multiple epochs

imageBatch, labelBatch = tf.train.shuffle_batch(
    [data, target], batch_size=BATCH_SIZE,
    num_threads=NUM_THREADS,
    allow_smaller_final_batch=True, #want to finish an epoch even if datasize doesn't divide by batchsize
    enqueue_many=False, #IMPORTANT to get right, False means each tensor represents a single example - https://www.tensorflow.org/api_docs/python/tf/train/shuffle_batch
    capacity=1000,  #1000,
    min_after_dequeue=500) #500

vimageBatch, vlabelBatch = tf.train.batch(
    [vdata, vtarget], batch_size=BATCH_SIZE,
    num_threads=NUM_THREADS,
    allow_smaller_final_batch=True, #want to finish an epoch even if datasize doesn't divide by batchsize
    enqueue_many=False, #IMPORTANT to get right, default=False - 
    capacity=1000)


# create the model
#*************************************
# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = save_path + "/filewriter/"
checkpoint_path = save_path + "/checkpoint/"

# Create parent path if it doesn't exist
if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)


# tf Graph input placeholders
x = tf.placeholder(tf.float32, [None, K_HEIGHT, K_NUMFRAMES, NUM_CHANNELS])
y = tf.placeholder(tf.int32, [None, N_LABELS])
keep_prob = tf.placeholder(tf.float32, (), name="keepProb") #dropout (keep probability)

# Construct model
pred = m.conv_net(x, m.weights, m.biases, keep_prob)

#L2 regularization
lossL2 = tf.add_n([tf.nn.l2_loss(val) for name,val in m.weights.items()]) * beta #L2 reg on all weight layers
lossL2_onlyfull = tf.add_n([tf.nn.l2_loss(m.weights['wd1']),tf.nn.l2_loss(m.weights['wout'])]) * beta #L2 reg on dense layers

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    if l2reg:
        if l2regfull:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + lossL2_onlyfull)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) + lossL2)
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# Train op
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(epsilon=epsilon).minimize(loss)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Predictions
prob = tf.nn.softmax(pred)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = 100*tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()


#create Session and run training/validation
#*************************************
test_acc_list = []

start_time_long = time.monotonic()
text_file = open(save_path + text_file_name + "_fold{}.txt".format(fold), "w") #save training data
print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

text_file.write('*** Initializing fold #%u as test set ***\n' % fold)
print('*** Initializing fold #%u as test set ***' % fold)

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path + str(fold))

def trainModel():
    with tf.Session() as sess:

        # Initialize all variables        
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        # Add the model graph to TensorBoard
        writer.add_graph(sess.graph)

        coord = tf.train.Coordinator()
        enqueue_threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        print("{} Start training...".format(datetime.now()))
        start_time = time.monotonic()

        try:
            if coord.should_stop():
                print("coord should stop")

            e = 1
            step = 1
            print("{} Epoch number: {}".format(datetime.now(), e))

            while True:  # for each minibatch until data runs out after specified number of epochs
                if coord.should_stop():
                    print("data feed done, quitting")
                    break

                #create training mini-batch here
                #print("Run batch number " + str(step))
                batch_data, batch_labels = sess.run([imageBatch, labelBatch])
                #train and backprop
                sess.run(optimizer, feed_dict= {x:batch_data, y:batch_labels, keep_prob:dropout})

                #print("step = " + str(step))

                #run merged_summary to display progress on Tensorboard
                if (step % display_step == 0):               
                    s = sess.run(merged_summary, feed_dict={x: batch_data, y: batch_labels, keep_prob: 1.})
                    ##writer.add_summary(s, e*train_batches_per_epoch + step) 
                    writer.add_summary(s, step)

                if (step % testNSteps == 0):
                    #print("testing ................")
                    test_acc = 0.
                    test_count = 0
                    #print("now test for " + str(test_batches_per_epoch) + " test steps")
                    for j in range(test_batches_per_epoch):
                        #print("test step = " + str(j))
                        try:
                            #prepare test mini-batch
                            test_batch, label_batch = sess.run([vimageBatch, vlabelBatch])
                            acc = sess.run(accuracy, feed_dict={x: test_batch, y: label_batch, keep_prob: 1.})
                            test_acc += acc*BATCH_SIZE
                            test_count += 1*BATCH_SIZE
                        except (Exception) as ex: #triggered if we run out of validation data to feed queue
                            print(ex)

                    #calculate total test accuracy
                    test_acc /= test_count 
                    print("{} Test Accuracy = {:.4f}".format(datetime.now(),test_acc))
                    text_file.write("{} Test Accuracy = {:.4f}\n".format(datetime.now(),test_acc))
                    test_acc_list.append(test_acc)

                if (step % train_batches_per_epoch == 0):
                    e += 1
                    print("{} Epoch number: {}".format(datetime.now(), e))
                        #save checkpoint of the model
                    if (e % checkpoint_epoch == 0):  
                        checkpoint_name = os.path.join(checkpoint_path, dataset_name+'model_fold'+str(fold)+'_epoch'+str(e)+'.ckpt')
                        saver.save(sess, checkpoint_name) 
                        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

                step += 1

        except (tf.errors.OutOfRangeError) as ex:
            coord.request_stop(ex)

        finally :
            coord.request_stop()
            coord.join(enqueue_threads)                                      
        
        # find the max test score and the epoch it belongs to        
        print("Max validation accuracy is",max(test_acc_list)," at epoch",test_acc_list.index(max(test_acc_list))+1)
        text_file.write("Max validation accuracy is {} at epoch {}\n".format(max(test_acc_list),test_acc_list.index(max(test_acc_list))+1))

        elapsed_time = time.monotonic() - start_time
        text_file.write("--- Training time taken: {} ---\n".format(time_taken(elapsed_time)))
        print("--- Training time taken:",time_taken(elapsed_time),"---")
        print("------------------------")
        
        print('now saving meta model...')
        meta_graph_def = tf.train.export_meta_graph(filename=save_path + '/my-model.meta')
        pickledModel.saveState(sess, m.weights, m.biases, parameters, save_path + '/state.pickle') 

        #write hyperparams to file
        text_file.write("------------------------\n")
        # Image/Data Parameters
        text_file.write("*** Image/Data Parameters ***\n")
        text_file.write("K_NUMFRAMES = {}\n".format(K_NUMFRAMES)) #pixel width ie. time bins
        text_file.write("K_FREQBINS = {}\n".format(K_FREQBINS)) #pixel height ie. frequency bins
        text_file.write("K_HEIGHT = {}\n".format(K_HEIGHT)) #pixel height ie. frequency bins        
        text_file.write("NUM_CHANNELS = {}\n".format(NUM_CHANNELS)) #no of image channels
        text_file.write("N_LABELS = {}\n".format(N_LABELS)) #no.of classes
        text_file.write("FRE_ORIENTATION = {}\n".format(FRE_ORIENTATION)) #supports 2D and 1D
        text_file.write("NUM_THREADS = {}\n".format(NUM_THREADS)) #threads to read in TFRecords
        text_file.write("files_per_fold = {}\n".format(files_per_fold)) #no. of samples per fold

        # Model Parameters
        text_file.write("*** Model Parameters ***\n")
        text_file.write("model = {}\n".format(FLAGS.model))
        text_file.write("L1_CHANNELS = {}\n".format(L1_CHANNELS))
        text_file.write("L2_CHANNELS = {}\n".format(L2_CHANNELS))
        text_file.write("L3_CHANNELS = {}\n".format(L3_CHANNELS))
        text_file.write("FC_SIZE = {}\n".format(FC_SIZE))

        # Learning Parameters
        text_file.write("*** Learning Parameters ***\n")
        text_file.write("BATCH_SIZE = {}\n".format(BATCH_SIZE))
        text_file.write("EPOCHS = {}\n".format(EPOCHS))
        text_file.write("TOTAL_RUNS = {}\n".format(TOTAL_RUNS))

        # Network Parameters
        text_file.write("*** Network Parameters ***\n")
        text_file.write("epsilon = {}\n".format(epsilon)) #epsilon value for Adam optimizer
        text_file.write("dropout = {}\n".format(dropout)) # Dropout, probability to keep units
        text_file.write("l2reg = {}\n".format(l2reg)) #if want l2 regularization 
        text_file.write("l2regfull = {}\n".format(l2regfull))
        text_file.write("beta = {}\n".format(beta)) # L2-regularization

        #Tensorboard and Checkpoint Parameters
        text_file.write("*** Tensorboard and Checkpoint Parameters ***\n")
        text_file.write("display_step = {}\n".format(display_step))
        text_file.write("checkpoint_epoch = {}\n".format(checkpoint_epoch))
        
        text_file.write("------------------------\n")
        writer.close()
        #elapsed_time_long = time.monotonic() - start_time_long
        #print("*** All runs completed ***")
        #text_file.write("Total time taken:")
        #text_file.write(time_taken(elapsed_time_long))
        #print("Total time taken:",time_taken(elapsed_time_long))
        text_file.close()
        
        print(' ===============================================================') 

# Do it
trainModel()