#
#
#Morgans great example code:
#https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
#
# GitHub utility for freezing graphs:
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py
#
#https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants


import tensorflow as tf
import numpy as np

from PIL import TiffImagePlugin, ImageOps
from PIL import Image


import pickle

g_graph=None

#k_freqbins=257
#k_width=856

VERBOSE=0

#------------------------------------------------------------

#global
# gleaned from the parmeters in the pickle file; used to load images
height=0
width=0
depth=0

#-------------------------------------------------------------

def getShape(g, name) :
	return g.get_tensor_by_name(name + ":0").get_shape()

def loadImage(fname) :
	#transform into 1D width with frequbins in channel dimension (we do this in the graph in the training net, but not with this reconstructed net)
	if (height==1) : 
		return np.transpose(np.reshape(np.array(Image.open(fname).point(lambda i: i*255)), [1,depth,width,1]), [0,3,2,1]) 
	else :
		return np.reshape(np.array(Image.open(fname).point(lambda i: i*255)), [1,height,width,1])


def generate_noise_image(content_image, noise_ratio=0.6):
	print('generate_noise_image with height=' + str(height) + ', width =' + str(width) + ', and depth =' + str(depth))
	noise_image = np.random.uniform(-1, 1, (1, height, width, depth)).astype(np.float32)
	print('noise_image shape is ' + str(noise_image.shape))
	return noise_image * noise_ratio + content_image * (1. - noise_ratio)

# Assumes caller puts image into the correct orientation
def save_image(image, fname, scaleinfo=None):
	print('save_image: shape is ' + str(image.shape))
	if (height==1) : # orientation is freq bins in channels
		print('saving image in channel orientation')
		image = np.transpose(image, [2,1,0])[:,:,0]
	else :
		print('saving image in image orientation')
		image = image[:,:,0]
	
	print('AFTER reshaping, save_image: shape is ' + str(image.shape))


	
	print('image max is ' + str(np.amax(image) ))
	print('image min is ' + str(np.amin(image) ))
	# Output should add back the mean pixels we subtracted at the beginning

	# [0,80db] -> [0, 255]
	# after style transfer, images range outside of [0,255].
	# To preserve scale, and mask low values, we shift by (255-max), then clip at 0 and then have all bins in the top 80dB.
	image = np.clip(image-np.amax(image)+255, 0, 255).astype('uint8')

	info = TiffImagePlugin.ImageFileDirectory()
    
	if (scaleinfo == None) :
	    info[270] = '80, 0'
	else :
	    info[270] = scaleinfo

	#scipy.misc.imsave(path, image)

	bwarray=np.asarray(image)/255.

	savimg = Image.fromarray(np.float64(bwarray)) #==============================
	savimg.save(fname, tiffinfo=info)
	#print('RGB2TiffGray : tiffinfo is ' + str(info))
	return info[270] # just in case you want it for some reason

##===========================================================================================================
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k_h=2, k_w=2):
    # MaxPool2D wrapper
    # ksize = [batch, height, width, channels]
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, k_h, k_w, 1], padding='SAME')

# This can produce any model that model1 and model3 can produce (1D or 2D, 1 layer or 2 layer)
# Rather than keeping weights, biases, and other model ops separate, and producing the logits layer as a return value,
#     it returns all variables and ops as a graph object
def constructSTModel(weights, biases, params) : #nlaysers is either 1 or 3

	print("now construct graph ")
	global g_graph
	g_graph = {} 

    #k_height = params['k_height']
    #k_inputChannels = params['k_inputChannels']
    #k_ConvRows = params['K_ConvRows'] #conv kernel height
    #k_ConvCols = params['K_ConvCols'] #conv kernel width
    #k_poolRows = params['k_poolRows']
    #k_downsampledHeight = params['k_downsampledHeight']
    #k_downsampledWidth = params['k_downsampledHeight']

	# model params common to both 1D and 2D
	#K_NUMCONVLAYERS = params['K_NUMCONVLAYERS']
	#k_ConvStrideRows = params['k_ConvStrideRows'] #kernel horizontal stride
	#k_ConvStrideCols = params['k_ConvStrideCols'] #kernel vertical stride
	#k_poolStrideRows = params['k_poolStrideRows']


	#Huz - is this right??
	if params['K_NUMCONVLAYERS'] == 3 :
		k_poolCols=2
	else :
		k_poolCols=4


	g_graph["X"] = tf.Variable(np.zeros([1,params['k_height'], params['k_numFrames'], params['k_inputChannels']]), dtype=tf.float32, name="s_X")
	
	g_graph["w1"]=tf.constant(weights["wc1"], name="s_w1")
	g_graph["b1"]=tf.constant(biases["bc1"], name="s_b1")
	g_graph["h1"]=conv2d(g_graph["X"], g_graph["w1"], g_graph["b1"])
	g_graph["h1pooled"] = maxpool2d(g_graph["h1"], k_h=params['k_poolRows'], k_w=k_poolCols)


	g_graph["W_fc1"] = tf.constant(weights['wd1'], name="s_W_fc1")
	g_graph["b_fc1"] = tf.constant(biases["bd1"], name="s_b_fc1")

	if params['K_NUMCONVLAYERS']== 3: 

		g_graph["w2"]=tf.constant(weights["wc2"], name="s_w2")
		g_graph["b2"]=tf.constant(biases["bc2"], name="s_b2")
		g_graph["h2"]=conv2d(g_graph["h1pooled"], g_graph["w2"], g_graph["b2"])
		g_graph["h2pooled"] = maxpool2d(g_graph["h2"], k_h=params['k_poolRows'], k_w=k_poolCols)

		g_graph["w3"]=tf.constant(weights["wc3"], name="s_w3")
		g_graph["b3"]=tf.constant(biases["bc3"], name="s_b3")
		g_graph["h3"]=conv2d(g_graph["h2pooled"], g_graph["w3"], g_graph["b3"])

		g_graph["fc1"] = tf.reshape(g_graph["h3"], [-1,  g_graph["W_fc1"].get_shape().as_list()[0]])  #convlayers_output

	else :
		g_graph["fc1"] = tf.reshape(g_graph["h1pooled"], [-1, g_graph["W_fc1"].get_shape().as_list()[0]]) #convlayers_output
	g_graph["h_fc1"] = tf.nn.relu(tf.matmul(g_graph["fc1"], g_graph["W_fc1"]) + g_graph["b_fc1"], name="s_h_fc1")


	g_graph["W_fc2"] = tf.constant(weights['wout'], name="s_W_fc2")
	g_graph["b_fc2"] = tf.constant(biases['bout'], name="s_b_fc2")


	g_graph["logits"] = tf.add(tf.matmul(g_graph["h_fc1"], g_graph["W_fc2"]) , g_graph["b_fc2"] , name="s_logits")  #"out"

	g_graph["softmax_preds"] = tf.nn.softmax(logits=g_graph["logits"], name="s_softmax_preds")

	print("graph built - returning ")
	return g_graph

#=============================================================================================================
def constructSTModel_old(weights, biases, params) :
	global g_graph
	g_graph = {} 


	#This is the variable that we will "train" to match style and content images.
	##g_graph["X"] = tf.Variable(np.zeros([1,k_width*k_freqbins]), dtype=tf.float32, name="s_x_image")
	##g_graph["x_image"] = tf.reshape(g_graph["X"], [1,k_height,k_width,k_inputChannels])

	g_graph["X"] = tf.Variable(np.zeros([1,params['k_height'], params['k_width'], params['k_inputChannels']]), dtype=tf.float32, name="s_X")
	
	g_graph["w1"]=tf.constant(state["w1:0"], name="s_w1")
	g_graph["b1"]=tf.constant(state["b1:0"], name="s_b1")
	#g_graph["w1"]=tf.Variable(tf.truncated_normal(getShape( tg, "w1"), stddev=0.1), name="w1")
	#g_graph["b1"]=tf.Variable(tf.constant(0.1, shape=getShape( tg, "b1")), name="b1")
	
	#             tf.nn.relu(tf.nn.conv2d(x_image,            w1,            strides=[1, k_ConvStrideRows, k_ConvStrideCols, 1], padding='SAME') + b1,            name="h1")
	g_graph["h1"]=tf.nn.relu(tf.nn.conv2d(g_graph["X"], g_graph["w1"], strides=[1, params['k_ConvStrideRows'], params['k_ConvStrideCols'], 1], padding='SAME') + g_graph["b1"], name="s_h1")
	# 2x2 max pooling
	g_graph["h1pooled"] = tf.nn.max_pool(g_graph["h1"], ksize=[1, params['k_poolRows'], 2, 1], strides=[1, params['k_poolStride'], 2, 1], padding='SAME', name="s_h1_pooled")

	g_graph["w2"]=tf.constant(state["w2:0"], name="s_w2")
	g_graph["b2"]=tf.constant(state["b2:0"], name="s_b2")
	#g_graph["w2"]=tf.Variable(tf.truncated_normal(getShape( tg, "w2"), stddev=0.1), name="w2")
	#g_graph["b2"]=tf.Variable(tf.constant(0.1, shape=getShape( tg, "b2")), name="b2")

	g_graph["h2"]=tf.nn.relu(tf.nn.conv2d(g_graph["h1pooled"], g_graph["w2"], strides=[1, params['k_ConvStrideRows'], params['k_ConvStrideCols'], 1], padding='SAME') + g_graph["b2"], name="s_h2")

	g_graph["h2pooled"] = tf.nn.max_pool(g_graph["h2"], ksize=[1, params['k_poolRows'], 2, 1], strides=[1, params['k_poolStride'], 2, 1], padding='SAME', name='s_h2_pooled')
	g_graph["convlayers_output"] = tf.reshape(g_graph["h2pooled"], [-1, params['k_downsampledWidth'] * params['k_downsampledHeight']*params['L2_CHANNELS']]) # to prepare it for multiplication by W_fc1

#
	g_graph["W_fc1"] = tf.constant(state["W_fc1:0"], name="s_W_fc1")
	g_graph["b_fc1"] = tf.constant(state["b_fc1:0"], name="s_b_fc1")

	#g_graph["keepProb"]=tf.placeholder(tf.float32, (), name= "keepProb")
	#g_graph["h_fc1"] = tf.nn.relu(tf.matmul(tf.nn.dropout(g_graph["convlayers_output"], g_graph["keepProb"]), g_graph["W_fc1"]) + g_graph["b_fc1"], name="h_fc1")
	g_graph["h_fc1"] = tf.nn.relu(tf.matmul(g_graph["convlayers_output"], g_graph["W_fc1"]) + g_graph["b_fc1"], name="s_h_fc1")


	#Read out layer
	g_graph["W_fc2"] = tf.constant(state["W_fc2:0"], name="s_W_fc2")
	g_graph["b_fc2"] = tf.constant(state["b_fc2:0"], name="s_b_fc2")


	g_graph["logits_"] = tf.matmul(g_graph["h_fc1"], g_graph["W_fc2"])
	g_graph["logits"] = tf.add(g_graph["logits_"] , g_graph["b_fc2"] , name="s_logits")


	g_graph["softmax_preds"] = tf.nn.softmax(logits=g_graph["logits"], name="s_softmax_preds")


	return g_graph

# Create and save the picke file of paramters 
def saveState(sess, weight_dic, bias_dic, parameters, fname) :
	# create object to stash tensorflow variables in
	state={}
	#for v in vlist :
	#	state[v.name] = sess.run(v)

	# convert tensors to python arrays
	for n in weight_dic.keys():
		weight_dic[n] = sess.run(weight_dic[n])

	for b in bias_dic.keys():
		bias_dic[b] = sess.run(bias_dic[b])

	# combine state and parameters into a single object for serialization
	netObject={
		#'state' : state,
		'weight_dic' : weight_dic,
		'bias_dic' : bias_dic, 
		'parameters' : parameters
	}
	pickle.dump(netObject, open( fname, "wb" ))


# Load the pickle file of parameters
def load(pickleFile, randomize=0) :
	print(' will read state from ' + pickleFile)
	netObject=pickle.load( open( pickleFile, "rb" ) )
	#state = netObject['state']
	weight_dic = netObject['weight_dic']
	bias_dic = netObject['bias_dic']
	parameters = netObject['parameters']

	if randomize ==1 :
		print('randomizing weights')
		for n in weight_dic.keys():
			print('shape of weights[' + n + '] is ' + str(weight_dic[n].shape))
			weight_dic[n] = .2* np.random.random_sample(weight_dic[n].shape).astype(np.float32) -.1

		print('randomizing biases')
		for n in bias_dic.keys():
			print('shape of biases[' + n + '] is ' + str(bias_dic[n].shape))
			bias_dic[n] = .2* np.random.random_sample(bias_dic[n].shape).astype(np.float32) -.1


	print("weight keys are " + str(weight_dic.keys()))
	print("bias keys are " + str(bias_dic.keys()))


	for p in parameters.keys() :
		print('param['  + p + '] = ' + str(parameters[p]))


	global height
	height = parameters['k_height']

	global width 
	#width = parameters['k_width']
	width = parameters['k_numFrames']

	global depth
	depth = parameters['k_inputChannels']

	return constructSTModel(weight_dic, bias_dic, parameters)

