### General description
This repo is used for ESC50 and UrbanSound8K classification using a convolutional neural net with Tensorflow. Developed on unix but should work on windows with a few tweaks. **Note currently only ESC50 supported and fully tested.**

Supported transformations:
* Linear/Mel scaled Short-time fourier transform (STFT)
* Constant-Q transform (CQT)
* Continuous Wavelet transform (CWT)
* MFCC

### Data Preparation
Required libraries:
* Python 3.5
* Jupyter Notebook
* pillow
* [librosa 0.5.1](https://librosa.github.io/librosa/install.html)
* pyWavelets (only for CWT)
* scipy
* Tensorflow (ver >1.0) for TFRecords

As input we will first generate time-frequency representations of the audio signal (i.e. spectrograms) from the .ogg or .wav files then convert them to TFRecords, Tensorflow's native data representation.
1. Go into DataPrep folder, launch and run WavToSpecConversion.ipynb
2. A small subset of the ESC50 dataset has been included in toy_data to get you started. Full datasets can be found at [ESC50](https://github.com/karoldvl/ESC-50) and [UrbanSound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html).
3. If all cells were run correctly, you will now have additional folders in DataPrep. {transform}_png and {transform}_tif holds the png and tif linear spectrograms respectively where {transform} is the name of the transformation chosen. Under these folders, spectrograms have been split into their respective folds for cross-validation during training. A labels.txt file containing class labels should also have been generated.
4. Next run spect2TFRecords.sh and follow the prompts to convert the images into TFRecords. When asked, specify the folder containing the folds i.e.{transform}_png and the number of shards per fold desired. Only point to the png folder since tif is not supported. TFRecords should now be generated in {transform}_png.

### Classification
Required libraries 
* Tensorflow (ver >1.0)

1. Go to the Training folder. Inside there should be a esc50_us8K_classification.py that allows one to do classification on the ESC50 dataset with a single fold held out as a validation set.
2. params.py holds some important parameters including data paths, model and network hyperparameters. You can change the hyperparameters to suit your experiment in this file.
3. In the same params file you can also specify the number of neurons for each layer in the model. Two models are provided - model1 (1 conv layer + 2 fc) and model3 (3 conv layer + 2 fc). If you want to make more drastic changes to the model do it in the individual model files.
4. When ready, decide whether you want to run validation on a single fold or do k-fold cross validation. For single fold, run esc50_us8K_classification.py. You can specify the folder containing the TFRecords, validation fold, whether to do 1D (freq bins as channels) or 2D (freq bins as height of input) convolution and the model to use in the arguments. 
5. A bash script kfold_classification.sh has been provided for your convenience to do k-fold. Run it without any arguments and follow the prompts.

TO DO: saveState() in pickledModel.py under utils to take in dictionaries. Also double check compatibility of parameters list to the variables used for style transfer.


