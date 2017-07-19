### General description
This repo is used for ESC50 and UrbanSound8K classification using a convolutional neural net with Tensorflow. Written for python 3.5. Developed on unix but should work on windows with a few tweaks. **Note currently only ESC50 supported and fully tested.**
Supported transformations:
* Linear/Mel scaled Short-time fourier transform (STFT)
* Constant-Q transform (CQT)
* Continuous Wavelet transform (CWT)
* MFCC

Required libraries:
* For data preperation
  * Jupyter Notebook
  * pillow
  * librosa
  * pyWavelets (only for CWT)
  * scipy
* For classification  
  * Tensorflow (>ver 1.0)

### Data Preparation
As input we will first generate time-frequency representations of the audio signal (i.e spectrograms) from the .ogg or .wav files then convert them to TFRecords, Tensorflow's native data representation.
1. Go into DataPrep folder and launch and run WavToSpecConversion.ipynb
2. A small subset of the ESC50 dataset has been included in toy_data to get you started. Full datasets can be found at [ESC50](https://github.com/karoldvl/ESC-50) and [UrbanSound8K](https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html).
3. If all cells were run correctly, you will now have additional folders in DataPrep. {transform}_png and {transform}_tif holds the png and tif linear spectrograms respectively where {transform} is the name of the transformation chosen. Under these folders, spectrograms have been split into their respective folds for cross-validation during training. A labels.txt file containing class labels should also have been generated.
4. Next run spect2TFRecords.sh to convert the images into TFRecords. Specify the folder containing the folds ie.{transform}_png and the number of shards per fold desired. Only point to the png folder since tif is not supported. TFRecords should now be generated in {transform}_png.

### Classification
!!To be written!!
