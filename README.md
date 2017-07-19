## General description
This repo is used for ESC50 and UrbanSound8K classification using a convolutional neural net with Tensorflow. Written for python 3.5.
As input we will generate time-frequency representations of the audio signal (i.e spectrograms) from the .ogg or .wav files first.
Supported transformations:
* Linear/Mel scaled Short-time fourier transform (STFT)
* Constant-Q transform (CQT)
* Continuous Wavelet transform (CWT)
* MFCC

Required libraries:
For data preperation
* pillow
* librosa
* pyWavelets (only for CWT)
* scipy
For classification
* Tensorflow (>ver 1.0)


