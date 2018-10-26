# vgg2knn
this is an example of automatic real-time facial classification from the webcam through the use of the neural network VGG-face 2 Resnet-50 (https://github.com/ox-vgg/vgg_face2).  The example extracts face features from a simple repository of five famous celebrities. From the frame of the webcam, the faces are detected and the features are extracted and they are matched using k nearest neighbor algorithm. If you exceed a certain confidence, the face is classified otherwise it is given for unknown.

The main matlab script is facerecognition.m
