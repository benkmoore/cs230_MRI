import math

import json

import cv

import numpy as np
import numpy.ma as ma

import scipy as scipy

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from mpl_toolkits.axes_grid1 import make_axes_locatable

import itertools

from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

import theano
import theano.tensor as T

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.engine import Layer

from keras.layers import Input, Dense, Convolution1D, Convolution2D, MaxPooling2D, Deconvolution2D, UpSampling2D, Reshape, Flatten, ZeroPadding2D, BatchNormalization, Lambda, Dropout, Activation
from keras.layers import Convolution3D, MaxPooling3D
from keras.models import Model, Sequential
from keras.models import model_from_json

from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing import image

from keras.callbacks import Callback
from keras.models import load_model

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from vis.utils import utils
from vis.visualization import visualize_saliency

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder



#Parameters (Modify as needed)
channels = 166
img_size_x = 256
img_size_y = 256

batch_size = 128
nb_classes = 3
nb_epoch = 25

c = 0

learning_rate = 0.003
early_stopping_patience = 20

class_names = ["CN", "MCI", "AD"]

def load_dataset(dimension = '3d'):
    
    #Still need to implement to use Ben's clean data
    
    return train_data, train_labels, test_data, test_labels, val_data, val_labels


# Modify this to incorporate Mohammad's autoencoder weights before first convolution3D
def build_cnn(dimension = '3d', activation = 'softmax', heatmap = False, w_path = None, compile_model = True):
    input_3d = (1, channels, img_size_x, img_size_y)
    input_2d = (1, img_size_x, img_size_y)
    
    pool_3d = (2, 2, 2)
    pool_2d = (2, 2)
    
    def build_conv_3d():
        model = Sequential()
        
        model.add(Convolution3D(8, 3, 3, 3, name='conv1', input_shape=input_3d))
        model.add(MaxPooling3D(pool_size=pool_3d, name='pool1'))

        model.add(Convolution3D(8, 3, 3, 3, name='conv2'))
        model.add(MaxPooling3D(pool_size=pool_3d, name='pool2'))

        model.add(Convolution3D(8, 3, 3, 3, name='conv3'))
        model.add(MaxPooling3D(pool_size=pool_3d, name='pool3'))
        
        return model
        
    def build_conv_2d():
        model = Sequential()
        
        model.add(Convolution2D(8, 3, 3, name='conv1', input_shape=input_2d))
        model.add(MaxPooling2D(pool_size=pool_2d, name='pool1'))

        model.add(Convolution2D(8, 3, 3, name='conv2'))
        model.add(MaxPooling2D(pool_size=pool_2d, name='pool2'))

        model.add(Convolution2D(8, 3, 3, name='conv3'))
        model.add(MaxPooling2D(pool_size=pool_2d, name='pool3'))
        
        return model
    
    if(dimension == '3d'):
        model = build_conv_3d()
    else:
        model = build_conv_2d()
        
    model.add(Flatten())
    model.add(Dense(2000, activation='relu', name='dense1'))
    model.add(Dropout(0.5, name='dropout1'))

    model.add(Dense(500, activation='relu', name='dense2'))
    model.add(Dropout(0.5, name='dropout2'))

    model.add(Dense(nb_classes, activation=activation, name='softmax'))

    if w_path:
        model.load_weights(w_path)

    opt = keras.optimizers.Adadelta(clipnorm=1.)
    
    if(compile_model):
        model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])
    
    print 'Done building model.'

    return model

# Tracks model learning rate
class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print str('\nLR: {:.6f}\n').format(float(lr))

# Fitting model architecture to data, also runs loss/accuracy tracker
def fit_model(model, v, train_data, train_labels, val_data, val_labels):
    model_weights_file = 'img_classifier_weights_%s.h5' %v
    epoch_weights_file = 'img_classifier_weights_%s_{epoch:02d}_{val_acc:.2f}.hdf5' %v
    model_file = 'img_classifier_model_%s.h5' %v
    history_file = 'img_classifier_history_%s.json' %v
    
    def save_model_and_weights():
        model.save(model_file)
        model.save_weights(model_weights_file)
        
        return 'Saved model and weights to disk!'

    def save_model_history(m):
        with open(history_file, 'wb') as history_json_file:
            json.dump(m.history, history_json_file)
        
        return 'Saved model history to disk!'
    
    def visualise_accuracy(m):
        plt.plot(m.history['acc'])
        plt.plot(m.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
      
    def visualise_loss(m):
        plt.plot(m.history['loss'])
        plt.plot(m.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    
    def model_callbacks():
        checkpoint = ModelCheckpoint(epoch_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stopping_patience, verbose=1, mode='auto')
        lr_tracker = SGDLearningRateTracker()
        
        return [checkpoint,early_stopping,lr_tracker]
        
    callbacks_list = model_callbacks()

    m = model.fit(train_data,train_labels,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,shuffle=True,validation_data=(val_data,val_labels),callbacks=callbacks_list)
    
    print save_model_and_weights()
    print save_model_history(m)
    
    visualise_accuracy(m)
    visualise_loss(m)
    
    return m

# Important function for mapping saliency
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations

# Evaluates test data performance and creates confusion matrix
def evaluate_model(m, weights, test_data, test_labels):    
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
     
    plt.close('all')

    m.load_weights(weights)
    m.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    print "Done compiling model."
    
    prediction = m.predict(test_data)
    prediction_labels = np_utils.to_categorical(np.argmax(prediction, axis=1), nb_classes)
    
    print 'Accuracy on test data:', accuracy_score(test_labels, prediction_labels)

    print 'Classification Report'
    print classification_report(test_labels, prediction_labels, target_names = class_names)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(np.argmax(test_labels, axis=1), np.argmax(prediction, axis=1))
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = class_names, normalize=False, title='Confusion matrix')

    plt.show()



#Main driver code
train_data, train_labels, test_data, test_labels, val_data, val_labels = load_dataset(dimension = '3d')

model = build_cnn(dimension = '3d')
#This will take time
fit_model(model, model_version, train_data, train_labels, val_data, val_labels)

#Load the fit model and evaluate performance on test
loaded_model = build_cnn(dimension = '3d')
evaluate_model(loaded_model, 'img_classifier_weights_v5.h5', test_data, test_labels)



def get_subject(subject_id):
    scan_range = train_data[subject_id][0]
    demographic = demographics[subject_id]
    
    return demographic, scan_range


# Visualize saliency map for a given subject. Might be better to use visualize_saliency
def saliencyMap(subject_number):
	subject = get_subject(subject_number)
	print subject[0] #Subject demographics

	img = np.expand_dims(np.expand_dims(subject[1],axis=0),axis=0)
	model = build_cnn(dimension = '3d')

	W = np.array(get_activations(model, 2, img))
	W = np.squeeze(W)
	img = np.squeeze(img)
	print W.shape
	w = W[5][14]
	w = scipy.misc.imresize(w, (img_size_x, img_size_y))
	plt.figure()
	jet_cmap = plt.cm.jet
	# Get the colormap colors
	heatmap = jet_cmap(np.arange(jet_cmap.N))
	# Set alpha
	heatmap[:,-1] = np.linspace(0, 1, jet_cmap.N)
	# Create new colormap
	heatmap = ListedColormap(heatmap)
	plt.axis('off')
	plt.imshow(img[31], interpolation='nearest', cmap=cm.binary)
	plt.imshow(w, cmap=heatmap, alpha=.9, interpolation='bilinear')
	plt.show()
	
