#-------------------------------------------------------------------
# Github page: https://github.com/Mvrjustid/IMWUT-Hierarchical-HAR-PBD
# Author: https://wangchongyang.ai
#
# @article{wang2020leveraging,
#      title={Leveraging Activity Recognition to Enable Protective Behavior Detection in Continuous Data},
#      author={Wang, Chongyang and Gao, Yuan and Mathur, Akhil and Williams, Amanda C. DE C. and Lane, Nicholas D and Bianchi-Berthouze, Nadia},
#      journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)},
#      publisher={ACM},
#      year={2021}}
#
#-------------------------------------------------------------------

import numpy as np
import scipy.io
import h5py
import os
import  gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # enable a clean command line  window. The thing is to define os. before importing tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
import hdf5storage
import tensorflow as tf
import keras
from sklearn.metrics import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from keras import backend as K
from scipy.linalg import fractional_matrix_power

# From tf 2.0, eager mode is enabled by default. Try the following if you are uncertain about it. Eager mode is needed for tf>=2.0 to run LSTM with GPU.
# tf.compat.v1.enable_eager_execution()
# tf.config.run_functions_eagerly(run_eagerly=True)

# the following two functions are needed for the Lambda layer to perform the graph normalization per GCN layer.
def output_of_adjmul(input_shape):
    return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])

def adjmul(x):
    AdjNorm = MakeGraph() # refer to utils.py for this function.
    x = tf.cast(x, tf.float64) # this step could be removed in earlier Tensorflow versions.
    return tf.matmul(AdjNorm,x)

def build_model(timestep,body_num,feature_dim,gcn_units,lstm_units,num_class):
    # timestep=the length of current input data segment.
    # body_num=the number of nodes/joints of the input graph.
    # feature_num=the feature dimension of each node/joint.
    # gcn/lstm_units=the units of gcn and lstm layers.
    # num_class=the number of categories.

    #LSTM with three layers.
    singleinput = Input(shape=(timestep, body_num*gcn_units))
    LSTM1 = LSTM(lstm_units, activation='tanh',recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True)(singleinput) #from Tensorflow 2.0, such LSTM definition meets the requirement of Eager execution.
    Dropout1 = Dropout(0.5)(LSTM1)                                                                                                                                                                                                                                             # for earlier Tensorflow, use CuDNNLSTM from Keras.
    LSTM2 = LSTM(lstm_units, activation='tanh',recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=True)(Dropout1)
    Dropout2 = Dropout(0.5)(LSTM2)
    LSTM3 = LSTM(lstm_units, activation='tanh',recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, return_sequences=False)(Dropout2)
    Dropout3 = Dropout(0.5)(LSTM3)
    TemporalProcessmodel = Model(inputs=[singleinput], outputs=[Dropout3])

    #GCN with three layers.
    inputs = Input(shape=(timestep, body_num, feature_dim,),name='maininputs') # the input is normalized graph sequence, refer to utils.py for how to transfer original TX(xyz) matrix into graph sequence input.
    Dense1 = TimeDistributed(Conv1D(gcn_units,1,activation='relu'))(inputs)
    Dense1 = Dropout(0.5)(Dense1)
    Dense2 = Lambda(adjmul, output_shape=output_of_adjmul)(Dense1) #please refer to Equation 1 and Thomas N Kipf et al. 2016 for the forward-passing of GCN.
    Dense2 = TimeDistributed(Conv1D(gcn_units, 1, activation='relu'))(Dense2)
    Dense2 = Dropout(0.5)(Dense2)
    Dense3 = Lambda(adjmul, output_shape=output_of_adjmul)(Dense2)
    Dense3 = TimeDistributed(Conv1D(gcn_units, 1, activation='relu'))(Dense3)
    Dense3 = Dropout(0.5)(Dense3)
    gcnoutput = Reshape((timestep, body_num*gcn_units),)(Dense3)
    Temporaloutput = TemporalProcessmodel(gcnoutput)
    Temporaloutput = Dense(num_class,activation='softmax')(Temporaloutput)
    model = Model(inputs=[inputs],outputs=[Temporaloutput])

    model.summary()
    return model
