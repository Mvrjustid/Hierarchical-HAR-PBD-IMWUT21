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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # enable a clean command line  window. The thing is to define os. before importing tensorflow.
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # use GPU with ID=0
import tensorflow as tf
import scipy.io
import h5py
import keras
import xlwt as xw
import hdf5storage
from sklearn.metrics import *
from keras.layers import *
from keras.layers.core import *
from keras.models import *
from keras.regularizers import *
from keras.optimizers import *
from keras.losses import *
from keras.metrics import *
from keras import backend as K
from keras.callbacks import EarlyStopping
from scipy.linalg import fractional_matrix_power

# the following three functions are needed for the hierarchical connection between HAR and PBD modules, and for the Lambda layer to perform the graph normalization per GCN layer.
def HARExtend(nodefeature,time_step,body_num,class_num):
    # Extend the HAR output from (feature_dim,) to (time_step,body_num,feature_num,), in order to be concatenated with the raw input for the PBD module.
    # Define the value for time_step,body_num,class_num
    HARExtend = K.argmax(nodefeature,-1)
    HARExtend = K.one_hot(HARExtend,class_num)
    HARextend = K.expand_dims(HARExtend, axis=1)
    HARextend = K.expand_dims(HARextend, axis=1)
    HARextend = K.tile(HARextend, n=[time_step, body_num, 1, ])
    return HARextend

def output_of_adjmul(input_shape):
    return (input_shape[0],input_shape[1],input_shape[2],input_shape[3])

def adjmul(x):
    AdjNorm = MakeGraph() # refer to utils.py for this function.
    x = tf.cast(x, tf.float64) # this step could be removed in earlier Tensorflow versions.
    return tf.matmul(AdjNorm,x)

def build_model(timestep,body_num,feature_dim,gcn_units_HAR,lstm_units_HAR,gcn_units_PBD,lstm_units_PBD,num_class_HAR,num_class_PBD):
    # timestep=the length of current input data segment.
    # body_num=the number of nodes/joints of the input graph.
    # feature_num=the feature dimension of each node/joint.
    # gcn/lstm_units=the units of gcn and lstm layers.
    # num_class=the number of categories.

    # Mutual Input
    inputs = Input(shape=(timestep, body_num, feature_dim,), name='maininputs')

    # HAR LSTM:
    HARsingleinput = Input(shape=(timestep, gcn_units_HAR * body_num))
    HARLSTM1 = CuDNNLSTM(lstm_units_HAR, return_sequences=True, name='HARLSTM1')(HARsingleinput) # refer to gc-LSTM for how to use GPU for LSTM  using Tensorflow>=2.0
    HARDropout1 = Dropout(0.5)(HARLSTM1)
    HARLSTM2 = CuDNNLSTM(lstm_units_HAR, return_sequences=True, name='HARLSTM2')(HARDropout1)
    HARDropout2 = Dropout(0.5)(HARLSTM2)
    HARLSTM3 = CuDNNLSTM(lstm_units_HAR, return_sequences=False, name='HARLSTM3')(HARDropout2)
    HARDropout3 = Dropout(0.5)(HARLSTM3)
    HARLSTM = Model(inputs=[HARsingleinput], outputs=[HARDropout3])

    # HAR GCN
    HARDense1 = TimeDistributed(Conv1D(gcn_units_HAR, 1, activation='relu'), name='HARGCN1')(inputs)
    HARDense1 = Dropout(0.5)(HARDense1)
    HARDense2 = Reshape((timestep, gcn_units_HAR * body_num), )(HARDense1)
    HARTemporaloutput = HARLSTM(HARDense2)
    HARTemporaloutput1 = Dense(num_class_HAR, activation='softmax', name='HARout')(HARTemporaloutput)

    # PBD LSTM:
    PBDsingleinput = Input(shape=(timestep, body_num * gcn_units_PBD))
    PBDLSTM1 = CuDNNLSTM(lstm_units_PBD, return_sequences=True)(PBDsingleinput)
    PBDDropout1 = Dropout(0.5)(PBDLSTM1)
    PBDLSTM2 = CuDNNLSTM(lstm_units_PBD, return_sequences=True)(PBDDropout1)
    PBDDropout2 = Dropout(0.5)(PBDLSTM2)
    PBDLSTM3 = CuDNNLSTM(lstm_units_PBD, return_sequences=False)(PBDDropout2)
    PBDDropout3 = Dropout(0.5)(PBDLSTM3)
    PBDLSTM = Model(inputs=[PBDsingleinput], outputs=[PBDDropout3])

    # PBD GCN
    HARextend = Lambda(HARExtend)(HARTemporaloutput1)
    PBDinputs = concatenate([inputs, HARextend], axis=-1)
    PBDDense1 = TimeDistributed(Conv1D(gcn_units_PBD, 1, activation='relu'))(PBDinputs)
    PBDDense1 = Dropout(0.5)(PBDDense1)
    PBDDense2 = Lambda(adjmul, output_shape=output_of_adjmul)(PBDDense1)
    PBDDense2 = TimeDistributed(Conv1D(gcn_units_PBD, 1, activation='relu'))(PBDDense2)
    PBDDense2 = Dropout(0.5)(PBDDense2)
    PBDDense3 = Lambda(adjmul, output_shape=output_of_adjmul)(PBDDense2)
    PBDDense3 = TimeDistributed(Conv1D(gcn_units_PBD, 1, activation='relu'))(PBDDense3)
    PBDDense3 = Dropout(0.5)(PBDDense3)
    PBDDense4 = Reshape((timestep, body_num * gcn_units_PBD), )(PBDDense3)
    PBDTemporaloutput = PBDLSTM(PBDDense4)
    PBDTemporaloutput1 = Dense(num_class_PBD, activation='softmax', name='PBDout')(PBDTemporaloutput)

    HARmodel = Model(inputs=[inputs], outputs=[HARTemporaloutput1])
    model = Model(inputs=[inputs], outputs=[HARTemporaloutput1, PBDTemporaloutput1])

    return model, HARmodel
