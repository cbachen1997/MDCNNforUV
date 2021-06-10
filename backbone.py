
from __future__ import print_function, division
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np
#依赖库
import tensorflow.keras as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import LayerNormalization

def BN_LeakyReLU(input):
    
    norm = BatchNormalization(axis=-1)(input)
    output = LeakyReLU(alpha=0.2)(norm)
    
    return output


def MDCN(input_layers, n_filters):

    # stream_left
    conv_left = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=4, kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(0.0001))(input_layers)
    conv_left = BN_LeakyReLU(conv_left)
    # stream_middle_up
    conv_middle_1 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=3, kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(0.0001))(input_layers)
    conv_middle_1 = BN_LeakyReLU(conv_middle_1)
    # stream_right_up
    conv_right_1 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(input_layers)
    conv_right_1 = BN_LeakyReLU(conv_right_1)
    conv_right_2 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(conv_right_1)
    conv_right_2 = BN_LeakyReLU(conv_right_2)

    # stream_sum_1
    sum_1 = add([conv_middle_1, conv_right_2])

    # stream_middle_down
    conv_middle_2 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=3, kernel_initializer='he_normal',
                           kernel_regularizer=regularizers.l2(0.0001))(sum_1)
    conv_middle_2 = BN_LeakyReLU(conv_middle_2)
    # stream_right_down
    conv_right_3 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(sum_1)
    conv_right_3 = BN_LeakyReLU(conv_right_3)
    conv_right_4 = Conv2D(n_filters, (3, 3), padding='same', dilation_rate=2, kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(0.0001))(conv_right_3)
    conv_right_4 = BN_LeakyReLU(conv_right_4)

    # stream_sum_2
    sum_2 = add([conv_left, conv_middle_2, conv_right_4])

    return sum_2
def transpose(input):
    x=tf.transpose(input, perm=[0, 2, 1])
    return x

def expand_dims1(input):
    x=tf.expand_dims(input, axis=1)
    return x

def expand_dims2(input):
    x=tf.expand_dims(input, axis=-1)
    return x

def matmul(input):
    """input must be a  list"""
    return tf.matmul(input[0],input[1])

def gcnet_layer(inputs):
    
    x=inputs
    bs, h, w, c = x.get_shape().as_list()
    input_x = x
    input_x = Reshape((-1, c))(input_x)  # [N, H*W, C]
    input_x = Lambda(transpose)(input_x)  # [N,C,H*W]
    input_x = Lambda(expand_dims1)(input_x)

    context_mask = Conv2D(filters=1, kernel_size=(1, 1))(x)
    context_mask = Reshape((-1, 1))(context_mask)
    context_mask = softmax(context_mask, axis=1)  # [N, H*W, 1]
    context_mask = Lambda(transpose)(context_mask)
    context_mask = Lambda(expand_dims2)(context_mask)

    context = Lambda(matmul)([input_x,context_mask])  # [N,1,c,1]
    context = Reshape((1, 1, c))(context)

    context_transform = Conv2D(int(c/8), (1, 1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = relu(context_transform)
    context_transform = Conv2D(c, (1, 1))(context_transform)

    x= add([x,context_transform])

    return x


def DNCNN_non_local():

    n_filters = 64

    input_layer = Input(shape=[224, 224, 3])

    conv_1 = Conv2D(n_filters, (7, 7), padding='valid', strides=(2, 2), kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0001))(input_layer)
    conv_1 = BN_LeakyReLU(conv_1)
    print('conv_1: {}'.format(conv_1.shape))
    conv_2 = Conv2D(2*n_filters, (3, 3), padding='valid', strides=(2, 2), kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0001))(conv_1)
    conv_2 = BN_LeakyReLU(conv_2)
    print('conv_2: {}'.format(conv_2.shape))

    block_1 = MDCN(conv_2, 2*n_filters)  # 128
    print('block_1: {}'.format(block_1.shape))
    block_2 = MDCN(block_1, 2*n_filters)  # 128
    print('block_2: {}'.format(block_2.shape))

    conv_3 = Conv2D(4*n_filters, (3, 3), padding='valid', strides=(2, 2), kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0001))(block_2)
    conv_3 = BN_LeakyReLU(conv_3)
    print('conv_3: {}'.format(conv_3.shape))

    block_3 = MDCN(conv_3, 4*n_filters)  # 256
    print('block_3: {}'.format(block_3.shape))
    block_4 = MDCN(block_3, 4*n_filters)  # 256
    print('block_4: {}'.format(block_4.shape))

    conv_4 = Conv2D(8*n_filters, (3, 3), padding='valid', strides=(2, 2), kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.0001))(block_4)  # 512
    conv_4 = BN_LeakyReLU(conv_4)
    print('conv_4: {}'.format(conv_4.shape))

    block_5 = MDCN(conv_4, 8*n_filters)  # 256
    print('block_5: {}'.format(block_5.shape))
    block_6 = MDCN(block_5, 8*n_filters)  # 256
    print('block_6: {}'.format(block_6.shape))

    block_6 = gcnet_layer(block_6)

    gap = GlobalAveragePooling2D()(block_6)  # GAP
    fc = Dense(128, activation='relu')(gap)  # FC
    fc = Dropout(0.5)(fc)  # Dropout
    predictions = Dense(2, activation='softmax')(fc)  # softmax

    model_DNCNN = Model(inputs=input_layer, outputs=predictions)

    #optm = K.optimizers.Adam(lr=1e-5)
    #optm=K.optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0005, nesterov=True)
    optm = K.optimizers.Nadam(
        lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    #optm = K.optimizers.RMSprop(learning_rate=1e-4,decay=0.004)

    model_DNCNN.compile(
        optimizer=optm, loss='categorical_crossentropy', metrics=['acc'])

    return model_DNCNN


