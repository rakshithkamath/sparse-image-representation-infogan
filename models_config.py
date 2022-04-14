

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Conv2D,Conv2DTranspose,LeakyReLU,BatchNormalization,Activation
from tensorflow.keras.initializers import RandomNormal




def define_discriminator_and_recognition(cat_dim, in_shape=(32,32,1)):
    """
    Args:
    cat_dim:  number of categorical variables present in the dataset for our study
    in_shape: input dimension vectors for our dataset
    """
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=in_shape)
    # layer 1
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.1)(d)
    # layer 2
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = BatchNormalization()(d)
    # layer 3
    d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.1)(d)
    d = BatchNormalization()(d)
    # flatten feature maps
    d = Flatten()(d)
    # real/fake output
    out_classifier = Dense(1, activation='sigmoid')(d)
    # define d model
    d_model = Model(in_image, out_classifier)
    # compile d model
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    
    # create q model layers
    q = Dense(128)(d)
    q = BatchNormalization()(q)
    q = LeakyReLU(alpha=0.1)(q)
    # q model output
    out_codes = Dense(cat_dim, activation='softmax')(q)
    # define q model
    q_model = Model(in_image, out_codes)

    return d_model, q_model




def define_generator(generator_input_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    # image generator input
    input_dim = Input(shape=(generator_input_dim,))
    
    # FC -layer 1
    gen = Dense(2*2*448, kernel_initializer=init)(input_dim)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    gen = Reshape((2, 2, 448))(gen)
    
    # Deconv -layer 2
    gen = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    gen = BatchNormalization()(gen)
    
    # Deconv -layer 3
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
  
    # Deconv -layer 4
    gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    gen = Activation('relu')(gen)
    
    # Deconv -layer 5
    gen = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
    # tanh output
    out_layer = Activation('tanh')(gen)
    
    # define model
    gen_model = Model(input_dim, out_layer)
    
    return gen_model




def define_gan(g_model, d_model, q_model):
    # make weights in the discriminator (some shared with the q model) as not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect g outputs to d inputs
    d_output = d_model(g_model.output)
    # connect g outputs to q inputs
    q_output = q_model(g_model.output)
    # define composite model
    model = Model(g_model.input, [d_output, q_output])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
    return model







