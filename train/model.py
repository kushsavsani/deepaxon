'''
-------------------------------- DEEPAXON --------------------------------
model contains the convolutional neural networks that can be used with the DeepAxon program to train a model
'''

#import necessary determinants for keras DeepLearning architecture development
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
)

def conv_block(inputs, filters, dropout=0.1):
    '''
    2 stage convolutional node that includes a dropout between the convolution tensors.
    The convolutional block is necessary for any downscaling process in the architecture.
    
    :param inputs: Tensor; input
    :param filters: Integer; the dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param dropout: Integer; dropout value for the dropout between convolutional layers; Default = 0.1
    
    :returns: Tensor; a tensor of rank 4+
    '''
    
    c1 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    d1 = Dropout(dropout)(c1)
    c2 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d1)
    return c2

def exp_block(up, skip, filters, dropout=0.1):
    '''
    Node that includes upscaling, concatentation, and 2 convolutional layers with a dropout in between.
    The convolutional block is necessary for any upscaling process in the architecture.
    
    :param up: Tensor; a single tensor that is below the current node of interest. Will be used in the upscaling process.
    :param skip: List of Tensor objects; all tensors that are on the same level as the current node of interest. Will be used in the concatenation process.
    :param filters: Integer; the dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param dropout: Integer; dropout value for the dropout between convolutional layers; Default = 0.1
    
    :returns: Tensor; a tensor of rank 4+
    '''
    
    #Upscaling process
    e1 = Conv2DTranspose(filters, (2,2), strides=(2,2), padding='same')(up)
    
    #Concatenation process
    g1 = skip #copy skip into list 'g1'
    g1.append(e1) #add the upscaled e1 tensor to the list g1
    g2 = concatenate(g1)
    
    c1 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(g2)
    d1 = Dropout(dropout)(c1)
    c2 = Conv2D(filters, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(d1)
    return c2

def deepaxon_plusplus_model(input_shape=(256, 256, 1), num_classes=3):
    '''
    U-Net++ architecture.
    
    :param input_shape: Tuple; format is (IMAGE HEIGHT, IMAGE WIDTH, IMAGE CHANNELS); Default = (256, 256, 1)
    :param num_classes: Integer; number of classes for multi-class segmentation.
    For segmented nerve images, the number of classes is 3 (background, myelin, axons)
    
    :returns: Model Object; A model grouping layers into an object with training/inference features.
    Once the model is created, you can config the model with losses and metrics with model.compile(), train the model with model.fit(),
    or use the model to do prediction with model.predict().
    '''
    
    inputs = Input(input_shape)
    
    x00 = conv_block(inputs, 16, dropout=0.1)
    x10 = conv_block(MaxPooling2D(pool_size=(2,2))(x00), 32, dropout=0.1)
    x20 = conv_block(MaxPooling2D(pool_size=(2,2))(x10), 64, dropout=0.2)
    x30 = conv_block(MaxPooling2D(pool_size=(2,2))(x20), 128, dropout=0.2)
    x40 = conv_block(MaxPooling2D(pool_size=(2,2))(x30), 256, dropout=0.3)
    
    x01 = exp_block(up=x10, skip=[x00], filters=16, dropout=0.1)
    x11 = exp_block(up=x20, skip=[x10], filters=32, dropout=0.1)
    x21 = exp_block(up=x30, skip=[x20], filters=64, dropout=0.2)
    x31 = exp_block(up=x40, skip=[x30], filters=128, dropout=0.2)
    
    x02 = exp_block(up=x11, skip=[x00,x01], filters=16, dropout=0.1)
    x12 = exp_block(up=x21, skip=[x10,x11], filters=32, dropout=0.1)
    x22 = exp_block(up=x31, skip=[x20,x21], filters=64, dropout=0.2)
    
    x03 = exp_block(up=x12, skip=[x00,x01,x02], filters=16, dropout=0.1)
    x13 = exp_block(up=x22, skip=[x10,x11,x12], filters=32, dropout=0.1)
    
    x04 = exp_block(up=x13, skip=[x00,x01,x02,x03], filters=16, dropout=0.1)
    
    outputs = Conv2D(num_classes, (1,1), activation='softmax')(x04)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def deepaxon_model(input_shape=(256, 256, 1), num_classes=3):
    '''
    Base U-Net architecture
    
    :param input_shape: Tuple; format is (IMAGE HEIGHT, IMAGE WIDTH, IMAGE CHANNELS); Default = (256, 256, 1)
    :param num_classes: Integer; number of classes for multi-class segmentation.
    For segmented nerve images, the number of classes is 3 (background, myelin, axons)
    
    :returns: Model Object; A model grouping layers into an object with training/inference features.
    Once the model is created, you can config the model with losses and metrics with model.compile(), train the model with model.fit(),
    or use the model to do prediction with model.predict().
    '''
    
    inputs = Input(input_shape)
    
    x00 = conv_block(inputs, 16, dropout=0.1)
    x10 = conv_block(MaxPooling2D(pool_size=(2,2))(x00), 32, dropout=0.1)
    x20 = conv_block(MaxPooling2D(pool_size=(2,2))(x10), 64, dropout=0.2)
    x30 = conv_block(MaxPooling2D(pool_size=(2,2))(x20), 128, dropout=0.2)
    x40 = conv_block(MaxPooling2D(pool_size=(2,2))(x30), 256, dropout=0.3)
    
    x31 = exp_block(up=x40, skip=[x30], filters=128, dropout=0.2)
    x21 = exp_block(up=x31, skip=[x20], filters=64, dropout=0.2)
    x11 = exp_block(up=x21, skip=[x10], filters=32, dropout=0.1)
    x01 = exp_block(up=x11, skip=[x00], filters=16, dropout=0.1)
    
    outputs = Conv2D(num_classes, (1,1), activation='softmax')(x01)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model