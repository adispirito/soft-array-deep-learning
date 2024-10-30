# Customary Imports:
import tensorflow as tf
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Add, Input, \
                                    BatchNormalization, UpSampling3D, \
                                    Concatenate, AveragePooling3D, \
                                    SpatialDropout3D, Activation

###################################################################################################
'''
MODEL DEFINITION:
UNet
'''
# Model Definition Based on https://www.tandfonline.com/doi/full/10.1080/17415977.2018.1518444?af=R
def func_model_def(img,
                   filters=32,
                   kernel_size=3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='glorot_normal',
                   kernel_regularizer=None,
                   bias_regularizer=None,
                   activity_regularizer=None,
                   dilation_rate=1,
                   interp_up=True,
                   strided_conv=False,
                   drop_prob=0,
                   norm_layer=BatchNormalization):

    conv_args = {
        'filters': filters,
        'kernel_size': kernel_size,
        'activation': activation,
        'padding': padding,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
        'activity_regularizer': activity_regularizer,
        'dilation_rate': dilation_rate,
        'drop_prob': drop_prob,
        'norm_layer': norm_layer,
    }
    out = img
    shortcut0 = out
    [out, shortcut1] = DownBlock(out, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    [out, shortcut2] = DownBlock(out, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    [out, shortcut3] = DownBlock(out, strided_conv, **conv_args)
    conv_args['filters'] *= 2
    [out, shortcut4] = DownBlock(out, strided_conv, **conv_args)

    conv_args['filters'] *= 2
    out = BridgeBlock(out, interp_up, **conv_args)

    out = Concatenate()([out, shortcut4])
    conv_args['filters'] //= 2
    out = UpBlock(out, interp_up, **conv_args)
    out = Concatenate()([out, shortcut3])
    conv_args['filters'] //= 2
    out = UpBlock(out, interp_up, **conv_args)
    out = Concatenate()([out, shortcut2])
    conv_args['filters'] //= 2
    out = UpBlock(out, interp_up, **conv_args)
    out = Concatenate()([out, shortcut1])
    
    conv_args['filters'] //= 2
    out = Conv3D_Block(out, **conv_args)
    out = Conv3D_Block(out, **conv_args)

    # 1 x 1 Convolution Followed by Identity as Activation:
    conv_args['filters'] = 1
    conv_args['kernel_size'] = 1
    conv_args['activation'] = 'linear'
    conv_args.pop("drop_prob")
    conv_args.pop("norm_layer")
    out = Conv3D(**conv_args)(out)
    out = Add()([out, shortcut0])
    return out


def DownBlock(img, strided_conv=True, **conv_args):
    #print('DOWN_in: '+str(img.shape))
    out = Conv3D_Block(img, **conv_args)
    out = Conv3D_Block(out, **conv_args)
    shortcut = out
    kwargs = conv_args.copy()
    if strided_conv:
        kwargs['strides'] = 2
        kwargs['dilation_rate'] = 1
        out = Conv3D_Block(out, **kwargs)
    else:
        out = MaxPooling3D(pool_size = 2)(out)
    #print('DOWN_out: '+str(out.shape))
    return [out, shortcut]


def BridgeBlock(img, interp_up=True, **conv_args):
    #print('UP_in: '+str(img.shape))
    out = Conv3D_Block(img, **conv_args)
    out = Conv3D_Block(out, **conv_args)
    kwargs = conv_args.copy()
    if interp_up:
        kwargs['filters'] //= 2
        kwargs['kernel_size'] = 2
        out = UpConv3D_Block(out, **kwargs)
    else:
        kwargs['filters'] *= 2
        kwargs['kernel_size'] = 1
        kwargs['activation'] = "linear"
        out = Conv3D_Block(out, **kwargs)
        out = Pixel_Shuffle(out, upscale_factor=2)
    #print('UP_out: '+str(out.shape))
    return out


def UpBlock(img, interp_up=True, **conv_args):
    #print('UP_in: '+str(img.shape))
    out = Conv3D_Block(img, **conv_args)
    out = Conv3D_Block(out, **conv_args)
    kwargs = conv_args.copy()
    if interp_up:
        kwargs['filters'] //= 2
        kwargs['kernel_size'] = 2
        out = UpConv3D_Block(out, **kwargs)
    else:
        kwargs['filters'] *= 2
        kwargs['kernel_size'] = 1
        kwargs['activation'] = "linear"
        out = Conv3D_Block(out, **kwargs)
        out = Pixel_Shuffle(out, upscale_factor=2)
    #print('UP_out: '+str(out.shape))
    return out

###################################################################################################
'''
MODEL FUNCTIONS:
'''

def Conv3D_Block(img, drop_prob=0, norm_layer=None, **conv_args):
    use_bias = norm_layer != BatchNormalization
    activation = conv_args["activation"]
    conv_args["activation"] = "linear"
    out = Conv3D(use_bias=use_bias, **conv_args)(img)
    if norm_layer is not None: out = norm_layer()(out)
    if activation != "linear": out = Activation(activation)(out)
    if drop_prob > 0 and drop_prob is not None: out = SpatialDropout3D(drop_prob)(out)
    return out


def UpConv3D_Block(img, **conv_args):
    out = UpSampling3D(size = 2)(img)
    out = Conv3D_Block(out, **conv_args)
    return out


def Pixel_Shuffle(img, upscale_factor=2):
    out = tf.nn.depth_to_space(img, upscale_factor)
    return out

def AvgPool(img, pool_size=2, strides=(1, 1), padding="same"):
    kwargs = {
        "pool_size": pool_size, 
        "strides": strides, 
        "padding": padding,
    }
    out = AveragePooling3D(**kwargs)(img)
    return out

###################################################################################################
'''
FUNCTIONAL MODEL DEFINITION:
'''

def DL_Model_Func(input_shape, model_name="DL_Model", **kwargs):
    accept_kwargs = ['filters',
                     'kernel_size',
                     'activation',
                     'kernel_initializer',
                     'padding',
                     'kernel_regularizer',
                     'bias_regularizer',
                     'activity_regularizer',
                     'dilation_rate',
                     'strided_conv',
                     'interp_up',
                     'drop_prob',
                     'norm_layer']
    model_inputs = Input(shape=input_shape, name='in_img')
    model_outputs = func_model_def(
        model_inputs,
        **{i:kwargs[i] for i in kwargs
           if i in accept_kwargs}
    )
    model = Model(model_inputs, model_outputs, name=model_name)
    return model
