import numpy as np
import tensorflow as tf

# @tf.function(experimental_relax_shapes=True)
def conv_2d_fft(data, psf):
    if len(data.shape) == 2:
        data = tf.expand_dims(data, axis=0)
        data = tf.expand_dims(data, axis=-1)
    elif len(data.shape) == 3:
        data = tf.expand_dims(data, axis=0)
    if len(psf.shape) == 2:
        psf = tf.expand_dims(psf, axis=0)
        psf = tf.expand_dims(psf, axis=-1)
    elif len(psf.shape) == 3:
        psf = tf.expand_dims(psf, axis=0)
    psf_shape = psf.shape.as_list()
    data_shape = data.shape.as_list()
    if psf_shape == data_shape:
        out = conv_2d_fft_func(data, psf) # Degradation Function
    elif psf_shape < data_shape:
        # Pad to make dimensions equal
        psf, paddings = pad(psf, psf_shape, data_shape)
        out = conv_2d_fft_func(data, psf)
    else:
        # Pad to make dimensions equal
        # print(psf_shape)
        # print(data_shape)
        data, paddings = pad(data, data_shape, psf_shape)
        data = conv_2d_fft_func(data, psf)
        out = data[:,
                   paddings[1][0]:-paddings[1][1],
                   paddings[2][0]:-paddings[2][1],
                   :]
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def conv_2d_fft_func(x, kernel):
    kernel_fft = fft_2d(kernel)
    x_fft = fft_2d(x)
    out = tf.math.multiply(x_fft, kernel_fft)
    out = ifft_2d(out)
    out = tf.signal.fftshift(out)
    # out = tf.squeeze(out, axis=0)
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def fft_2d(img):
    out = tf.transpose(img, perm=[0, 3, 1, 2])
    out = tf.signal.fft2d(tf.cast(out, tf.complex64))
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def ifft_2d(img):
    out = tf.transpose(img, perm=[0, 3, 1, 2])
    out = tf.signal.ifft2d(out)
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    out = tf.math.real(out) # was complex so get real part
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def pad(img, in_shape, out_shape, **kwargs):
    i_pad = out_shape[1] - in_shape[1]
    j_pad = out_shape[2] - in_shape[2]
    paddings = [[0, 0],
                [i_pad//2, i_pad - i_pad//2],
                [j_pad//2, j_pad - j_pad//2],
                [0, 0]]
    # print(img.shape)
    out = tf.pad(img, paddings, **kwargs)
    return out, paddings

@tf.function(reduce_retracing=True, jit_compile=True)
def rfft_2d(img):
    out = tf.transpose(img, perm=[0, 3, 1, 2])
    out = tf.cast(out, dtype=tf.float32)
    out = tf.signal.rfft2d(out)
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def irfft_2d(img):
    out = tf.transpose(img, perm=[0, 3, 1, 2])
    out = tf.signal.irfft2d(out)
    out = tf.transpose(out, perm=[0, 2, 3, 1])
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def conv_2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME"):
    # kernel = tf.expand_dims(kernel, axis=0)
    out = tf.nn.conv2d(x, kernel, 
                       strides=strides, 
                       padding=padding)
    # out = tf.squeeze(out, axis=0)
    return out