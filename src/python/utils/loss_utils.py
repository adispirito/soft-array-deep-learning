# Customary Imports:
import tensorflow as tf
import numpy as np
from functools import partial
# import tensorflow_addons as tfa


###############################################################################
'''
MODEL UTILS:
'''
###############################################################################
# Custom Metrics:

@tf.function(reduce_retracing=True, jit_compile=True)
def normalize(tensor, clip=False,
              out_range=[0, 1]):
    # Normalizes Tensor from 0-1
    out = tf.cast(tensor, tf.float32)
    if clip:
        out = tf.clip_by_value(out,
                               out_range[0],
                               out_range[1])
    else:
        out = tf.math.divide_no_nan(tf.math.subtract(out,
                                                     tf.math.reduce_min(out)),
                                    tf.math.subtract(tf.math.reduce_max(out),
                                                     tf.math.reduce_min(out)))
        out *= (out_range[1] - out_range[0])
        out += out_range[0]
    return out

@tf.function(reduce_retracing=True, jit_compile=True)
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    PSNR = tf.image.psnr(y_true_norm, y_pred_norm, max_pixel)
    return PSNR


@tf.function(reduce_retracing=True, jit_compile=True)
def PSNR_batch(y_true, y_pred):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    psnr = partial(tf.image.psnr, max_pixel=max_pixel)
    PSNR_val = tf.stack([psnr(y_true_norm[k, ...], y_pred_norm[k, ...]) 
                         for k in range(y_pred_norm.shape[0])])
    return PSNR_val


@tf.function(reduce_retracing=True, jit_compile=True)
def SSIM(y_true, y_pred, **kwargs):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    SSIM_val = tf.image.ssim(y_true_norm, y_pred_norm, max_pixel, **kwargs)
    # SSIM_val = tf.clip_by_value(SSIM_val, 0, 1)
    return SSIM_val


@tf.function(reduce_retracing=True)
def SSIM_non_jit(y_true, y_pred, **kwargs):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    SSIM_val = tf.image.ssim(y_true_norm, y_pred_norm, max_pixel, **kwargs)
    # SSIM_val = tf.clip_by_value(SSIM_val, 0, 1)
    return SSIM_val


@tf.function(reduce_retracing=True, jit_compile=True)
def SSIM_batch(y_true, y_pred, **kwargs):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    ssim = partial(tf.image.ssim, max_pixel=max_pixel, **kwargs)
    SSIM_val = tf.stack([ssim(y_true_norm[k, ...], y_pred_norm[k, ...]) 
                         for k in range(y_pred_norm.shape[0])])
    # SSIM_val = tf.clip_by_value(SSIM_val, 0, 1)
    return SSIM_val


@tf.function(reduce_retracing=True, jit_compile=True)
def MS_SSIM(y_true, y_pred, **kwargs):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    MS_SSIM_val = tf.image.ssim_multiscale(y_true_norm,
                                           y_pred_norm,
                                           max_pixel,
                                           **kwargs)
    # MS_SSIM_val = tf.clip_by_value(MS_SSIM_val, 0, 1)
    return MS_SSIM_val


@tf.function(reduce_retracing=True)
def MS_SSIM_non_jit(y_true, y_pred, **kwargs):
    max_pixel = 1.0
    clip = True
    out_range = [-1, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true)
    y_pred_norm = tf.map_fn(map_norm, y_pred)
    clip = False
    out_range = [0, 1]
    map_norm = partial(normalize, clip=clip, out_range=out_range)
    y_true_norm = tf.map_fn(map_norm, y_true_norm)
    y_pred_norm = tf.map_fn(map_norm, y_pred_norm)
    MS_SSIM_val = tf.image.ssim_multiscale(y_true_norm,
                                           y_pred_norm,
                                           max_pixel, 
                                           **kwargs)
    # MS_SSIM_val = tf.clip_by_value(MS_SSIM_val, 0, 1)
    return MS_SSIM_val


@tf.function(reduce_retracing=True, jit_compile=True)
def downsample(y_true, y_pred, down_ratio=[5, 1]):

    down_y_true = y_true[:, ::down_ratio[0], ::down_ratio[1], :]
    down_y_pred = y_pred[:, ::down_ratio[0], ::down_ratio[1], :]
    return down_y_true, down_y_pred


@tf.function(reduce_retracing=True, jit_compile=True)
def down_consistency_loss(y_true, y_pred, down_ratio=[5, 1],
                          pixelwise_loss='MAE'):

    down_y_true, down_y_pred = downsample(y_true,
                                          y_pred,
                                          down_ratio)
    if pixelwise_loss == 'MSE':
        pixelwise_loss = tf.keras.losses.MeanSquaredError()
    else:
        pixelwise_loss = tf.keras.losses.MeanAbsoluteError()
    loss = pixelwise_loss(down_y_true, down_y_pred)

    return loss


@tf.function(reduce_retracing=True, jit_compile=True)
def rfft_loss(y_true, y_pred):
    from python.utils.conv_utils import rfft_2d
    loss = tf.math.reduce_mean(
            tf.math.abs(
                tf.math.squared_difference(rfft_2d(y_true), 
                                           rfft_2d(y_pred))
            )
        )
    return loss


# Model Loss Function:
def model_loss(B1=1.0,
               B2=0.0,
               pixelwise_loss=tf.keras.losses.MeanSquaredError(),
               eager=False):
    B1 = tf.cast(B1, tf.float32)
    B2 = tf.cast(B2, tf.float32)
    @tf.function(reduce_retracing=True)#, jit_compile=True)
    def custom_loss_func(y_true, y_pred):
        loss = pixelwise_loss(y_true, y_pred)
        out = tf.math.multiply(B1, loss)
        if B2 != 0 and B2 is not None:
            tv_loss = tf.reduce_mean(total_variation(y_pred))
            out = tf.math.add(out, tf.math.multiply(B2, tv_loss))
        return out
    return custom_loss_func


@tf.function(reduce_retracing=True, jit_compile=True)
def ssim_loss(y_true, y_pred):
    loss = tf.math.subtract(1.0, SSIM(y_true, y_pred))
    return loss


@tf.function(reduce_retracing=True)
def total_variation(vol, name='total_variation'):
    with tf.name_scope(name):
        ndims = vol.get_shape().ndims

        if ndims == 3 or ndims == 4:
            # The input is a single vol with shape [height, width, depth] or [height, width, depth, channels].

            # Calculate the difference of neighboring pixel-values.
            # The vol are shifted one pixel along the height and width by slicing.
            pixel_dif1 = vol[1:, :, :, ...] - vol[:-1, :, :, ...]
            pixel_dif2 = vol[:, 1:, :, ...] - vol[:, :-1, :, ...]
            pixel_dif3 = vol[:, :, 1:, ...] - vol[:, :, :-1, ...]

            # Sum for all axis. (None is an alias for all axis.)
            sum_axis = None
        elif ndims == 5:
            # The input is a batch of vol with shape:
            # [batch, height, width, depth, channels].

            # Calculate the difference of neighboring pixel-values.
            # The vol are shifted one pixel along the height and width by slicing.
            pixel_dif1 = vol[:, 1:, :, :, ...] - vol[:, :-1, :, :, ...]
            pixel_dif2 = vol[:, :, 1:, :, ...] - vol[:, :, :-1, :, ...]
            pixel_dif3 = vol[:, :, :, 1:, ...] - vol[:, :, :, :-1, ...]

            # Only sum for the last 3 axis.
            # This results in a 1-D tensor with the total variation for each image.
            sum_axis = [1, 2, 3, 4]
        else:
            raise ValueError('\'vols\' must be either 3, 4, or 5-dimensional.')

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences and summing over the appropriate axis.
        # print(sum_axis)
        # print(pixel_dif1.shape)
        tot_var = (
            tf.math.reduce_sum(tf.math.abs(pixel_dif1), axis=sum_axis) +
            tf.math.reduce_sum(tf.math.abs(pixel_dif2), axis=sum_axis) + 
            tf.math.reduce_sum(tf.math.abs(pixel_dif3), axis=sum_axis)
        )

    return tot_var