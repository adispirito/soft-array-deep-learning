# Customary Imports:
import tensorflow as tf
from tensorflow.image import random_flip_left_right as random_flip_lr
from tensorflow.image import random_flip_up_down as random_flip_ud
import keras_cv

###############################################################################
'''
AUGMENTATION UTILS:
'''
###############################################################################
# General Augmentation Functions:


@tf.function(reduce_retracing=True, jit_compile=True)
def normalize(tensor, clip=False,
              out_range=[0, 1],
              if_uint8=False):
    # Normalizes Tensor from 0-1
    out = tf.cast(tensor, tf.float32)
    if clip:
        out = tf.clip_by_value(out,
                               out_range[0],
                               out_range[1])
    elif if_uint8:
        out /= 255
        out *= (out_range[1] - out_range[0])
        out += out_range[0]
    else:
        out = tf.math.divide_no_nan(tf.math.subtract(out,
                                                     tf.math.reduce_min(out)),
                                    tf.math.subtract(tf.math.reduce_max(out),
                                                     tf.math.reduce_min(out)))
        out *= (out_range[1] - out_range[0])
        out += out_range[0]
    return out

@tf.function(reduce_retracing=True)
def add_rand_gaussian_noise(x, mean_val=0.0, std_lower=0.001,
                            std_upper=0.005, prob=0.1, out_range=[0, 1],
                            seed=None):
    '''
    This function introduces additive Gaussian Noise
    with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        # min_val = tf.math.reduce_min(x)
        # max_val = tf.math.reduce_max(x)
        # out_range = [min_val, max_val]
        
        std = tf.random.uniform(shape=(), minval=std_lower,
                                          maxval=std_upper, seed=seed)
        noise = tf.random.normal(shape=tf.shape(x), mean=mean_val,
                                 stddev=std, dtype=tf.float32,
                                 seed=seed)
        x = tf.math.add(x, noise)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function(reduce_retracing=True)
def batch_add_rand_gaussian_noise(x, 
                                  mean_val=0.0, 
                                  std_lower=0.001,
                                  std_upper=0.005, 
                                  prob=0.1, 
                                  out_range=[0, 1],
                                  seed=None):
    '''
    This function introduces additive Gaussian Noise
    with a given mean and std, at
    a certain given probability.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        # min_val = tf.math.reduce_min(x)
        # max_val = tf.math.reduce_max(x)
        # out_range = [min_val, max_val]
        
        std = tf.random.uniform(shape=(), minval=std_lower,
                                          maxval=std_upper, seed=seed)
        noise = tf.random.normal(shape=tf.shape(x)[1:], mean=mean_val,
                                 stddev=std, dtype=tf.float32,
                                 seed=seed)
        x = tf.math.add(x, noise)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function(reduce_retracing=True)
def add_rand_bright_shift(x, max_shift=0.12, prob=0.1,
                          out_range=[0, 1], seed=None):
    '''
    Equivalent to adjust_brightness() using a delta randomly
    picked in the interval [-max_delta, max_delta) with a
    given probability that this function is performed on an image.
    The pixels lower than 0 are clipped to 0 and the pixels higher
    than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        # min_val = tf.math.reduce_min(x)
        # max_val = tf.math.reduce_max(x)
        # out_range = [min_val, max_val]
        
        x = tf.image.random_brightness(image=x,
                                       max_delta=max_shift,
                                       seed=seed)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


@tf.function(reduce_retracing=True)
def add_rand_contrast(x, lower=0.2, upper=1.8,
                      prob=0.1, out_range=[0, 1],
                      seed=None):
    '''
    For each channel, this Op computes the mean
    of the image pixels in the channel
    and then adjusts each component x of each pixel
    to (x - mean) * contrast_factor + mean
    with a given probability that this function is
    performed on an image. The pixels lower
    than 0 are clipped to 0 and the pixels higher
    than 1 are clipped to 1.
    '''
    rand_var = tf.random.uniform(shape=(), seed=seed)
    if rand_var < prob:
        # min_val = tf.math.reduce_min(x)
        # max_val = tf.math.reduce_max(x)
        # out_range = [min_val, max_val]
        
        x = tf.image.random_contrast(image=x, lower=lower,
                                     upper=upper, seed=seed)
        x = tf.clip_by_value(x, out_range[0], out_range[1])
    return x


# @tf.function
# def random_jpeg_quality(x, 
#                         jpeg_quality_range=[70, 100], 
#                         prob=0.1, 
#                         out_range=[0, 1],
#                         seed=None):
#     rand_var = tf.random.uniform(shape=(), seed=seed)
#     if rand_var < prob:
#         min_val = tf.math.reduce_min(x)
#         max_val = tf.math.reduce_max(x)
#         out_range = [min_val, max_val]
#         x = normalize(x, clip=False, out_range=[0, 1])
#         x = tf.image.random_jpeg_quality(x, 
#                                          jpeg_quality_range[0], 
#                                          jpeg_quality_range[1], 
#                                          seed=seed)
#         x = normalize(x, clip=False, out_range=out_range)
#     return x


@tf.function
def augment(ds, prob=1/3, 
            max_shift=0.10,
            con_factor_range=[0.2, 1.8], 
            jpeg_quality_range=[70, 100],
            seed=7,
            out_range=[0, 1]):
    x = ds
    x = tf.expand_dims(x, axis=-1)
    # min_val = tf.math.reduce_min(x)
    # max_val = tf.math.reduce_max(x)
    # out_range = [min_val, max_val]
    # if tf.random.uniform(shape=(), seed=seed) > 0.5:
    #     x = random_flip_lr(x, seed=seed)
    # if tf.random.uniform(shape=(), seed=seed) > 0.5:
    #     x = random_flip_ud(x, seed=seed)
    x = add_rand_bright_shift(x, 
                              max_shift=max_shift, 
                              prob=prob, 
                              out_range=out_range, 
                              seed=seed)
    x = add_rand_contrast(x, 
                          lower=con_factor_range[0],
                          upper=con_factor_range[1],
                          prob=prob, 
                          out_range=out_range,
                          seed=seed)
    x = tf.squeeze(x)
    return x


@tf.function(reduce_retracing=True)
def augment_noise(ds, prob=1/3, mean=0.0, std_lower=0.003, std_upper=0.015,
                  seed=7,
                  out_range=[0, 1]):
    return add_rand_gaussian_noise(ds, 
                                   mean_val=mean, 
                                   std_lower=std_lower,
                                   std_upper=std_upper, 
                                   prob=prob,
                                   out_range=out_range,
                                   seed=seed)


@tf.function(reduce_retracing=True)
def batch_augment_noise(ds, 
                        prob=1/3, 
                        mean=0.0, 
                        std_lower=0.003, 
                        std_upper=0.015,
                        seed=7,
                        out_range=[0, 1]):
    return batch_add_rand_gaussian_noise(ds, 
                                         mean_val=mean, 
                                         std_lower=std_lower,
                                         std_upper=std_upper, 
                                         prob=prob,
                                         out_range=out_range,
                                         seed=seed)


# @tf.function
def get_keras_augment_pipeline(
    prob, 
    fill_mode='constant', 
    interpolation="bilinear",
    rg=20, zoom_range=[0.95, 0.95], 
    max_shift_xy=[0.1, 0.1], 
    shear_intensity=[0.2, 0.2],
    fill_value=-1.0,
    seed=None):

    layers = [
        # Random Rotation
        tf.keras.layers.RandomRotation(factor=rg, 
                                       fill_mode=fill_mode,
                                       interpolation=interpolation,
                                       seed=seed,
                                       fill_value=fill_value),
        # Random Zoom
        tf.keras.layers.RandomZoom(height_factor=zoom_range[0], 
                                   width_factor=zoom_range[1],
                                   fill_mode=fill_mode,
                                   interpolation=interpolation,
                                   seed=seed,
                                   fill_value=fill_value),
        # Random Translation
        tf.keras.layers.RandomTranslation(height_factor=max_shift_xy[0], 
                                          width_factor=max_shift_xy[1],
                                          fill_mode="constant",
                                          interpolation=interpolation,
                                          seed=seed,
                                          fill_value=fill_value),
        # Random Shear
        keras_cv.layers.RandomShear(x_factor=shear_intensity[0], 
                                    y_factor=shear_intensity[1],
                                    fill_mode=fill_mode,
                                    interpolation=interpolation,
                                    seed=seed,
                                    fill_value=fill_value),
        # Grid Mask
        keras_cv.layers.GridMask(ratio_factor=(0, 0.2),
                                 rotation_factor=0.15,
                                 fill_mode="constant",
                                 fill_value=fill_value,
                                 seed=seed),
        
    ]
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=layers,
        augmentations_per_image=len(layers),
        rate=prob
    )
    
    return pipeline


@tf.function
def keras_augment(ds, pipeline):
    return pipeline(ds)