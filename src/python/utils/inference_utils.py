import numpy as np
import tensorflow as tf
import time

@tf.function(reduce_retracing=True, jit_compile=True)
def image_inference(img, model, min_model_shape=(32, 32)):
    img = tf.squeeze(img)
    orig_shape = img.shape
    pad_i = (np.ceil(img.shape[0]/min_model_shape[0])
             * min_model_shape[0]
             - img.shape[0]).astype(np.int32)
    pad_j = (np.ceil(img.shape[1]/min_model_shape[1])
             * min_model_shape[1]
             - img.shape[1]).astype(np.int32)
    img = tf.pad(img,
                 [[0, pad_i],
                  [0, pad_j]],
                 'CONSTANT')  # Replaced 'SYMMETRIC' with 'CONSTANT'
    img = img[tf.newaxis, ..., tf.newaxis]
    out_img = model(img, training=False)
    # out_img = tf.clip_by_value(out_img, 0, 1)
    out_img = tf.image.crop_to_bounding_box(out_img,
                                            0, 0,
                                            orig_shape[0],
                                            orig_shape[1])
    out_img = tf.squeeze(out_img)
    return out_img


def image_inference_batch(batch, model, min_model_shape=(32, 32), batch_size=1, verbose=False, tensor_rt=False):
    if len(batch.shape) == 2:
        batch = batch[tf.newaxis, ..., tf.newaxis]
    elif len(batch.shape) == 3:
        batch = batch[tf.newaxis, ...]
    out_batch = np.zeros(batch.shape, dtype=np.float32)
    times = []
    n_vals = []
    if verbose: print(f"Batch Shape: {batch.shape}")
    for i in range(0, batch.shape[0], batch_size):
        img = batch[i:i+batch_size, ...]
        if verbose: print(f"Image Shape: {img.shape}")
        if verbose: print(f"Processing Index: {i} to {i+batch_size} out of {batch.shape[0]}")
        start = time.time()
        out_img = image_inference_map(img, 
                                      model, 
                                      min_model_shape=min_model_shape, 
                                      batch_size=batch_size,
                                      tensor_rt=tensor_rt)
        end = time.time()
        out_batch[i:i+batch_size, ...] = out_img.numpy()
        times.append(end-start)
        n_vals.append(img.shape[0])
    mean_time = np.sum(times)/np.sum(n_vals)
    stddev_time = np.std(times) # This will not calculate the true stddev
    if verbose: print(f'\nTime Per Iter (Mean +/- StdDev): {mean_time} +/- {stddev_time}\n\n')
    time_stats = {
        "times": times,
        "mean": mean_time,
        "stddev": stddev_time,
        "n": n_vals,
    }
    return out_batch, time_stats


def image_inference_map(img, model, min_model_shape=(32, 32), batch_size=1, tensor_rt=False):
    img = tf.cast(img, dtype=tf.float32)
    orig_shape = img.shape
    if orig_shape[0] == 1:
        img = tf.squeeze(img)
        img = img[tf.newaxis, ...]
    else:
        img = tf.squeeze(img)
    pad_i = (np.ceil(img.shape[1]/min_model_shape[0])
             * min_model_shape[0]
             - img.shape[1]).astype(np.int32)
    pad_j = (np.ceil(img.shape[2]/min_model_shape[1])
             * min_model_shape[1]
             - img.shape[2]).astype(np.int32)
    img = tf.pad(img,
                 [[0, 0],
                  [0, pad_i],
                  [0, pad_j]],
                 'CONSTANT')  # Replaced 'SYMMETRIC' with 'CONSTANT'
    img = img[..., tf.newaxis]
    if tensor_rt:
        out_img = model(img)
        out_img = out_img[next(iter(out_img))]
    else:
        out_img = model(img, training=False)
    out_img = tf.image.crop_to_bounding_box(out_img,
                                            0, 0,
                                            orig_shape[1],
                                            orig_shape[2])
    return out_img