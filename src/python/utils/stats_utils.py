import os
import numpy as np
import tensorflow as tf
from python.utils.img_utils import load_image
from python.utils.inference_utils import image_inference_batch
from python.utils.conv_utils import conv_2d
from python.utils.loss_utils import SSIM_batch, PSNR_batch, SSIM, PSNR

def get_test_stats(test_ds_list, model, psf, inference_kwargs):
    while len(psf.shape) < 4:
        psf = np.expand_dims(psf, axis=-1)
    batch_size = len(test_ds_list)
    stats = {"FirstPass": {"MAE": [], "MSE": [], "SSIM": [], "PSNR": []},
             "SecondPass": {"MAE": [], "MSE": [], "SSIM": [], "PSNR": []},}
    for i in range(0, len(test_ds_list), batch_size):
        imgs = []
        for file in test_ds_list[i:i+batch_size]:
            img = load_image(file)
            imgs.append(img)
        in_img_batch = np.stack(imgs, axis=0)
        out_img_batch, time_stats = image_inference_batch(in_img_batch, model, **inference_kwargs)
        conv_out_img_batch = conv_2d(out_img_batch, psf)
        conv_out_img_batch = conv_out_img_batch.numpy()
        MAE = tf.reduce_mean(tf.math.abs(in_img_batch - conv_out_img_batch), axis=(1, 2, 3))
        MAE = [val.numpy() for val in MAE]
        MSE = tf.reduce_mean(tf.math.square(in_img_batch - conv_out_img_batch), axis=(1, 2, 3))
        MSE = [val.numpy() for val in MSE]
        SSIM_val = SSIM(in_img_batch, conv_out_img_batch)
        SSIM_val = [val.numpy() for val in SSIM_val]
        PSNR_val = PSNR(in_img_batch, conv_out_img_batch)
        PSNR_val = [val.numpy() for val in PSNR_val]
        stats["FirstPass"]["MAE"].extend(MAE)
        # print(MAE)
        stats["FirstPass"]["MSE"].extend(MSE)
        # print(MSE)
        stats["FirstPass"]["SSIM"].extend(SSIM_val)
        # print(SSIM_val)
        stats["FirstPass"]["PSNR"].extend(PSNR_val)
        # print(PSNR_val)
        stats["FirstPass"]["time_stats"] = time_stats
        # print(time_stats)
        
        out_img_batch_cyc, time_stats = image_inference_batch(conv_out_img_batch, model, **inference_kwargs)
        MAE = tf.reduce_mean(tf.math.abs(out_img_batch - out_img_batch_cyc), axis=(1, 2, 3))
        MAE = [val.numpy() for val in MAE]
        MSE = tf.reduce_mean(tf.math.square(out_img_batch - out_img_batch_cyc), axis=(1, 2, 3))
        MSE = [val.numpy() for val in MSE]
        SSIM_val = SSIM(out_img_batch, out_img_batch_cyc)
        SSIM_val = [val.numpy() for val in SSIM_val]
        PSNR_val = PSNR(out_img_batch, out_img_batch_cyc)
        PSNR_val = [val.numpy() for val in PSNR_val]
        stats["SecondPass"]["MAE"].extend(MAE)
        # print(MAE)
        stats["SecondPass"]["MSE"].extend(MSE)
        # print(MSE)
        stats["SecondPass"]["SSIM"].extend(SSIM_val)
        # print(SSIM_val)
        stats["SecondPass"]["PSNR"].extend(PSNR_val)
        # print(PSNR_val)
        stats["SecondPass"]["time_stats"] = time_stats
        # print(time_stats)
    return stats