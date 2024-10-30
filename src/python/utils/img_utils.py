import os
import imageio
import numpy as np

###############################################################################
'''
IMAGE UTIL FUNCTIONS:
'''

def load_image(path):
    path = path.decode()
    # Load Image
    if path.endswith('.npy'):
        img = np.load(path)
    else:
        img = imageio.imread(path)
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    img = np.array(img, dtype=np.float32)
    return img

def normalize_image(img, out_range=[0,1]):
    if img.dtype != np.float32:
        img = img.astype(np.float32)/np.iinfo(img.dtype).max
    else:
        img = (img - np.min(img))/(np.max(img) - np.min(img))
    out = img
    out *= (out_range[1] - out_range[0])
    out += out_range[0]
    return out

def crop_img(img, crop_size, divis=None):
    if len(img.shape) < 3:
        img = img[..., None]
        remove_dim = True
    else:
        remove_dim = False
    crop_size = [int(size) for size in crop_size]
    mid_i = img.shape[0]//2
    mid_j = img.shape[1]//2
    if (list(img.shape[:2]) > crop_size):
        img = img[mid_i - crop_size[0]//2:mid_i - crop_size[0]//2 + crop_size[0], 
                  mid_j - crop_size[1]//2:mid_j - crop_size[0]//2 + crop_size[0],
                  ...]
    pad_x = divis - (img.shape[0] % divis)
    pad_y = divis - (img.shape[1] % divis)
    if pad_x == divis:
        pad_x = 0
    if pad_y == divis:
        pad_y = 0
    img = np.pad(img, 
                 ((int(np.floor(pad_x/2)), int(np.ceil(pad_x/2))),
                  (int(np.floor(pad_y/2)), int(np.ceil(pad_x/2))),
                  (0, 0)), 
                 'constant')
    if remove_dim:
        img = img[..., 0]
    return img