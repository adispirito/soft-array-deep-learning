import numpy as np
import os
import shutil
import imageio
import scipy
import scipy.io as sio
import h5py
import math
from functools import partial
import multiprocessing as mp
from tqdm.auto import tqdm
###############################################################################
'''
IMPORTING MAT VOLUMES / SAVING SLICES
'''


def import_mat(path):
    try:
        mat_dict = sio.loadmat(path, struct_as_record=True)
        mat_dict['mat_file_version'] = '<7.3'
    except NotImplementedError:
        mat_dict = dict(h5py.File(path, 'r'))
        mat_dict['mat_file_version'] = '7.3'
    return mat_dict


def save_mat(path, mdict):
    if type(mdict) is not dict:
        key = os.path.splitext(os.path.basename(path))[0]
        mdict = {key: mdict}
    sio.savemat(path, mdict)


def import_vol(path,
               keyword='pa_rec0',
               norm=False):
    mat_dict = import_mat(path)
    vol = np.array(mat_dict[keyword], dtype=np.float32)
    if mat_dict['mat_file_version'] == '7.3': vol = np.transpose(vol)
    if norm:
        vol = normalize(vol,
                        norm=norm)
    return vol


def normalize(data, norm="norm_amp"):
    if norm == "norm_0_1":
        data = (data - np.min(data)) / \
               (np.max(data) - np.min(data))
    elif norm == "norm_amp":
        data = data / np.max(np.abs(data))
        data = np.nan_to_num(data)
    return data


def get_filename(path):
    base = os.path.basename(path)
    filename = os.path.splitext(base)[0]
    return filename


def save_vol(vol, 
             orig_filename,
             output_dir,
             file_format='.npy',
             norm="norm_amp",
             save_min_max=False,
             sig_figs=8):
    if save_min_max:
        filename = f'{orig_filename}_' + \
                   f'min{int(np.round(np.min(vol), sig_figs)*(10**sig_figs))}_' + \
                   f'max{int(np.round(np.max(vol), sig_figs)*(10**sig_figs))}'
    else:
        filename = orig_filename
    vol = normalize(vol, norm=norm)
    filename += file_format
    filepath = os.path.join(output_dir, filename)
    save_npy(vol, filepath)


def save_npy(tensor, filepath):
    # Save npy
    np.save(filepath, 
            tensor,
            allow_pickle=True,
            fix_imports=True)


def save_npy_vols(input_dir,
                  output_dir,
                  keyword="pa_rec0",
                  delete_previous=False,
                  verbose=True,
                  norm="norm_amp",
                  **kwargs):
    FILE_FORMAT = ".npy"
    if not os.path.exists(output_dir):
        if verbose: print(f'Creating {output_dir} directory...')
        os.mkdir(output_dir)
    elif delete_previous:
        if verbose: print(f'Deleting existing {output_dir} directory...')
        shutil.rmtree(output_dir)
        if verbose: print(f'Creating {output_dir} directory...')
        os.mkdir(output_dir)
    file_list = [file for file in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, file))]
    vol_count = 0
    if verbose: print('Saving volumes...')
    for file in tqdm(file_list):
        if verbose: print(f'Saving vol {vol_count+1} out of {len(file_list)}')
        path = os.path.join(input_dir, file)
        vol = import_vol(path, keyword=keyword)
        if verbose: print(f'Volume shape: {vol.shape}')
        orig_filename = get_filename(file)
        save_vol(vol,
                 orig_filename=orig_filename,
                 output_dir=output_dir,
                 file_format=FILE_FORMAT,
                 norm=norm,
                 **kwargs)

        vol_count += 1
