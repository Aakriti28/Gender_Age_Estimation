import argparse
import cv2
import glob
import numpy as np
import os, sys
import torch
from basicsr.utils import imwrite
import copy

from gfpgan import GFPGANer


# ##print("aaaaaaaaaaaaaaaaaaaaaaaaaaaas")
aligned=False
suffix=None
only_center_face=False
ext='auto'


def prepare_gan():

    os.system('python GFPGAN/setup.py develop')
    version='1.3'
    upscale = 2
    bg_upsampler='realsrgan'
    bg_tile=400

    # ------------------------ set up background upsampler ------------------------
    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():  # CPU
            import warnings
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=bg_tile,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
    else:
        bg_upsampler = None

    # ------------------------ set up GFPGAN restorer ------------------------
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
    else:
        raise ValueError(f'Wrong model version {version}.')

    # determine model paths
    model_path = os.path.join('GFPGAN/experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {model_name} does not exist.')

    return model_path, upscale, arch, channel_multiplier, bg_upsampler

model_path, upscale, arch, channel_multiplier, bg_upsampler = prepare_gan()

restorer = GFPGANer(
    model_path=model_path,
    upscale=upscale,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler)


def face_sr(global_people_info):
# running the images
    for i in range(len(global_people_info)):
        if global_people_info[i].is_face and global_people_info[i].frame_number % 5 == 0:
            ##print(global_people_info[i].frame_face.size)
            cropped_faces, restored_faces, restored_img = restorer.enhance(
                copy.deepcopy(global_people_info[i].frame_face), has_aligned=aligned, only_center_face=only_center_face, paste_back=True)
            global_people_info[i].sr_img = copy.deepcopy(restored_img)
            # plt.imshow(restored_img)
            ##print(i)
    return global_people_info