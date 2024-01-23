#!/usr/bin/env python3

# script to resize ImageNet dataset
#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......

import os
import glob
import multiprocessing
from PIL import Image
from torchvision.transforms import Resize

def process_folder(root, folder, transform):
    print('process folder: {}'.format(folder))
    target_folder = os.path.join(root, os.path.basename(folder))
    os.makedirs(target_folder)
    file_list = glob.glob('{}/*'.format(folder))
    for file in file_list:
        target_file = os.path.join(target_folder, os.path.basename(file))
        transform(Image.open(file)).save(target_file)

if __name__ == '__main__':
    transform = Resize(128)
    num_cores = multiprocessing.cpu_count()
    process_pool = multiprocessing.Pool(processes=num_cores)
    folder_list = glob.glob('imagenet-1k/train/*')
    for folder in folder_list:
        process_pool.apply_async(process_folder, args=('imagenet-1k-128/train', folder, transform))
    folder_list = glob.glob('imagenet-1k/val/*')
    for folder in folder_list:
        process_pool.apply_async(process_folder, args=('imagenet-1k-128/val', folder, transform))
    process_pool.close()
    process_pool.join()
