import os
import sys

import cv2
import hydra
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision import transforms

from facial.mtcnn import MTCNN
from facial.arcface import Backbone
from utils.logger import Logger
from model.generator import InjectiveGenerator


def swap_face(device, detector_model, generator_model, source_identity, target_path, tensor_transform, mask_aligned):
    # load target
    target_image = cv2.imread(target_path)
    try:
        target_image_aligned, target_inversion = detector_model.align(Image.fromarray(target_image[:, :, ::-1]), crop_size=(224, 224), return_trans_inv=True)
    except Exception as _:
        return None
    target_image_aligned = tensor_transform(target_image_aligned).unsqueeze(0).to(device)

    # start inference
    with torch.no_grad():
        result_image_aligned, _ = generator_model(target_image_aligned, source_identity)
        result_image_aligned = (0.5 * result_image_aligned + 0.5).squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])[:, :, ::-1]

    # generate result
    mask = cv2.warpAffine(mask_aligned, target_inversion, target_image.shape[:2][::-1], borderValue=(0, 0, 0))[:, :, np.newaxis]
    result_image = cv2.warpAffine(result_image_aligned, target_inversion, target_image.shape[:2][::-1], borderValue=(0, 0, 0))
    result_image = (1 - mask) * target_image + mask * (255 * result_image)
    return result_image


@hydra.main(version_base=None, config_path='./config', config_name='inference')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    source_path = str(config.parameter.source_path)
    target_path = str(config.parameter.target_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    file_type = str(config.parameter.file_type)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'inference.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    detector_model = MTCNN()

    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/arcface.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator().to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    
    # create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create mask
    mask_aligned = np.zeros((224, 224), dtype=np.float32)
    for i in range(224):
        for j in range(224):
            mask_aligned[i, j] = 1 - np.minimum(1, np.sqrt(np.square(i - 112) + np.square(j - 112)) / 112)
    mask_aligned = cv2.dilate(mask_aligned, None, iterations=20)

    # load source
    source_image = cv2.imread(source_path)
    source_image_aligned = detector_model.align(Image.fromarray(source_image[:, :, ::-1]), crop_size=(224, 224))
    source_image_aligned = transform(source_image_aligned).unsqueeze(0).to(device)
    print('source shape: {}'.format(source_image.transpose([2, 0, 1]).shape))

    # extract identity
    with torch.no_grad():
        source_identity = identity_model(F.interpolate(source_image_aligned, 112, mode='bilinear', align_corners=True))
    
    # image inference
    if file_type == 'image':
        print('target shape: {}\n'.format(cv2.imread(target_path).transpose([2, 0, 1]).shape))
        result_image = swap_face(device, detector_model, generator_model, source_identity, target_path, transform, mask_aligned)
        cv2.imwrite(os.path.join(checkpoint_path, 'result.jpg'), result_image)
    # video inference
    elif file_type == 'video':
        frame_path = os.path.join(checkpoint_path, 'video_frame')
        os.makedirs(frame_path)

        video_capture = cv2.VideoCapture(target_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        print('frame count: {}'.format(frame_count))
        print('frame rate: {}'.format(frame_rate))
        print('frame size: {}\n'.format((frame_height, frame_width)))

        frame_list = []
        for frame_index in tqdm(range(frame_count), desc='extracting frame'):
            _, frame_image = video_capture.read()
            frame_file = os.path.join(frame_path, '{:0>8d}.jpg'.format(frame_index))
            frame_list.append(frame_file)
            cv2.imwrite(frame_file, frame_image)
        video_capture.release()

        for frame_file in tqdm(frame_list, desc='generating result'):
            result_image = swap_face(device, detector_model, generator_model, source_identity, frame_file, transform, mask_aligned)
            if result_image is not None:
                cv2.imwrite(frame_file, result_image)

        video_writer = cv2.VideoWriter(os.path.join(checkpoint_path, 'result.mp4'), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        for frame_file in tqdm(frame_list, desc='exporting video'):
            frame_image = cv2.imread(frame_file)
            video_writer.write(frame_image)
        video_writer.release()
    
    print('save inference result in: {}\n'.format(checkpoint_path))


if __name__ == '__main__':
    main()
