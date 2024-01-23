import os
import sys
import glob

import hydra
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.resnet import Bottleneck

from facial.arcface import Backbone
from facial.hopenet import Hopenet
from utils.logger import Logger
from model.generator import InjectiveGenerator


def load_image(device, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(Image.open(image_path)).unsqueeze(0).to(device)


def swap_face(device, identity_model, generator_model, source_file, target_file):    
    source_image = load_image(device, source_file)
    target_image = load_image(device, target_file)

    with torch.no_grad():
        source_identity = identity_model(F.interpolate(source_image, 112, mode='bilinear', align_corners=True))
        result_image, _ = generator_model(target_image, source_identity)

    result_image = (0.5 * result_image + 0.5).squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])
    return result_image


def generate_result(device, model_path, dataset_path, checkpoint_path):
    source_path = os.path.join(checkpoint_path, 'source_index')
    os.makedirs(source_path)
    target_path = os.path.join(checkpoint_path, 'target_index')
    os.makedirs(target_path)

    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/arcface.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator().to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    file_list = glob.glob('{}/*.*g'.format(dataset_path))
    for source_file in tqdm(file_list, desc='generating result'):
        source_name, _ = os.path.splitext(os.path.basename(source_file))
        source_folder = os.path.join(source_path, source_name)
        os.makedirs(source_folder, exist_ok=True)
        for target_file in file_list:
            target_name, _ = os.path.splitext(os.path.basename(target_file))
            target_folder = os.path.join(target_path, target_name)
            os.makedirs(target_folder, exist_ok=True)
            result_image = swap_face(device, identity_model, generator_model, source_file, target_file)
            Image.fromarray((255 * result_image).astype(np.uint8)).save(os.path.join(source_folder, '{}_{}.jpg'.format(source_name, target_name)))
            Image.fromarray((255 * result_image).astype(np.uint8)).save(os.path.join(target_folder, '{}_{}.jpg'.format(source_name, target_name)))

    return source_path, target_path


def compute_identity(device, dataset_path, source_path):
    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/arcface.pth', map_location=device), strict=False)

    real_identity, fake_identity, fake_label = [], [], []

    real_list = glob.glob('{}/*.*g'.format(dataset_path))
    for label, real_file in enumerate(tqdm(real_list, desc='computing identity')):
        real_name, _ = os.path.splitext(os.path.basename(real_file))

        real_image = load_image(device, real_list[label])
        with torch.no_grad():
            identity = identity_model(F.interpolate(real_image, 112, mode='bilinear', align_corners=True))
        real_identity.append(identity)

        fake_list = glob.glob('{}/{}/*.*g'.format(source_path, real_name))
        for fake_file in fake_list:
            fake_image = load_image(device, fake_file)
            with torch.no_grad():
                identity = identity_model(F.interpolate(fake_image, 112, mode='bilinear', align_corners=True))
            fake_identity.append(identity)

            fake_label.append(label)

    real_identity = torch.cat(real_identity, dim=0)
    fake_identity = torch.cat(fake_identity, dim=0)
    fake_label = torch.tensor(fake_label).to(device)

    fake_distance = (real_identity.unsqueeze(1) - fake_identity.unsqueeze(0)).square().sum(dim=2)
    fake_prediction = fake_distance.argmin(dim=0)
    return (fake_prediction == fake_label).float().mean().item()


def compute_posture(device, dataset_path, target_path):
    model = Hopenet(Bottleneck, [3, 4, 6, 3], 66).to(device)
    model.eval()
    model.load_state_dict(torch.load('./facial/hopenet.pth', map_location=device), strict=False)

    index = torch.arange(66).to(device).float()
    posture_loss = []

    real_list = glob.glob('{}/*.*g'.format(dataset_path))
    for real_file in tqdm(real_list, desc='computing posture'):
        real_name, _ = os.path.splitext(os.path.basename(real_file))

        real_image = load_image(device, real_file)
        with torch.no_grad():
            yaw, pitch, roll = model(real_image)
        yaw, pitch, roll = torch.softmax(yaw, 1), torch.softmax(pitch, 1), torch.softmax(roll, 1)

        real_yaw = 3 * torch.sum(index * yaw, 1) - 99
        real_pitch = 3 * torch.sum(index * pitch, 1) - 99
        real_roll = 3 * torch.sum(index * roll, 1) - 99
        
        fake_list = glob.glob('{}/{}/*.*g'.format(target_path, real_name))
        for fake_file in fake_list:
            fake_image = load_image(device, fake_file)
            with torch.no_grad():
                yaw, pitch, roll = model(fake_image)
            yaw, pitch, roll = torch.softmax(yaw, 1), torch.softmax(pitch, 1), torch.softmax(roll, 1)

            fake_yaw = 3 * torch.sum(index * yaw, 1) - 99
            fake_pitch = 3 * torch.sum(index * pitch, 1) - 99
            fake_roll = 3 * torch.sum(index * roll, 1) - 99

            posture_loss.append(torch.sqrt(torch.square(real_yaw - fake_yaw) + torch.square(real_pitch - fake_pitch) + torch.square(real_roll - fake_roll)))

    return torch.cat(posture_loss, dim=0).mean().item()


@hydra.main(version_base=None, config_path='./config', config_name='evaluation')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    perform_inference = bool(config.parameter.perform_inference)
    evaluate_identity = bool(config.parameter.evaluate_identity)
    evaluate_posture = bool(config.parameter.evaluate_posture)
    temporary_path = str(config.parameter.temporary_path)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'evaluation.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # generate result
    if perform_inference:
        source_path, target_path = generate_result(device, model_path, dataset_path, checkpoint_path)
        print('generate temporary result in: {}\n'.format(checkpoint_path))
    else:
        source_path = os.path.join(temporary_path, 'source_index')
        target_path = os.path.join(temporary_path, 'target_index')
        print('load temporary result from: {}\n'.format(checkpoint_path))

    # start evaluation
    if evaluate_identity:
        id_retrieval = compute_identity(device, dataset_path, source_path,)
        print('id-retrieval: {:.2%}\n'.format(id_retrieval))
    if evaluate_posture:
        posture = compute_posture(device, dataset_path, target_path)
        print('posture: {:.2f}\n'.format(posture))
    print('save evaluation result in: {}\n'.format(checkpoint_path))


if __name__ == '__main__':
    main()
