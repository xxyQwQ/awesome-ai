import os
import sys
import argparse

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facial.arcface import Backbone
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
        result_identity = identity_model(F.interpolate(result_image, 112, mode='bilinear', align_corners=True))
        cosine_distance = 1 - F.cosine_similarity(F.normalize(source_identity), F.normalize(result_identity)).item()

    result_image = (0.5 * result_image + 0.5).squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])
    return result_image, cosine_distance


def main(config):
    os.makedirs(config.result, exist_ok=True)
    config.device = torch.device('cuda') if config.device == 'gpu' else torch.device('cpu')
    for key, value in vars(config).items():
        print('{}: {}'.format(key, value))

    identity_model = Backbone(50, 0.6, 'ir_se').to(config.device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/arcface.pth', map_location=config.device), strict=False)

    generator_model = InjectiveGenerator().to(config.device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(config.model, map_location=config.device), strict=False)

    list_distance = []
    for index in tqdm(range(20), desc='generating result'):
        source_file = os.path.join(config.image, 'source_{:02d}.jpg'.format(index))
        target_file = os.path.join(config.image, 'target_{:02d}.jpg'.format(index))

        result_image, cosine_distance = swap_face(config.device, identity_model, generator_model, source_file, target_file)
        Image.fromarray((255 * result_image).astype(np.uint8)).save(os.path.join(config.result, 'source_{}.jpg'.format(index)))
        list_distance.append(cosine_distance)

        result_image, cosine_distance = swap_face(config.device, identity_model, generator_model, target_file, source_file)
        Image.fromarray((255 * result_image).astype(np.uint8)).save(os.path.join(config.result, 'target_{}.jpg'.format(index)))
        list_distance.append(cosine_distance)

    mean_distance = np.mean(list_distance)
    print('mean identity cosine distance: {:.3f}'.format(mean_distance))


if __name__ == '__main__':
    sys.stdout = Logger('./appendix/result.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model path')
    parser.add_argument('--image', type=str, default='./appendix/image', help='image path')
    parser.add_argument('--result', type=str, default='./appendix/result', help='result path')
    parser.add_argument('--device', type=str, default='gpu', help='device')
    config = parser.parse_args()
    main(config)
