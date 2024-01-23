import os
import sys
import glob
import pickle

import hydra
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torchvision import transforms

from utils.logger import Logger
from utils.function import SquarePad, ColorReverse, RecoverNormalize, SciptTyper
from model.generator import SynthesisGenerator


@hydra.main(version_base=None, config_path='./config', config_name='inference')
def main(config):
    # load configuration
    model_path = str(config.parameter.model_path)
    reference_path = str(config.parameter.reference_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    reference_count = int(config.parameter.reference_count)
    target_text = str(config.parameter.target_text)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'inference.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    generator_model = SynthesisGenerator(reference_count=reference_count).to(device)
    generator_model.eval()
    generator_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # create transform
    input_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        ColorReverse(),
        SquarePad(),
        transforms.Resize((128, 128), antialias=True),
        transforms.Normalize((0.5,), (0.5,))
    ])
    output_transform = transforms.Compose([
        RecoverNormalize(),
        transforms.Resize((64, 64), antialias=True),
        ColorReverse(),
        transforms.ToPILImage()
    ])
    align_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64), antialias=True),
    ])

    # fetch reference
    reference_list = []
    file_list = glob.glob('{}/*.png'.format(reference_path))
    for file in tqdm(file_list, desc='fetching reference'):
        image = Image.open(file)
        reference_list.append(image)
    while len(reference_list) < reference_count:
        reference_list.extend(reference_list)
    reference_list = reference_list[:reference_count]
    reference_image = [np.array(align_transform(image)) for image in reference_list]
    reference_image = np.concatenate(reference_image, axis=1)
    Image.fromarray(reference_image).save(os.path.join(checkpoint_path, 'reference.png'))
    reference = [input_transform(image) for image in reference_list]
    reference = torch.cat(reference, dim=0).unsqueeze(0).to(device)
    print('fetch {} reference images\n'.format(reference_count))

    # load dictionary
    with open('./assets/dictionary/character.pkl', 'rb') as file:
        character_map = pickle.load(file)
    character_remap = {value: key for key, value in character_map.items()}
    with open('./assets/dictionary/punctuation.pkl', 'rb') as file:
        punctuation_map = pickle.load(file)
    punctuation_remap = {value: key for key, value in punctuation_map.items()}
    print('load dictionary from archive\n')

    # generate script
    script_typer = SciptTyper()
    for word in tqdm(target_text, desc='generating script'):
        if word in character_remap.keys():
            image = Image.open(os.path.join('./assets/character', '{}.png'.format(character_remap[word])))
            template = input_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                result, _, _ = generator_model(reference, template)
            result = output_transform(result.squeeze(0).detach().cpu())
            script_typer.insert_word(result, word_type='character')
        elif word in punctuation_remap.keys():
            image = Image.open(os.path.join('./assets/punctuation', '{}.png'.format(punctuation_remap[word])))
            result = align_transform(image)
            script_typer.insert_word(result, word_type='punctuation')
        else:
            raise ValueError('word {} is not supported'.format(word))
    print('generate {} words from text\n'.format(len(target_text)))
    
    # save result
    result_image = script_typer.plot_result()
    result_image.save(os.path.join(checkpoint_path, 'result.png'))
    print('save inference result in: {}\n'.format(checkpoint_path))


if __name__ == '__main__':
    main()
