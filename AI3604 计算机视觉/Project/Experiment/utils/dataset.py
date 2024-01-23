import os
import glob

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import TensorDataset
from torchvision import transforms

from utils.function import SquarePad, ColorReverse


class CharacterDataset(TensorDataset):
    def __init__(self, data_root, reference_count: int = 1):
        self.reference_count = reference_count
        self.template_root = os.path.join(data_root, 'template')
        self.script_root = os.path.join(data_root, 'script')

        self.template_list = glob.glob('{}/*.png'.format(self.template_root))
        self.character_map = {os.path.basename(path): i for i, path in enumerate(self.template_list)}

        self.script_list = []
        writer_list = glob.glob('{}/*'.format(self.script_root))
        for writer in tqdm(writer_list, desc='loading dataset'):
            character_list = glob.glob('{}/*.png'.format(writer))
            self.script_list.append(character_list)

        self.remap_list = []
        for writer in range(len(self.script_list)):
            for character in range(len(self.script_list[writer])):
                self.remap_list.append((writer, character))

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            ColorReverse(),
            SquarePad(),
            transforms.Resize((128, 128), antialias=True),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __getitem__(self, index):
        writer, character = self.remap_list[index]
        reference_path = np.random.choice(self.script_list[writer], self.reference_count, replace=True)
        script_path = self.script_list[writer][character]
        character_name = os.path.basename(script_path)
        template_path = os.path.join(self.template_root, character_name)
        writer_label, character_label = torch.tensor(writer), torch.tensor(self.character_map[character_name])

        reference_image = torch.concat([self.transforms(Image.open(path)) for path in reference_path])
        template_image = self.transforms(Image.open(template_path))
        script_image = self.transforms(Image.open(script_path))
        return reference_image, writer_label, template_image, character_label, script_image

    def __len__(self):
        return len(self.remap_list)
    
    @property
    def writer_count(self):
        return len(self.script_list)
    
    @property
    def character_count(self):
        return len(self.template_list)
