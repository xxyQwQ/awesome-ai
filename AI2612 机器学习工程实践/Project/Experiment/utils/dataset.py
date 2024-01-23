import glob

import numpy as np
from tqdm import tqdm
from PIL import Image

from torch.utils.data import TensorDataset
from torchvision import transforms


class FaceDataset(TensorDataset):
    def __init__(self, data_root, same_prob=0.2):
        self.dataset = []
        path_list = glob.glob('{}/*'.format(data_root))

        for path in tqdm(path_list, desc='loading dataset'):
            file_list = glob.glob('{}/*.*g'.format(path))
            self.dataset.append(file_list)

        self.remap = []
        for identity in range(len(self.dataset)):
            for serial in range(len(self.dataset[identity])):
                self.remap.append((identity, serial))

        self.same_prob = same_prob
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        identity, serial = self.remap[index]
        source_image = Image.open(self.dataset[identity][serial])

        if np.random.rand() > self.same_prob:
            index = np.random.randint(len(self.remap))
            identity, serial = self.remap[index]
            target_image = Image.open(self.dataset[identity][serial])
            same_identity = False
        else:
            target_image = source_image.copy()
            same_identity = True

        source_image = self.transforms(source_image)
        target_image = self.transforms(target_image)
        return source_image, target_image, same_identity

    def __len__(self):
        return len(self.remap)
