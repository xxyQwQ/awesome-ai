import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid


class SciptTyper(object):
    def __init__(self, word_size=64, line_width=960):
        self.word_size = word_size
        self.line_width = line_width
        self.result_list = None
        self.insert_line()

    def __stochastic_transform(self, word):
        transform = transforms.RandomAffine(degrees=5, scale=(0.95, 1.05), shear=5, fill=255)
        return transform(word)

    def __convert_word(self, word, threshold=224):
        matrix = np.array(word)
        matrix[matrix >= threshold] = 255
        pixel = np.argwhere(matrix < threshold)
        left, right = np.min(pixel[:, 1]), np.max(pixel[:, 1])
        return matrix[:, left: right + 1]

    def insert_line(self):
        if self.result_list is None:
            self.result_list = []
        else:
            self.result_list.append(self.result_line)
        self.result_line = np.full((self.word_size, self.line_width), 255, dtype=np.uint8)
        self.result_cursor = 0

    def __insert_matrix(self, matrix, padding=4, blank=False):
        matrix = np.pad(matrix, ((0, 0), (padding, 0)), 'constant', constant_values=255)
        width = matrix.shape[1]
        if self.result_cursor + width > self.line_width:
            if blank:
                return
            self.insert_line()
        self.result_line[:, self.result_cursor: self.result_cursor + width] = matrix
        self.result_cursor += width

    def insert_space(self):
        space = np.full((self.word_size, self.word_size//2), 255, dtype=np.uint8)
        self.__insert_matrix(space, blank=True)

    def insert_word(self, word, word_type='character'):
        word = self.__stochastic_transform(word)
        matrix = self.__convert_word(word)
        blank = np.full((self.word_size, 4), 255, dtype=np.uint8)
        if self.result_cursor == 0:
            self.__insert_matrix(blank, blank=True)
        self.__insert_matrix(matrix, blank=False)
        if word_type == 'punctuation':
            self.__insert_matrix(blank, blank=True)
            
    def plot_result(self):
        if self.result_line is not None and self.result_cursor > 0:
            self.insert_line()
        result = np.concatenate(self.result_list, axis=0)
        return Image.fromarray(result)

    def plot_result_gui(self):
        result_list = self.result_list.copy()
        if self.result_line is not None and self.result_cursor > 0:
            result_list.append(self.result_line)
        result = np.concatenate(result_list, axis=0)
        return result

class SquarePad(object):
    def __call__(self, image):
        _, width, height = image.shape
        target_size = max(width, height)
        pad_width = (target_size - width) // 2 + 10
        pad_height = (target_size - height) // 2 + 10
        return F.pad(image, (pad_width, pad_height, pad_width, pad_height), 'constant', 0)


class ColorReverse(object):
    def __call__(self, image):
        image = 1 - image
        image /= image.max()
        return image


class RecoverNormalize(object):
    def __call__(self, image):
        image = 0.5 * image + 0.5
        return image


def plot_sample(reference_image, template_image, script_image, result_image):
    def plot_grid(input):
        batch_size = input.shape[0]
        return 0.5 * make_grid(input.detach().cpu(), nrow=batch_size) + 0.5
    
    reference_count = reference_image.shape[1]
    reference_image = [plot_grid(reference_image[:, i, :, :].unsqueeze(1)) for i in range(reference_count)]
    template_image, script_image, result_image = plot_grid(template_image), plot_grid(script_image), plot_grid(result_image)
    sample_image = torch.cat([*reference_image, template_image, script_image, result_image], dim=1)
    return sample_image.numpy()
