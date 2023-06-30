import sys
from tqdm import tqdm
import numpy as np


class Logger:
    def __init__(self, path='./output/output.txt'):
        self.terminal = sys.stdout
        self.file = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        pass


def ComputeAccuracy(predication, label):
    return np.mean(predication == label)


def RGB2Gray(image):
    image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    return image.reshape(1, *image.shape)


def HOG(image, block=4, partition=8):
    image = RGB2Gray(image).squeeze(axis=0)
    height, width = image.shape
    gradient = np.zeros((2, height, width), dtype=np.float32)
    for i in range(1, height-1):
        for j in range(1, width-1):
            delta_x = image[i, j-1] - image[i, j+1]
            delta_y = image[i+1, j] - image[i-1, j]
            gradient[0, i, j] = np.sqrt(delta_x ** 2 + delta_y ** 2)
            gradient[1, i, j] = np.degrees(np.arctan2(delta_y, delta_x))
            if gradient[1, i, j] < 0:
                gradient[1, i, j] += 180
    unit = 360 / partition
    vertical, horizontal = height // block, width // block
    feature = np.zeros((vertical, horizontal, partition), dtype=np.float32)
    for i in range(vertical):
        for j in range(horizontal):
            for k in range(block):
                for l in range(block):
                    rho = gradient[0, i*block+k, j*block+l]
                    theta = gradient[1, i*block+k, j*block+l]
                    index = int(theta // unit)
                    feature[i, j, index] += rho
            feature[i, j] /= np.linalg.norm(feature[i, j]) + 1e-6
    return feature.reshape(-1)


def BatchRGB2Gray(images):
    image_list = []
    for image in tqdm(images):
        image_list.append(RGB2Gray(image))
    return np.array(image_list)


def BatchHOG(images, block=4, partition=8):
    feature_list = []
    for image in tqdm(images):
        feature_list.append(HOG(image, block, partition))
    return np.array(feature_list)


def FrobeniusProduct(X, Y):
    return np.sum(X * Y)


def MatrixSimilarity(X, Y):
    return FrobeniusProduct(X, Y) / np.sqrt(FrobeniusProduct(X, X) * FrobeniusProduct(Y, Y))
