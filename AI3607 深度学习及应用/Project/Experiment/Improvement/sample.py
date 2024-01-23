import os
import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from dataset import PuzzleDataset
from model import PuzzleSolver


def sample(model_segment, model_path, sample_height=4, sample_width=4, save_path='./figure', save_name='figure'):
    loader = PuzzleDataset(path='./dataset', group='test', batch=sample_height*sample_width, segment=model_segment)
    model = PuzzleSolver(segment=model_segment)
    model.load(model_path)

    model.eval()
    images, _ = next(iter(loader))
    outputs = model(images)
    predictions = outputs.argmax(2)[0]
    images, predictions = images.numpy(), predictions.numpy()

    plt.figure(figsize=(2*sample_width, 2*sample_height))
    for i in range(sample_height):
        for j in range(sample_width):
            position = i * sample_width + j
            image = np.transpose(images[position], (0, 2, 3, 1))
            pieces = []
            for u in range(model_segment):
                piece = []
                for v in range(model_segment):
                    index = u * model_segment + v
                    piece.append(image[index])
                pieces.append(np.concatenate(piece, axis=1))
            image = np.concatenate(pieces, axis=0)
            plt.subplot(sample_height, sample_width, position+1)
            plt.imshow(image)
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '{}_sample_problem.png'.format(save_name)))

    plt.figure(figsize=(2*sample_width, 2*sample_height))
    for i in range(sample_height):
        for j in range(sample_width):
            position = i * sample_width + j
            image, prediction = np.transpose(images[position], (0, 2, 3, 1)), np.argsort(predictions[position])
            pieces = []
            for u in range(model_segment):
                piece = []
                for v in range(model_segment):
                    index = u * model_segment + v
                    piece.append(image[prediction[index]])
                pieces.append(np.concatenate(piece, axis=1))
            image = np.concatenate(pieces, axis=0)
            plt.subplot(sample_height, sample_width, position+1)
            plt.imshow(image)
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '{}_sample_solution.png'.format(save_name)))


if __name__ == '__main__':
    np.random.seed(0)
    jt.set_global_seed(0)
    jt.flags.use_cuda = 0
    sample(2, './model/2x2_epoch_100.pkl', save_name='2x2')
    sample(3, './model/3x3_epoch_100.pkl', save_name='3x3')
    sample(4, './model/4x4_epoch_300.pkl', save_name='4x4')
