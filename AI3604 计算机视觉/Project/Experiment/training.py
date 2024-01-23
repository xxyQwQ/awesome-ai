import os
import sys
import time

import hydra
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.logger import Logger
from utils.dataset import CharacterDataset
from utils.function import plot_sample
from model.generator import SynthesisGenerator
from model.discriminator import MultiscaleDiscriminator


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    # load configuration
    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    batch_size = int(config.parameter.batch_size)
    num_workers = int(config.parameter.num_workers)
    reference_count = int(config.parameter.reference_count)
    num_iterations = int(config.parameter.num_iterations)
    report_interval = int(config.parameter.report_interval)
    save_interval = int(config.parameter.save_interval)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # load dataset
    dataset = CharacterDataset(dataset_path, reference_count=reference_count)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    print('image number: {}\n'.format(len(dataset)))

    # create model
    generator_model = SynthesisGenerator(reference_count=reference_count).to(device)
    generator_model.train()

    discriminator_model = MultiscaleDiscriminator(dataset.writer_count, dataset.character_count).to(device)
    discriminator_model.train()

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=config.parameter.generator.learning_rate, betas=(0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=config.parameter.discriminator.learning_rate, betas=(0, 0.999), weight_decay=1e-4)

    # start training
    current_iteration = 0
    current_time = time.time()

    while current_iteration < num_iterations:
        for reference_image, writer_label, template_image, character_label, script_image in dataloader:
            current_iteration += 1

            reference_image, writer_label, template_image, character_label, script_image = reference_image.to(device), writer_label.to(device), template_image.to(device), character_label.to(device), script_image.to(device)

            # generator
            generator_optimizer.zero_grad()

            result_image, template_structure, reference_style = generator_model(reference_image, template_image)

            loss_generator_adversarial = 0
            loss_generator_classification = 0
            for prediction_reality, prediction_writer, prediction_character in discriminator_model(result_image):
                loss_generator_adversarial += F.binary_cross_entropy(prediction_reality, torch.ones_like(prediction_reality))
                loss_generator_classification += F.cross_entropy(prediction_writer, writer_label) + F.cross_entropy(prediction_character, character_label)

            result_structure = generator_model.structure(result_image)
            loss_generator_structure = 0
            for i in range(len(result_structure)):
                loss_generator_structure += 0.5 * torch.mean(torch.square(template_structure[i] - result_structure[i]))

            result_style = generator_model.style(result_image.repeat_interleave(reference_count, dim=1))
            loss_generator_style = 0.5 * torch.mean(torch.square(reference_style - result_style))

            loss_generator_reconstruction = F.l1_loss(result_image, script_image)

            loss_generator = config.parameter.generator.loss_function.weight_adversarial * loss_generator_adversarial + config.parameter.generator.loss_function.weight_classification * loss_generator_classification + config.parameter.generator.loss_function.weight_structure * loss_generator_structure + config.parameter.generator.loss_function.weight_style * loss_generator_style + config.parameter.generator.loss_function.weight_reconstruction * loss_generator_reconstruction
            loss_generator.backward()
            generator_optimizer.step()

            # discriminator
            discriminator_optimizer.zero_grad()

            loss_discriminator_adversarial = 0
            loss_discriminator_classification = 0
            for prediction_reality, prediction_writer, prediction_character in discriminator_model(result_image.detach()):
                loss_discriminator_adversarial += F.binary_cross_entropy(prediction_reality, torch.zeros_like(prediction_reality))
                loss_discriminator_classification += F.cross_entropy(prediction_writer, writer_label) + F.cross_entropy(prediction_character, character_label)

            for prediction_reality, prediction_writer, prediction_character in discriminator_model(script_image):
                loss_discriminator_adversarial += F.binary_cross_entropy(prediction_reality, torch.ones_like(prediction_reality))
                loss_discriminator_classification += F.cross_entropy(prediction_writer, writer_label) + F.cross_entropy(prediction_character, character_label)

            loss_discriminator = config.parameter.discriminator.loss_function.weight_adversarial * loss_discriminator_adversarial + config.parameter.discriminator.loss_function.weight_classification * loss_discriminator_classification
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # report
            if current_iteration % report_interval == 0:
                last_time = current_time
                current_time = time.time()
                iteration_time = (current_time - last_time) / report_interval

                print('iteration {} / {}:'.format(current_iteration, num_iterations))
                print('time: {:.6f} seconds per iteration'.format(iteration_time))
                print('generator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}, structure loss: {:.6f}, style loss: {:.6f}, reconstruction loss: {:.6f}'.format(loss_generator.item(), loss_generator_adversarial.item(), loss_generator_classification.item(), loss_generator_structure.item(), loss_generator_style.item(), loss_generator_reconstruction.item()))
                print('discriminator loss: {:.6f}, adversarial loss: {:.6f}, classification loss: {:.6f}\n'.format(loss_discriminator.item(), loss_discriminator_adversarial.item(), loss_discriminator_classification.item()))

            # save
            if current_iteration % save_interval == 0:
                save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))
                os.makedirs(save_path, exist_ok=True)

                image_path = os.path.join(save_path, 'sample.png')
                generator_path = os.path.join(save_path, 'generator.pth')
                discriminator_path = os.path.join(save_path, 'discriminator.pth')

                image = plot_sample(reference_image, template_image, script_image, result_image)[0]
                Image.fromarray((255 * image).astype(np.uint8)).save(image_path)
                torch.save(generator_model.state_dict(), generator_path)
                torch.save(discriminator_model.state_dict(), discriminator_path)

                print('save sample image in: {}'.format(image_path))
                print('save generator model in: {}'.format(generator_path))
                print('save discriminator model in: {}\n'.format(discriminator_path))

            if current_iteration >= num_iterations:
                break


if __name__ == '__main__':
    main()
