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

from facial.arcface import Backbone
from utils.logger import Logger
from utils.dataset import FaceDataset
from utils.function import hinge_loss, plot_sample
from model.generator import InjectiveGenerator
from model.discriminator import MultiscaleDiscriminator


@hydra.main(version_base=None, config_path='./config', config_name='training')
def main(config):
    # load configuration
    dataset_path = str(config.parameter.dataset_path)
    checkpoint_path = str(config.parameter.checkpoint_path)
    model_path = str(config.parameter.model_path) if config.parameter.finetune else None
    device = torch.device('cuda') if config.parameter.device == 'gpu' else torch.device('cpu')
    batch_size = int(config.parameter.batch_size)
    num_workers = int(config.parameter.num_workers)
    learning_rate = float(config.parameter.learning_rate)
    weight_adversarial = float(config.parameter.loss_function.weight_adversarial)
    weight_attribute = float(config.parameter.loss_function.weight_attribute)
    weight_identity = float(config.parameter.loss_function.weight_identity)
    weight_reconstruction = float(config.parameter.loss_function.weight_reconstruction)
    num_iterations = int(config.parameter.num_iterations)
    report_interval = int(config.parameter.report_interval)
    save_interval = int(config.parameter.save_interval)

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'training.log'))
    config.parameter.checkpoint_path = checkpoint_path
    config.parameter.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    identity_model = Backbone(50, 0.6, 'ir_se').to(device)
    identity_model.eval()
    identity_model.load_state_dict(torch.load('./facial/arcface.pth', map_location=device), strict=False)

    generator_model = InjectiveGenerator().to(device)
    generator_model.train()

    discriminator_model = MultiscaleDiscriminator().to(device)
    discriminator_model.train()

    if model_path is not None:
        generator_model.load_state_dict(torch.load(os.path.join(model_path, 'generator.pth'), map_location=device), strict=False)
        discriminator_model.load_state_dict(torch.load(os.path.join(model_path, 'discriminator.pth'), map_location=device), strict=False)

    # create optimizer
    generator_optimizer = Adam(generator_model.parameters(), lr=learning_rate, betas=(0, 0.999), weight_decay=1e-4)
    discriminator_optimizer = Adam(discriminator_model.parameters(), lr=learning_rate, betas=(0, 0.999), weight_decay=1e-4)

    # load dataset
    dataset = FaceDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    print('image number: {}\n'.format(len(dataset)))

    # start training
    current_iteration = 0
    current_time = time.time()

    while current_iteration < num_iterations:
        for source_image, target_image, same_identity in dataloader:
            current_iteration += 1

            source_image, target_image, same_identity = source_image.to(device), target_image.to(device), same_identity.to(device)

            with torch.no_grad():
                source_identity = identity_model(F.interpolate(source_image, 112, mode='bilinear', align_corners=True))

            # generator
            generator_optimizer.zero_grad()

            result_image, target_attribute = generator_model(target_image, source_identity)

            prediction_result = discriminator_model(result_image)
            loss_adversarial = 0
            for prediction in prediction_result:
                loss_adversarial += hinge_loss(prediction, positive=True)

            result_identity = identity_model(F.interpolate(result_image, 112, mode='bilinear', align_corners=True))
            loss_identity = (1 - torch.cosine_similarity(source_identity, result_identity, dim=1)).mean()

            result_attribute = generator_model.attribute(result_image)
            loss_attribute = 0
            for i in range(len(target_attribute)):
                loss_attribute += 0.5 * torch.mean(torch.square(target_attribute[i] - result_attribute[i]))

            loss_reconstruction = torch.sum(same_identity * 0.5 * torch.mean(torch.square(result_image - target_image).reshape(batch_size, -1), dim=1)) / (torch.sum(same_identity) + 1e-6)

            loss_generator = weight_adversarial * loss_adversarial + weight_identity * loss_identity + weight_attribute * loss_attribute + weight_reconstruction * loss_reconstruction
            loss_generator.backward()
            generator_optimizer.step()

            # discriminator
            discriminator_optimizer.zero_grad()

            prediction_fake = discriminator_model(result_image.detach())
            loss_fake = 0
            for prediction in prediction_fake:
                loss_fake += hinge_loss(prediction, positive=False)

            prediction_real = discriminator_model(source_image)
            loss_true = 0
            for prediction in prediction_real:
                loss_true += hinge_loss(prediction, positive=True)

            loss_discriminator = 0.5 * (loss_true + loss_fake)
            loss_discriminator.backward()
            discriminator_optimizer.step()

            # report
            if current_iteration % report_interval == 0:
                last_time = current_time
                current_time = time.time()
                iteration_time = (current_time - last_time) / report_interval

                print('iteration {} / {}:'.format(current_iteration, num_iterations))
                print('time: {:.6f} seconds per iteration'.format(iteration_time))
                print('discriminator loss: {:.6f}, generator loss: {:.6f}'.format(loss_discriminator.item(), loss_generator.item()))
                print('adversarial loss: {:.6f}, identity loss: {:.6f}, attribute loss: {:.6f}, reconstruction loss: {:.6f}\n'.format(loss_adversarial.item(), loss_identity.item(), loss_attribute.item(), loss_reconstruction.item()))

            # save
            if current_iteration % save_interval == 0:
                save_path = os.path.join(checkpoint_path, 'iteration_{}'.format(current_iteration))
                os.makedirs(save_path, exist_ok=True)

                image_path = os.path.join(save_path, 'image.jpg')
                generator_path = os.path.join(save_path, 'generator.pth')
                discriminator_path = os.path.join(save_path, 'discriminator.pth')

                image = plot_sample(source_image, target_image, result_image).transpose([1, 2, 0])
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
