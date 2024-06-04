import os
import sys
import random

import hydra
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from utils.task import make
from agent.rsp import RSPAgent
from utils.logger import Logger


def evaluate(device, environment, agent, episodes):
    lengths = []
    returns = []
    for episode in tqdm(range(episodes), desc='Evaluating', leave=False):
        count = 0
        reward = 0
        step = environment.reset()
        while not step.last():
            state = torch.from_numpy(step.observation).to(device)
            action = agent.take_action(state).cpu().numpy()
            step = environment.step(action)
            count += 1
            reward += step.reward
        lengths.append(count)
        returns.append(reward)
    return {'Episode Length': lengths, 'Average Return': returns}


@hydra.main(version_base=None, config_path='./config', config_name='rsp')
def main(config):
    checkpoint_path = str(config.checkpoint)
    if config.device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sys.stdout = Logger(os.path.join(checkpoint_path, 'train.log'))
    config.checkpoint = str(checkpoint_path)
    config.device = str(device)
    print(OmegaConf.to_yaml(config))

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    task = f'walker_{config.setting.task_name}'
    environment = make(task, seed=config.seed)
    agent = RSPAgent(
        config.setting.state_dims,
        config.setting.action_dims
    )

    feedback = evaluate(
        device,
        environment,
        agent,
        config.strategy.sample_episodes
    )
    report = []
    for key, value in feedback.items():
        report.append(f'{key}: {np.mean(value):7.2f}')
    report = ' | '.join(report)
    print(f'Test | Task: {task:11} | {report} |')


if __name__ == '__main__':
    main()
