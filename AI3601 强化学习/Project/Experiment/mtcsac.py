import os
import sys
import random

import hydra
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.task import make
from utils.logger import Logger
from agent.mtcsac import MTCSACAgent
from utils.torch import convert_batch_to_tensor
from utils.dataset import MTCSACDataset, mtcsac_collate


def evaluate(device, task, environment, agent, episodes):
    lengths = []
    returns = []
    task = torch.tensor([task], dtype=torch.long).to(device)
    for episode in tqdm(range(episodes), desc='Evaluating', leave=False):
        count = 0
        reward = 0
        step = environment.reset()
        while not step.last():
            state = torch.from_numpy(step.observation).to(device)
            action = agent.take_action(task, state).cpu().numpy()
            step = environment.step(action)
            count += 1
            reward += step.reward
        lengths.append(count)
        returns.append(reward)
    return {'Episode Length': lengths, 'Average Return': returns}


@hydra.main(version_base=None, config_path='./config', config_name='mtcsac')
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

    tasks = ['walker_walk', 'walker_run']
    environments = [
        make('walker_walk', seed=config.seed),
        make('walker_run', seed=config.seed)
    ]
    dataset = MTCSACDataset([
        f'dataset/walker_walk-td3-{config.setting.dataset_name}/data',
        f'dataset/walker_run-td3-{config.setting.dataset_name}/data'
    ])
    loader = DataLoader(
        dataset,
        batch_size=config.strategy.batch_size,
        shuffle=True,
        collate_fn=mtcsac_collate
    )
    agent = MTCSACAgent(
        config.setting.num_tasks,
        config.setting.state_dims,
        config.setting.action_dims,
        **dict(config.parameter)
    )
    agent.to_device(device)

    best = 0
    for epoch in range(1, config.strategy.num_epochs + 1):
        record = {}
        for batch in tqdm(loader, desc='Training', leave=False):
            batch = convert_batch_to_tensor(batch, device)
            feedback = agent.train_batch(
                batch,
                warmup=epoch < config.strategy.warmup_epochs
            )
            for key, value in feedback.items():
                if key not in record:
                    record[key] = []
                record[key].append(value)
        report = []
        for key, value in record.items():
            report.append(f'{key}: {np.mean(value):7.2f}')
        report = ' | '.join(report) 
        print(f'Train | Epoch: {epoch:4d} | {report} |')

        if epoch % config.strategy.eval_interval == 0:
            result = [0, 0]
            for task, environment in enumerate(environments):
                feedback = evaluate(
                    device,
                    task,
                    environment,
                    agent,
                    config.strategy.sample_episodes
                )
                result[task] = np.mean(feedback['Average Return'])
                report = []
                for key, value in feedback.items():
                    report.append(f'{key}: {np.mean(value):7.2f}')
                report = ' | '.join(report)
                print(f'Evaluate | Epoch: {epoch:4d} | Task: {tasks[task]:11} | {report} |')
            if np.mean(result) > best:
                best = np.mean(result)
                agent.save_model(os.path.join(checkpoint_path, 'model.pth'))

    agent.load_model(os.path.join(checkpoint_path, 'model.pth'))
    for task, environment in enumerate(environments):
        feedback = evaluate(
            device,
            task,
            environment,
            agent,
            config.strategy.sample_episodes
        )
        report = []
        for key, value in feedback.items():
            report.append(f'{key}: {np.mean(value):7.2f}')
        report = ' | '.join(report)
        print(f'Test | Task: {tasks[task]:11} | {report} |')


if __name__ == '__main__':
    main()
