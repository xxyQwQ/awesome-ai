import glob
import random

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class BCDataset(Dataset):
    def __init__(self, dataset_path):
        self.buffer = {key: [] for key in ['state', 'action']}
        path_list = sorted(glob.glob(f'{dataset_path}/*.npz'))
        for path in path_list:
            with open(path, 'rb') as file:
                record = np.load(file)
                record = {key: record[key] for key in record.keys()}
                state = record['observation'][:-1]
                action = record['action'][1:]
                self.buffer['state'].append(state)
                self.buffer['action'].append(action)
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)

    def __len__(self):
        return self.buffer['state'].shape[0]

    def __getitem__(self, index):
        return [self.buffer[key][index] for key in self.buffer.keys()]


class CSACDataset(Dataset):
    def __init__(self, dataset_path):
        self.buffer = {key: [] for key in ['state', 'action', 'reward', 'next_state', 'done']}
        path_list = sorted(glob.glob(f'{dataset_path}/*.npz'))
        for path in path_list:
            with open(path, 'rb') as file:
                record = np.load(file)
                record = {key: record[key] for key in record.keys()}
                state = record['observation'][:-1]
                action = record['action'][1:]
                reward = record['reward'][1:]
                next_state = record['observation'][1:]
                done = np.zeros_like(reward)
                self.buffer['state'].append(state)
                self.buffer['action'].append(action)
                self.buffer['reward'].append(reward)
                self.buffer['next_state'].append(next_state)
                self.buffer['done'].append(done)
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)

    def __len__(self):
        return self.buffer['state'].shape[0]

    def __getitem__(self, index):
        return [self.buffer[key][index] for key in self.buffer.keys()]


class MTCSACDataset(Dataset):
    def __init__(self, dataset_paths):
        self.buffer = {key: [] for key in ['task', 'state', 'action', 'reward', 'next_state', 'done']}
        for task_index, dataset_path in enumerate(dataset_paths):
            path_list = sorted(glob.glob(f'{dataset_path}/*.npz'))
            for path in path_list:
                with open(path, 'rb') as file:
                    record = np.load(file)
                    record = {key: record[key] for key in record.keys()}
                    task = np.full(record['observation'].shape[0], task_index, dtype=np.int64)
                    state = record['observation'][:-1]
                    action = record['action'][1:]
                    reward = record['reward'][1:]
                    next_state = record['observation'][1:]
                    done = np.zeros_like(reward)
                    self.buffer['task'].append(task)
                    self.buffer['state'].append(state)
                    self.buffer['action'].append(action)
                    self.buffer['reward'].append(reward)
                    self.buffer['next_state'].append(next_state)
                    self.buffer['done'].append(done)
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)

    def __len__(self):
        return self.buffer['state'].shape[0]

    def __getitem__(self, index):
        return [self.buffer[key][index] for key in self.buffer.keys()]


class CDSDataset(Dataset):
    def __init__(self, dataset_paths):
        self.stores = []
        for task_index, dataset_path in enumerate(dataset_paths):
            store = {key: [] for key in ['task', 'state', 'action', 'reward', 'next_state', 'done']}
            path_list = sorted(glob.glob(f'{dataset_path}/*.npz'))
            for path in path_list:
                with open(path, 'rb') as file:
                    record = np.load(file)
                    record = {key: record[key] for key in record.keys()}
                    task = np.full(record['observation'].shape[0], task_index, dtype=np.int64)
                    state = record['observation'][:-1]
                    action = record['action'][1:]
                    reward = record['reward'][1:]
                    next_state = record['observation'][1:]
                    done = np.zeros_like(reward)
                    store['task'].append(task)
                    store['state'].append(state)
                    store['action'].append(action)
                    store['reward'].append(reward)
                    store['next_state'].append(next_state)
                    store['done'].append(done)
            for key in store.keys():
                store[key] = np.concatenate(store[key], axis=0)
            self.stores.append(store)
        self.buffer = {key: [] for key in ['task', 'state', 'action', 'reward', 'next_state', 'done']}
        for store in self.stores:
            for key in store.keys():
                self.buffer[key].append(store[key])
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)

    def __len__(self):
        return self.buffer['state'].shape[0]

    def __getitem__(self, index):
        return [self.buffer[key][index] for key in self.buffer.keys()]

    def update(self, device, agent, percent=0.1):
        self.buffer = {key: [] for key in ['task', 'state', 'action', 'reward', 'next_state', 'done']}
        for store in self.stores:
            for key in store.keys():
                self.buffer[key].append(store[key])
        values = []
        for task_index, task_store in enumerate(self.stores):
            value_list = []
            task = torch.tensor(task_index, dtype=torch.long).to(device)
            for item_index in tqdm(range(task_store['state'].shape[0]), desc='Updating', leave=False):
                state = torch.from_numpy(task_store['state'][item_index]).to(device)
                action = torch.from_numpy(task_store['action'][item_index]).to(device)
                value = agent.compute_value(task, state, action).cpu().numpy()
                value_list.append(value)
            values.append(np.percentile(value_list, 100 * (1 - percent)))
        for source_index in range(len(self.stores)):
            store = self.stores[source_index]
            for target_index in range(len(self.stores)):
                if source_index == target_index:
                    continue
                task = torch.tensor(target_index, dtype=torch.long).to(device)
                for item_index in tqdm(range(store['state'].shape[0]), desc='Updating', leave=False):
                    state = torch.from_numpy(store['state'][item_index]).to(device)
                    action = torch.from_numpy(store['action'][item_index]).to(device)
                    value = agent.compute_value(task, state, action).cpu().numpy()
                    if value > values[target_index]:
                        for key in store.keys():
                            if key == 'task':
                                self.buffer[key].append(np.array([target_index]))
                            else:
                                self.buffer[key].append(store[key][item_index][np.newaxis])
        for key in self.buffer.keys():
            self.buffer[key] = np.concatenate(self.buffer[key], axis=0)
        return {'Critical Value': values}


def bc_collate(batch):
    state, action = zip(*batch)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    return {'state': state, 'action': action}


def csac_collate(batch):
    state, action, reward, next_state, done = zip(*batch)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    reward = np.stack(reward, axis=0)
    next_state = np.stack(next_state, axis=0)
    done = np.stack(done, axis=0)
    return {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}


def mtcsac_collate(batch):
    task, state, action, reward, next_state, done = zip(*batch)
    task = np.stack(task, axis=0)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    reward = np.stack(reward, axis=0)
    next_state = np.stack(next_state, axis=0)
    done = np.stack(done, axis=0)
    return {'task': task, 'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}


def cds_collate(batch):
    task, state, action, reward, next_state, done = zip(*batch)
    task = np.stack(task, axis=0)
    state = np.stack(state, axis=0)
    action = np.stack(action, axis=0)
    reward = np.stack(reward, axis=0)
    next_state = np.stack(next_state, axis=0)
    done = np.stack(done, axis=0)
    return {'task': task, 'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}
