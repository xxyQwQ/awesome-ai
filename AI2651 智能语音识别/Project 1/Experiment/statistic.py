import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.compliance import kaldi
import utils


class VAD_Dataset(Dataset):
    def __init__(self, mode='test'):
        save_path = os.path.join('./data', mode + '_data.pt')
        if os.path.exists(save_path):
            self.feature, self.label = torch.load(save_path)
        else:
            data_path = os.path.join('./data', mode + '_label.txt')
            wave_path = os.path.join('./wavs', mode)
            data = utils.read_label_from_file(data_path)
            feature_list, label_list = [], []
            for name, value in tqdm(data.items()):
                path = os.path.join(wave_path, name + '.wav')
                wave, rate = torchaudio.load(path)
                feature = kaldi.fbank(wave, sample_frequency=rate, num_mel_bins=40, snip_edges=False)
                label = F.pad(torch.Tensor(value), (0, feature.shape[0] - len(value)), 'constant', 0)
                feature_list.append(feature)
                label_list.append(label)
            self.feature = torch.cat(feature_list, dim=0)
            self.label = torch.cat(label_list, dim=0)
            torch.save((self.feature, self.label), save_path)

    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    def __len__(self):
        return self.feature.shape[0]


class VAD_Model(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, output_dim=1):
        super(VAD_Model, self).__init__()
        linear_layer = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ]
        norm_layer = [
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ]
        for layer in linear_layer:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.network = nn.Sequential(
            linear_layer[0],
            norm_layer[0],
            nn.ReLU(True),
            linear_layer[1],
            norm_layer[1],
            nn.ReLU(True),
            linear_layer[2],
            norm_layer[2],
            nn.ReLU(True),
            linear_layer[3],
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.network(inputs)
        return outputs


def train(device, total_epoch=1):
    dataset = VAD_Dataset('train')
    loader = DataLoader(dataset, batch_size=1024, drop_last=False)
    print('[progress] training dataset has been created')
    print('size: {}'.format(len(dataset)))

    model = VAD_Model(input_dim=40, hidden_dim=256, output_dim=1).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), 1e-3)
    for epoch in range(total_epoch):
        print('[progress] epoch {} has been started'.format(epoch + 1))
        epoch_loss = 0
        with tqdm(loader) as pbar:
            for feature, label in pbar:
                feature, label = feature.to(device), label.to(device)
                result = model(feature).squeeze(-1)
                loss = criterion(result, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        print('epoch: {}, loss: {}'.format(epoch + 1, epoch_loss))
    print('[progress] model training has been finished')
    return model


def estimate(device, model):
    dataset = VAD_Dataset('dev')
    loader = DataLoader(dataset, batch_size=1024, drop_last=False)
    print('[progress] verifying dataset has been created')
    print('size: {}'.format(len(dataset)))

    prob_list, pred_list, real_list = [], [], []
    with torch.no_grad():
        for feature, label in tqdm(loader):
            feature, label = feature.to(device), label.to(device)
            result = model(feature).squeeze(-1)
            prob = utils.mean_filtering(result.cpu().numpy(), width=12)
            pred = utils.generate_prediction(prob, holder=3)
            real = label.cpu().numpy()
            prob_list.append(prob)
            pred_list.append(pred)
            real_list.append(real)
    prob_list = np.concatenate(prob_list, axis=0)
    pred_list = np.concatenate(pred_list, axis=0)
    real_list = np.concatenate(real_list, axis=0)
    print('[progress] prediction has been generated')
    print('probability: {}, prediction: {}, reality: {}'.format(prob_list.shape, pred_list.shape, real_list.shape))

    acc, auc, err = utils.compute_estimation(prob_list, pred_list, real_list)
    print('[progress] estimation has been finished')
    print('acc: {}, auc: {}, err: {}'.format(acc, auc, err))


def predict(device, model):
    wave_path = './wavs/test'
    with open('./data/test_label.txt', 'w') as label, open('./result/test_ce.csv', 'w') as form:
        form.write('uttid,label\n')
        for audio in tqdm(os.listdir(wave_path)):
            path = os.path.join(wave_path, audio)
            wave, rate = torchaudio.load(path)
            feature = kaldi.fbank(wave, sample_frequency=rate, num_mel_bins=40, snip_edges=False).to(device)
            with torch.no_grad():
                result = model(feature).squeeze(-1)
            probability = utils.mean_filtering(result.cpu().numpy(), width=12)
            prediction = utils.generate_prediction(probability, holder=3)
            name, _ = os.path.splitext(audio)
            label.write('{} {}\n'.format(name, utils.prediction_to_vad_label(prediction)))
            for frame in range(len(probability)):
                form.write('{}-{},{:.2f}\n'.format(name, frame, probability[frame]))
    included = pd.read_csv('./result/test_ce.csv')
    required = pd.read_csv('./result/sample_ce.csv')['uttid'].tolist()
    print('[progress] prediction has been generated')
    print('included: {}, required: {}'.format(len(included), len(required)))

    result = included[included['uttid'].isin(required)]
    result.to_csv('./result/test_ce.csv', index=None)
    print('[progress] result has been standardized')
    print('included: {}, required: {}'.format(len(result), len(required)))


def sample(device, model):
    truth = np.array(utils.read_label_from_file('./data/dev_label.txt')['107-22885-0023'])
    wave, rate = torchaudio.load('./wavs/dev/107-22885-0023.wav')
    feature = kaldi.fbank(wave, sample_frequency=rate, num_mel_bins=40, snip_edges=False).to(device)
    with torch.no_grad():
        result = model(feature).squeeze(-1)
    probability = utils.mean_filtering(result.cpu().numpy(), width=12)
    prediction = utils.generate_prediction(probability, holder=3)
    truth = np.pad(truth, (0, prediction.shape[0] - len(truth)), 'constant', constant_values=(0, 0))
    utils.plot_sample(truth, prediction)


def execute(mode='sample'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if mode == 'train':
        model = train(device, total_epoch=30)
        torch.save(model, './model/statistic.pth')
        return
    assert os.path.exists('./model/statistic.pth')
    model = torch.load('./model/statistic.pth').to(device)
    if mode == 'estimate':
        estimate(device, model)
    elif mode == 'predict':
        predict(device, model)
    elif mode == 'sample':
        sample(device, model)


if __name__ == '__main__':
    execute(mode='sample')
