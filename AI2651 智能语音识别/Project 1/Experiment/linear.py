import os
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import utils


def train():
    data = utils.read_label_from_file('./data/train_label.txt')
    feature_list, label_list = [], []
    for name, value in tqdm(data.items()):
        path = os.path.join('./wavs/train', name + '.wav')
        sample_rate, wave = wavfile.read(path)
        feature = utils.wave_feature(wave, sample_rate)
        label = np.pad(np.array(value), (0, feature.shape[0] - len(value)), 'constant', constant_values=(0, 0))
        feature_list.append(feature)
        label_list.append(label)
    feature_list = np.concatenate(feature_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    print('[progress] training dataset has been created')
    print('feature: {}, label: {}'.format(feature_list.shape, label_list.shape))

    classifier = LogisticRegression(penalty='l1', solver='liblinear')
    classifier.fit(feature_list, label_list)
    print('[progress] classifier training has been finished')
    print('weight: {}'.format(classifier.coef_[0]))
    return classifier


def estimate(classifier):
    data = utils.read_label_from_file('./data/dev_label.txt')
    prob_list, pred_list, real_list = [], [], []
    for name, value in tqdm(data.items()):
        path = os.path.join('./wavs/dev', name + '.wav')
        sample_rate, wave = wavfile.read(path)
        feature = utils.wave_feature(wave, sample_rate)
        prob = utils.mean_filtering(classifier.predict_proba(feature)[:, 1])
        pred = utils.generate_prediction(prob)
        real = np.pad(np.array(value), (0, feature.shape[0] - len(value)), 'constant', constant_values=(0, 0))
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


def predict(classifier):
    wave_path = './wavs/test'
    with open('./data/test_label.txt', 'w') as label, open('./result/test_ce.csv', 'w') as form:
        form.write('uttid,label\n')
        for audio in tqdm(os.listdir(wave_path)):
            path = os.path.join(wave_path, audio)
            sample_rate, wave = wavfile.read(path)
            feature = utils.wave_feature(wave, sample_rate)
            probability = utils.mean_filtering(classifier.predict_proba(feature)[:, 1])
            prediction = utils.generate_prediction(probability)
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


def sample(classifier):
    truth = np.array(utils.read_label_from_file('./data/dev_label.txt')['107-22885-0023'])
    sample_rate, wave = wavfile.read('./wavs/dev/107-22885-0023.wav')
    feature = utils.wave_feature(wave, sample_rate)
    probability = utils.mean_filtering(classifier.predict_proba(feature)[:, 1])
    prediction = utils.generate_prediction(probability)
    truth = np.pad(truth, (0, prediction.shape[0] - len(truth)), 'constant', constant_values=(0, 0))
    utils.plot_sample(truth, prediction)


def execute(mode='sample'):
    if mode == 'train':
        classifier = train()
        with open('./model/linear.pkl', 'wb') as file:
            pickle.dump(classifier, file)
        return
    assert os.path.exists('./model/linear.pkl')
    with open('./model/linear.pkl', 'rb') as file:
        classifier = pickle.load(file)
    if mode == 'estimate':
        estimate(classifier)
    elif mode == 'predict':
        predict(classifier)
    elif mode == 'sample':
        sample(classifier)


if __name__ == '__main__':
    execute(mode='sample')
