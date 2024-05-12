import os
from pathlib import Path

import fire
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
from tabulate import tabulate
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

from model.crnn import CRNN
from utils import dataset, augment, function, metrics


DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)


class Runner(object):
    def __init__(self, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def forward(model, batch):
        aid, feat, target = batch
        feat = feat.to(DEVICE).float()
        target = target.to(DEVICE).float()
        output = model(feat)
        output['aid'] = aid
        output['target'] = target
        return output

    def train(self, cfg, **kwargs):
        config = function.parse_config_or_kwargs(cfg, **kwargs)
        ckpt = config['training']['save_path']
        Path(ckpt).mkdir(exist_ok=True, parents=True)
        function.dump_config(os.path.join(ckpt, 'config.yaml'), config)

        logger = function.create_logger(os.path.join(ckpt, 'train.log'))
        logger.info('Storing files in {}'.format(ckpt))
        function.pprint_dict(config, logger.info)
        logger.info('Running on device {}'.format(DEVICE))

        label_to_idx = {}
        with open(config['dataset']['reference'], 'r') as reader:
            for line in reader.readlines():
                idx, label = line.strip().split(',')
                label_to_idx[label] = int(idx)
        labels_df = pd.read_csv(config['dataset']['dev']['label'], sep='\s+').convert_dtypes()
        label_array = labels_df['event_labels'].apply(
            lambda x: function.encode_label(x, label_to_idx)
        )
        label_array = np.stack(label_array.values)
        train_df, cv_df = function.split_train_cv(
            labels_df,
            y=label_array,
            stratified=config['dataset']['stratified']
        )

        trainloader = DataLoader(
            dataset.TrainingDataset(
                config['dataset']['dev']['feature'],
                train_df,
                label_to_idx,
                transform=augment.parse_transform(config['dataset']['augmentation'])
            ),
            num_workers=config['training']['num_workers'],
            shuffle=True,
            batch_size=config['training']['batch_size'],
            collate_fn=dataset.sequential_collate(False)
        )

        cvdataloader = DataLoader(
            dataset.TrainingDataset(
                config['dataset']['dev']['feature'],
                cv_df,
                label_to_idx
            ),
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=dataset.sequential_collate(return_length=False)
        )

        loss_fn = function.BinaryCrossEntropyLoss().to(DEVICE)

        if config['model']['name'] == 'CRNN':
            model = CRNN(
                num_bins=trainloader.dataset.datadim,
                num_classes=len(label_to_idx),
                **config['model']['argument']
            )
        model = model.to(DEVICE)
        function.pprint_dict(model, logger.info, formatter='pretty')

        optimizer = getattr(optim, config['optimizer']['name'])(
            model.parameters(),
            **config['optimizer']['argument']
        )
        function.pprint_dict(optimizer, logger.info, formatter='pretty')

        scheduler = getattr(optim.lr_scheduler, config['scheduler']['name'])(
            optimizer,
            **config['scheduler']['argument']
        )
        
        not_improve_cnt = 0
        best_loss = float('inf')

        for epoch in range(1, config['training']['num_epochs'] + 1):
            model.train()
            loss_history = []
            with torch.enable_grad():
                for batch in tqdm(trainloader, unit='batch', leave=False):
                    optimizer.zero_grad()
                    output = self.forward(model, batch)
                    loss = loss_fn(output)
                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())
            train_loss = np.mean(loss_history)

            model.eval()
            loss_history = []
            preds = []
            targets = []
            with torch.no_grad():
                for batch in tqdm(cvdataloader, unit='batch', leave=False):
                    output = self.forward(model, batch)
                    loss = loss_fn(output)
                    loss_history.append(loss.item())
                    y_pred = output['clip_prob']
                    y_pred = torch.round(y_pred)
                    preds.append(y_pred.cpu().numpy())
                    targets.append(output['target'].cpu().numpy())

            val_loss = np.mean(loss_history)
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            p, r, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
            logger.info('| Epoch: {:>3d} | Train Loss: {:.2f} | Valid Loss: {:.2f} '
                        '| Precision: {:.2f} | Recall: {:.2f} | F1 Score: {:.2f} |'.format(
                            epoch, train_loss, val_loss, p, r, f1))

            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                not_improve_cnt = 0
                torch.save(model.state_dict(), os.path.join(ckpt, 'model.pth'))
            else:
                not_improve_cnt += 1

            if not_improve_cnt == config['training']['early_stop']:
                break

        return ckpt

    def evaluate(self, ckpt):
        config = function.parse_config_or_kwargs(os.path.join(ckpt, 'config.yaml'))
        feature = config['dataset']['eval']['feature']
        label = config['dataset']['eval']['label']
        state_dict = torch.load(os.path.join(ckpt, 'model.pth'), map_location='cpu')
        label_df = pd.read_csv(label, sep='\t')

        label_to_idx = {}
        idx_to_label = {}
        with open(config['dataset']['reference'], 'r') as reader:
            for line in reader.readlines():
                idx, label = line.strip().split(',')
                label_to_idx[label] = int(idx)
                idx_to_label[int(idx)] = label

        dataloader = DataLoader(
            dataset.InferenceDataset(feature),
            batch_size=1,
            shuffle=False,
            num_workers=0
        )

        if config['model']['name'] == 'CRNN':
            model = CRNN(
                num_bins=dataloader.dataset.datadim,
                num_classes=len(label_to_idx),
                **config['model']['argument']
            )
        model.load_state_dict(state_dict)
        model = model.to(DEVICE)
        model.eval()

        clip_targets = []
        clip_probs = []
        clip_preds = []
        frame_preds = []

        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False):
                aids, feats = batch
                feats = feats.to(DEVICE).float()
                output = model(feats)
                frame_prob_batch = output['frame_prob'].cpu().numpy()
                clip_prob_batch = output['clip_prob'].cpu().numpy()
                filtered_pred = function.median_filter(
                    frame_prob_batch,
                    window_size=config['evaluation']['window_size'],
                    threshold=config['evaluation']['binary_threshold']
                )
                frame_pred_batch = function.decode_with_timestamps(idx_to_label, filtered_pred)

                for sample_idx in range(len(frame_pred_batch)):
                    aid = aids[sample_idx]

                    # clip results for mAP
                    clip_probs.append(clip_prob_batch[sample_idx])
                    clip_target = label_df.loc[label_df['filename'] == aid]['event_label'].unique()
                    clip_targets.append(function.encode_label(clip_target, label_to_idx))
                    
                    # clip results after postprocessing
                    clip_pred = clip_prob_batch[sample_idx].reshape(1, -1)
                    clip_pred = function.binarize(clip_pred)[0]
                    clip_pred = [idx_to_label[i] for i, tgt in enumerate(clip_pred) if tgt == 1]
                    for clip_label in clip_pred:
                        clip_preds.append({
                            'filename': aid,
                            'event_label': clip_label,
                            'probability': clip_prob_batch[sample_idx][label_to_idx[clip_label]]
                        })

                    # frame results after postprocessing
                    frame_pred = frame_pred_batch[sample_idx]
                    for event_label, onset, offset in frame_pred:
                        frame_preds.append({
                            'filename': aid,
                            'event_label': event_label,
                            'onset': onset,
                            'offset': offset
                        })

        assert len(clip_preds) > 0 and len(frame_preds) > 0
        frame_pred_df = pd.DataFrame(
            frame_preds,
            columns=['filename', 'event_label', 'onset', 'offset']
        )
        clip_pred_df = pd.DataFrame(
            clip_preds,
            columns=['filename', 'event_label', 'probability']
        )
        frame_pred_df = function.predictions_to_time(
            frame_pred_df,
            ratio=config['evaluation']['time_ratio']
        )
        frame_pred_df.to_csv(
            os.path.join(ckpt, config['evaluation']['prediction']),
            sep='\t',
            float_format='%.3f',
            index=False
        )
        tagging_df = metrics.audio_tagging_results(
            label_df,
            clip_pred_df,
            label_to_idx
        )

        clip_targets = np.stack(clip_targets)
        clip_probs = np.stack(clip_probs)
        average_precision = average_precision_score(
            np.array(clip_targets),
            np.array(clip_probs),
            average=None
        )
        print('mAP: {}'.format(average_precision))

        tagging_df.to_csv(
            os.path.join(ckpt, config['evaluation']['tagging']),
            sep='\t',
            float_format='%.3f',
            index=False
        )

        event_result, segment_result = metrics.compute_metrics(
            label_df,
            frame_pred_df,
            time_resolution=1.0
        )

        with open(os.path.join(ckpt, config['evaluation']['event']), 'w') as wp:
            wp.write(event_result.__str__())
        with open(os.path.join(ckpt, config['evaluation']['segment']), 'w') as wp:
            wp.write(segment_result.__str__())
        event_based_results = pd.DataFrame(
            event_result.results_class_wise_average_metrics()['f_measure'],
            index=['event_based']
        )
        segment_based_results = pd.DataFrame(
            segment_result.results_class_wise_average_metrics()['f_measure'],
            index=['segment_based']
        )
        result_quick_report = pd.concat((
            event_based_results,
            segment_based_results,
        ))

        tagging_res = tagging_df.loc[tagging_df['label'] == 'macro'].values[0][1:]
        result_quick_report.loc['tagging_based'] = list(tagging_res)

        with open(os.path.join(ckpt, 'report.md'), 'w') as wp:
            print(tabulate(result_quick_report, headers='keys', tablefmt='github'), file=wp)
            print('mAP: {}'.format(np.mean(average_precision)), file=wp)

        print('Quick Report: \n{}'.format(tabulate(
            result_quick_report,
            headers='keys',
            tablefmt='github'))
        )

    def mainloop(self, cfg, **kwargs):
        ckpt = self.train(cfg, **kwargs)
        self.evaluate(ckpt)


if __name__ == '__main__':
    fire.Fire(Runner)
