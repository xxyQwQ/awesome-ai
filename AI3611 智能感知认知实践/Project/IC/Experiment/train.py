import os
import random
from pathlib import Path

import fire
import json
import yaml
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from model.captioner import Captioner
from utils.metrics import bleu_score_fn
from utils.dataset import Flickr8kDataset
from utils.function import get_logger, ptb_tokenize, words_from_tensors_fn, sched_sampling_eps_fn


class Runner(object): 
    def __init__(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.device = torch.device(device)

    def create_loader(self, dataset_base_path, batch_size, num_workers):
        train_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path,
            return_type='tensor',
            dist='train'
        )
        vocab_set = train_set.get_vocab()
        train_eval_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path,
            return_type='corpus',
            dist='train',
            vocab_set=vocab_set,
        )
        val_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path,
            return_type='corpus',
            dist='val',
            vocab_set=vocab_set
        )
        test_set = Flickr8kDataset(
            dataset_base_path=dataset_base_path,
            return_type='corpus',
            dist='test',
            vocab_set=vocab_set,
        )
        train_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        eval_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_set.transformations = train_transformations
        val_set.transformations = eval_transformations
        test_set.transformations = eval_transformations
        train_eval_set.transformations = eval_transformations
        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch],
            [x[2] for x in batch],
            [x[3] for x in batch]
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        train_eval_loader = DataLoader(
            train_eval_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=eval_collate_fn
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=eval_collate_fn
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=eval_collate_fn
        )
        loader_dict = {
            'train': train_loader,
            'train_eval': train_eval_loader,
            'val': val_loader,
            'test': test_loader
        }
        return loader_dict

    def train_model(self, loader, model, criterion, optimizer, prompt, epsilon=1.0):
        running_acc = 0
        running_loss = 0
        model.train()
        bar = tqdm(iter(loader), desc=f'{prompt}', leave=False)
        for batch_idx, batch in enumerate(bar):
            images, captions, lengths = batch
            images = images.to(self.device)
            captions = captions.to(self.device)
            optimizer.zero_grad()
            scores, caps_sorted, decode_lengths, _, _ = model(
                source_image=images,
                caption_token=captions,
                caption_length=lengths,
                sample_epsilon=epsilon
            )
            targets = caps_sorted[:, 1:]
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            correct = (torch.argmax(scores, dim=1) == targets).sum().float().item()
            running_acc += correct / len(targets)
            running_loss += loss.item()
            bar.set_postfix(ordered_dict={
                'loss': running_loss / (batch_idx + 1),
                'acc': running_acc / (batch_idx + 1)
            }, refresh=True)
        bar.close()
        return running_loss / len(loader)

    def evaluate_model(self, loader, model, prompt, word2idx, sample_method,
        beam_size, tensor_to_word_fn, bleu_score_fn, return_output=False):
        bleu = [0] * 5
        references = []
        predictions = []
        imgids = []
        model.eval()
        bar = tqdm(iter(loader), desc=f'{prompt}', leave=False)
        for batch_idx, batch in enumerate(bar):
            images, captions, _, imgid_batch = batch
            images = images.to(self.device)
            outputs = tensor_to_word_fn(model.sample(
                source_image=images,
                start_index=word2idx['<start>'],
                sample_method=sample_method,
                beam_size=beam_size
            ).cpu().numpy())
            references.extend(captions)
            predictions.extend(outputs)
            imgids.extend(imgid_batch)
            bar.set_postfix(ordered_dict={'batch': batch_idx}, refresh=True)
        bar.close()
        for i in (1, 2, 3, 4):
            bleu[i] = bleu_score_fn(
                reference_corpus=references,
                candidate_corpus=predictions,
                n=i
            )
        references = [[' '.join(cap) for cap in caption] for caption in references]
        predictions = [' '.join(caption) for caption in predictions]
        return (bleu, references, predictions, imgids) if return_output else bleu

    def train(self, cfg, **kwargs):
        with open(cfg) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)
        Path(args['checkpoint_path']).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args['checkpoint_path'], 'config.yaml'), 'w') as f:
            yaml.dump(args, f, default_flow_style=False)
        logger = get_logger(os.path.join(args['checkpoint_path'], 'train.log'))

        dataloaders = self.create_loader(
            args['dataset_path'],
            args['training']['batch_size'],
            args['training']['num_workers']
        )
        vocab_set = dataloaders['train'].dataset.get_vocab()
        with open(args['vocabulary_path'], 'wb') as f:
            pickle.dump(vocab_set, f)
        vocab, word2idx, idx2word, _ = vocab_set
        vocab_size = len(vocab)
        if args['model']['name'] == 'Captioner':
            model = Captioner(
                vocabulary_size=vocab_size,
                feature_size=args['model']['feature_size'],
                embedding_dims=args['model']['embedding_dims'],
                encoder_dims=args['model']['encoder_dims'],
                decoder_dims=args['model']['decoder_dims'],
                hidden_dims=args['model']['hidden_dims'],
                dropout_rate=args['model']['dropout_rate']
            ).to(self.device)
        logger.info(model)
        pad_value = dataloaders['train'].dataset.pad_value
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_value).to(self.device)
        tensor_to_word_fn = words_from_tensors_fn(idx2word)
        corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')
        optimizer = getattr(optim, args['optimizer']['name'])(
            model.parameters(),
            lr=args['optimizer']['learning_rate']
        )

        val_bleu4_max = 0
        num_epochs = args['training']['num_epochs']
        sched_sampling_eps = sched_sampling_eps_fn(args['training']['sample_schedule'])
        for epoch in range(num_epochs):
            train_loss = self.train_model(
                loader=dataloaders['train'],
                model=model,
                criterion=loss_fn,
                optimizer=optimizer,
                prompt=f'Epoch {epoch + 1}/{num_epochs}',
                epsilon=sched_sampling_eps(epoch)
            )
            with torch.no_grad():
                val_bleu = self.evaluate_model(
                    loader=dataloaders['val'],
                    model=model,
                    prompt='Val eval: ',
                    word2idx=word2idx,
                    sample_method=args['model']['sample_method'],
                    beam_size=args['model']['beam_size'],
                    tensor_to_word_fn=tensor_to_word_fn,
                    bleu_score_fn=corpus_bleu_score_fn
                )
                logger.info(f'| Epoch: {epoch + 1:>3d} | Train Loss: {train_loss:.3f} ' \
                    f'| Val Bleu-1: {val_bleu[1]:.3f} | Val Bleu-4: {val_bleu[4]:.3f} |')
                state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_bleus': val_bleu
                }
                if val_bleu[4] > val_bleu4_max:
                    val_bleu4_max = val_bleu[4]
                    torch.save(state, os.path.join(args['checkpoint_path'], 'model.pth'))

        model.load_state_dict(torch.load(
            os.path.join(args['checkpoint_path'], 'model.pth'),
            map_location='cpu'
        )['state_dict'])
        with torch.no_grad():
            model.eval()
            train_bleu = self.evaluate_model(
                loader=dataloaders['train_eval'],
                model=model,
                prompt='Train: ',
                word2idx=word2idx,
                sample_method=args['model']['sample_method'],
                beam_size=args['model']['beam_size'],
                tensor_to_word_fn=tensor_to_word_fn,
                bleu_score_fn=corpus_bleu_score_fn
            )
            val_bleu = self.evaluate_model(
                loader=dataloaders['val'],
                model=model,
                prompt='Val: ',
                word2idx=word2idx,
                sample_method=args['model']['sample_method'],
                beam_size=args['model']['beam_size'],
                tensor_to_word_fn=tensor_to_word_fn,
                bleu_score_fn=corpus_bleu_score_fn
            )
            test_bleu = self.evaluate_model(
                loader=dataloaders['test'],
                model=model,
                prompt='Test: ',
                word2idx=word2idx,
                sample_method=args['model']['sample_method'],
                beam_size=args['model']['beam_size'],
                tensor_to_word_fn=tensor_to_word_fn,
                bleu_score_fn=corpus_bleu_score_fn
            )
            logger.info('Evaluation of the best validation performance model:')
            for setname, result in zip(
                ('Train', 'Val', 'Test'),
                (train_bleu, val_bleu, test_bleu)
            ):
                logger.info(setname, end='\t')
                for ngram in (1, 2, 3, 4):
                    logger.info(f'Bleu-{ngram}: {result[ngram]:.3f}')

    def evaluate(self, cfg, **kwargs):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

        with open(cfg) as reader:
            config = yaml.load(reader, Loader=yaml.FullLoader)
        args = dict(config, **kwargs)

        with open(args['vocabulary_path'], 'rb') as f:
            vocab_set = pickle.load(f)
        vocab, word2idx, idx2word, _ = vocab_set
        vocab_size = len(vocab)
        test_set = Flickr8kDataset(
            dataset_base_path=args['dataset_path'],
            return_type='corpus',
            dist='test',
            vocab_set=vocab_set
        )
        eval_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_set.transformations = eval_transformations
        eval_collate_fn = lambda batch: (
            torch.stack([x[0] for x in batch]),
            [x[1] for x in batch],
            [x[2] for x in batch],
            [x[3] for x in batch]
        )
        test_loader = DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            collate_fn=eval_collate_fn
        )
        if args['model']['name'] == 'Captioner':
            model = Captioner(
                vocabulary_size=vocab_size,
                feature_size=args['model']['feature_size'],
                embedding_dims=args['model']['embedding_dims'],
                encoder_dims=args['model']['encoder_dims'],
                decoder_dims=args['model']['decoder_dims'],
                hidden_dims=args['model']['hidden_dims'],
                dropout_rate=args['model']['dropout_rate']
            ).to(self.device)
        model.load_state_dict(torch.load(
            os.path.join(args['checkpoint_path'], 'model.pth'),
            map_location='cpu'
        )['state_dict'])
        model = model.to(self.device)
        tensor_to_word_fn = words_from_tensors_fn(idx2word=idx2word)
        corpus_bleu_score_fn = bleu_score_fn(4, 'corpus')

        with torch.no_grad():
            model.eval()
            _, references, predictions, imgids = self.evaluate_model(
                loader=test_loader,
                model=model,
                prompt='Test: ',
                word2idx=word2idx,
                sample_method=args['model']['sample_method'],
                beam_size=args['model']['beam_size'],
                tensor_to_word_fn=tensor_to_word_fn,
                bleu_score_fn=corpus_bleu_score_fn,
                return_output=True
            )
            key_to_pred = {}
            key_to_refs = {}
            output_pred = []
            for imgid, pred, refs in zip(imgids, predictions, references):
                key_to_pred[imgid] = [pred]
                key_to_refs[imgid] = refs
                output_pred.append({'img_id': imgid, 'prediction': [pred]})
            key_to_refs = ptb_tokenize(key_to_refs)
            key_to_pred = ptb_tokenize(key_to_pred)
            scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
            output = {'SPIDEr': 0}
            with open(os.path.join(args['checkpoint_path'], 'score.txt'), 'w') as writer:
                for scorer in scorers:
                    score, _ = scorer.compute_score(key_to_refs, key_to_pred)
                    method = scorer.method()
                    output[method] = score
                    if method == 'Bleu':
                        for n in range(4):
                            print(f'Bleu-{n + 1}: {score[n]:.3f}', file=writer)
                    else:
                        print(f'{method}: {score:.3f}', file=writer)
                    if method in ['CIDEr', 'SPICE']:
                        output['SPIDEr'] += score
                output['SPIDEr'] /= 2
                print(f'SPIDEr: {output["SPIDEr"]:.3f}', file=writer)
            with open(os.path.join(args['checkpoint_path'], 'prediction.json'), 'w') as writer:
                json.dump(output_pred, writer, indent=4)

    def mainloop(self, cfg, **kwargs):
        self.train(cfg, **kwargs)
        self.evaluate(cfg, **kwargs)


if __name__ == '__main__':
    fire.Fire(Runner)
