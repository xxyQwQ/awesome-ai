import os
import io
import glob

import nltk
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from utils.function import split_data


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        dataset_base_path='./dataset/flickr8k/',
        return_type='tensor',
        dist='val',
        startseq='<start>',
        endseq='<end>',
        unkseq='<unk>',
        padseq='<pad>',
        pad_value=0,
        vocab_set=None,
        transformations=None,
        load_img_to_memory=False,
    ):
        self.token = os.path.join(dataset_base_path, 'caption.txt')
        self.images_path = os.path.join(dataset_base_path, 'image')
        self.dist_list = {
            'train': os.path.join(dataset_base_path, 'train_imgs.txt'),
            'val': os.path.join(dataset_base_path, 'val_imgs.txt'),
            'test': os.path.join(dataset_base_path, 'test_imgs.txt'),
        }
        self.return_type = return_type
        if return_type == 'corpus':
            self.__get_item__fn = self.__getitem__corpus
        elif return_type == 'tensor':
            self.__get_item__fn = self.__getitem__tensor
        self.imgpath_list = glob.glob(os.path.join(self.images_path, '*.jpg'))
        self.all_imgname_to_caplist = self.__all_imgname_to_caplist_dict()
        self.imgname_to_caplist = self.__get_imgname_to_caplist_dict(self.__get_imgpath_list(dist))
        self.startseq = startseq.strip()
        self.endseq = endseq.strip()
        self.unkseq = unkseq.strip()
        self.padseq = padseq.strip()
        self.pad_value = pad_value
        if vocab_set is None:
            self.vocab, self.word2idx, self.idx2word, self.max_len = self.__construct_vocab()
        else:
            self.vocab, self.word2idx, self.idx2word, self.max_len = vocab_set
        if transformations is None:
            self.transformations = transforms.ToTensor()
        else:
            self.transformations = transformations
        self.load_img_to_memory = load_img_to_memory
        self.pil_d = None
        self.db = self.get_db()

    def __len__(self):
        return len(self.db)

    def __getitem__(self, index):
        return self.__get_item__fn(index)

    def __getitem__corpus(self, index):
        imgname = self.db[index][0]
        cap_wordlist = self.db[index][1]
        cap_lenlist = self.db[index][2]
        if self.load_img_to_memory:
            img_tens = self.pil_d[imgname]
        else:
            img_tens = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens)
        return img_tens, cap_wordlist, cap_lenlist, imgname

    def __getitem__tensor(self, index):
        imgname = self.db[index][0]
        caption = self.db[index][1]
        cap_toks = [self.startseq] + nltk.word_tokenize(caption) + [self.endseq]
        cap_len = len(cap_toks)
        if self.load_img_to_memory:
            img_tens = self.pil_d[imgname]
        else:
            img_tens = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')
        img_tens = self.transformations(img_tens)
        cap_tens = torch.LongTensor(self.max_len).fill_(self.pad_value)
        cap_tens[:cap_len] = torch.LongTensor([self.word2idx[word] for word in cap_toks])
        return img_tens, cap_tens, cap_len

    def __all_imgname_to_caplist_dict(self):
        captions = open(self.token, 'r').read().strip().split('\n')
        imgname_to_caplist = {}
        for row in captions:
            row = row.split('\t')
            row[0] = row[0][:-2]
            if row[0] in imgname_to_caplist:
                imgname_to_caplist[row[0]].append(row[1])
            else:
                imgname_to_caplist[row[0]] = [row[1]]
        return imgname_to_caplist

    def __get_imgpath_list(self, dist='val'):
        dist_images = set(open(self.dist_list[dist], 'r').read().strip().split('\n'))
        dist_imgpathlist = split_data(dist_images, self.imgpath_list)
        return dist_imgpathlist

    def __get_imgname_to_caplist_dict(self, img_path_list):
        d = {}
        for i in img_path_list:
            img_name = os.path.basename(i)
            if img_name in self.all_imgname_to_caplist:
                d[img_name] = self.all_imgname_to_caplist[img_name]
        return d

    def __construct_vocab(self):
        words = [self.startseq, self.endseq, self.unkseq, self.padseq]
        max_len = 0
        for _, caplist in self.imgname_to_caplist.items():
            for cap in caplist:
                cap_words = nltk.word_tokenize(cap.lower())
                words.extend(cap_words)
                max_len = max(max_len, len(cap_words) + 2)
        vocab = sorted(list(set(words)))
        word2idx = {word: index for index, word in enumerate(vocab)}
        idx2word = {index: word for index, word in enumerate(vocab)}
        return vocab, word2idx, idx2word, max_len

    def get_db(self):
        if self.load_img_to_memory:
            self.pil_d = {}
            for imgname in self.imgname_to_caplist.keys():
                self.pil_d[imgname] = Image.open(os.path.join(self.images_path, imgname)).convert('RGB')
        if self.return_type == 'corpus':
            df = []
            for imgname, caplist in self.imgname_to_caplist.items():
                cap_wordlist = []
                cap_lenlist = []
                for caption in caplist:
                    toks = nltk.word_tokenize(caption.lower())
                    cap_wordlist.append(toks)
                    cap_lenlist.append(len(toks))
                df.append([imgname, cap_wordlist, cap_lenlist])
        elif self.return_type == 'tensor':
            tb = ['image_id\tcaption\tcaption_length\n']
            for imgname, caplist in self.imgname_to_caplist.items():
                for cap in caplist:
                    tb.append(f'{imgname}\t{cap.lower()}\t{len(nltk.word_tokenize(cap.lower()))}\n')
            img_id_cap_str = ''.join(tb)
            df = pd.read_csv(io.StringIO(img_id_cap_str), delimiter='\t').to_numpy()
        return df

    def get_vocab(self):
        return self.vocab, self.word2idx, self.idx2word, self.max_len

    def get_image_captions(self, index):
        imgname = self.db[index][0]
        return os.path.join(self.images_path, imgname), self.imgname_to_caplist[imgname]
