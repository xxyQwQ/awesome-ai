import os
import json
import random
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

from utils import indexes2sent


class CodeSearchDataset(Dataset):
    """
    A Dataset class for code search data.

    XXX: Do NOT modify this class.
    """

    def __init__(self, data_instances: List[Dict[str, str]], is_train: bool = False):
        self.is_train = is_train

        self.method_names = []
        self.tokens = []
        self.apiseqs = []
        self.pos_descs = []
        self.neg_descs = []

        for obj in data_instances:
            self.method_names.append(obj["method_name"])
            self.tokens.append(obj["tokens"])
            self.pos_descs.append(obj["desc"])
            self.apiseqs.append(obj["apiseq"])
            # sample a random description as negative sample
            self.neg_descs.append(random.choice(data_instances)["desc"])

        # sanity check
        ref_len = len(self.method_names)
        assert len(self.tokens) == ref_len
        assert len(self.apiseqs) == ref_len
        assert len(self.pos_descs) == ref_len
        assert len(self.neg_descs) == ref_len

        self.data_len = ref_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.is_train:
            return (
                self.method_names[index],
                self.tokens[index],
                self.apiseqs[index],
                self.pos_descs[index],
                self.neg_descs[index],
            )
        else:
            # evaluation, do not provide negative samples
            return (
                self.method_names[index],
                self.tokens[index],
                self.apiseqs[index],
                self.pos_descs[index],
            )


class CodeSearchDataCollator:
    """A collator function used in DataLoader for code search data.
    This collator pads the input sequences to the same length.

    XXX: Do NOT modify this class.
    """

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def _pad_seq(self, input_ids: List[List[int]]) -> torch.LongTensor:
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        return pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )

    def _collate_train(self, batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        (method_ids, token_ids, apiseq_ids, pos_desc_ids, neg_desc_ids) = zip(*batch)

        method_ids = self._pad_seq(method_ids)
        token_ids = self._pad_seq(token_ids)
        apiseq_ids = self._pad_seq(apiseq_ids)
        pos_desc_ids = self._pad_seq(pos_desc_ids)
        neg_desc_ids = self._pad_seq(neg_desc_ids)

        method_padding_mask = (method_ids != self.pad_token_id).float()
        token_padding_mask = (token_ids != self.pad_token_id).float()
        apiseq_padding_mask = (apiseq_ids != self.pad_token_id).float()
        pos_desc_padding_mask = (pos_desc_ids != self.pad_token_id).float()
        neg_desc_padding_mask = (neg_desc_ids != self.pad_token_id).float()

        return {
            "method_ids": method_ids,
            "method_padding_mask": method_padding_mask,
            "token_ids": token_ids,
            "token_padding_mask": token_padding_mask,
            "apiseq_ids": apiseq_ids,
            "apiseq_padding_mask": apiseq_padding_mask,
            "pos_desc_ids": pos_desc_ids,
            "pos_desc_padding_mask": pos_desc_padding_mask,
            "neg_desc_ids": neg_desc_ids,
            "neg_desc_padding_mask": neg_desc_padding_mask,
        }

    def _collate_valid(self, batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        (method_ids, token_ids, apiseq_ids, pos_desc_ids) = zip(*batch)

        method_ids = self._pad_seq(method_ids)
        token_ids = self._pad_seq(token_ids)
        apiseq_ids = self._pad_seq(apiseq_ids)
        pos_desc_ids = self._pad_seq(pos_desc_ids)

        method_padding_mask = (method_ids != self.pad_token_id).float()
        token_padding_mask = (token_ids != self.pad_token_id).float()
        apiseq_padding_mask = (apiseq_ids != self.pad_token_id).float()
        pos_desc_padding_mask = (pos_desc_ids != self.pad_token_id).float()

        return {
            "method_ids": method_ids,
            "method_padding_mask": method_padding_mask,
            "token_ids": token_ids,
            "token_padding_mask": token_padding_mask,
            "apiseq_ids": apiseq_ids,
            "apiseq_padding_mask": apiseq_padding_mask,
            "pos_desc_ids": pos_desc_ids,
            "pos_desc_padding_mask": pos_desc_padding_mask,
        }

    def __call__(self, batch: List[Tuple]) -> Tuple[torch.Tensor, ...]:
        if len(batch[0]) == 4:
            return self._collate_valid(batch)
        elif len(batch[0]) == 5:
            return self._collate_train(batch)
        else:
            raise ValueError("Invalid size of dataset elements.")


def load_dataset(filename: str, is_train: bool) -> CodeSearchDataset:
    """
    Load a JSON dataset file and return a CodeSearchDataset object.

    XXX: Do NOT modify this function.
    """

    with open(filename, "r") as f:
        data_instances = json.load(f)
    return CodeSearchDataset(data_instances, is_train)


def load_vocab(filename: str) -> Dict[str, int]:
    """
    Load a JSON vocabulary file and return a dictionary of [token -> index].

    XXX: Do NOT modify this function.
    """

    with open(filename, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import pprint

    train_set = load_dataset("./data/train.json", is_train=True)
    valid_set = load_dataset("./data/valid.json", is_train=False)

    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=2,
        shuffle=False,
        collate_fn=CodeSearchDataCollator(),
    )

    valid_data_loader = DataLoader(
        dataset=valid_set,
        batch_size=2,
        shuffle=False,
        collate_fn=CodeSearchDataCollator(),
    )

    input_dir = "./data"
    vocab_name = load_vocab(os.path.join(input_dir, "vocab.method_name.json"))
    vocab_tokens = load_vocab(os.path.join(input_dir, "vocab.tokens.json"))
    vocab_apiseq = load_vocab(os.path.join(input_dir, "vocab.apiseq.json"))
    vocab_desc = load_vocab(os.path.join(input_dir, "vocab.comment.json"))

    print("============ Train Data ================")
    k = 0
    for batch in train_data_loader:
        pprint.pprint(batch)

        k += 1
        if k > 20:
            break
        print("-------------------------------")
        print(indexes2sent(batch["method_ids"].tolist(), vocab_name))
        print(indexes2sent(batch["token_ids"].tolist(), vocab_tokens))
        print(indexes2sent(batch["apiseq_ids"].tolist(), vocab_apiseq))
        print(indexes2sent(batch["pos_desc_ids"].tolist(), vocab_desc))

    print("\n\n============ Valid Data ================")
    k = 0
    for batch in valid_data_loader:
        k += 1
        if k > 20:
            break
        print("-------------------------------")
        print(indexes2sent(batch["method_ids"].tolist(), vocab_name))
        print(indexes2sent(batch["token_ids"].tolist(), vocab_tokens))
        print(indexes2sent(batch["apiseq_ids"].tolist(), vocab_apiseq))
        print(indexes2sent(batch["pos_desc_ids"].tolist(), vocab_desc))
