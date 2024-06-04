import os
import torch
import torch.nn as nn

import logging
import treelib
import tree_sitter
from configs import Config
from datetime import datetime
from typing import List, Dict


def pprint_tree(root: tree_sitter.Node):
    tree = treelib.Tree()

    def _build_treelib_tree(current: tree_sitter.Node, parent=None):
        def _format_node(node: tree_sitter.Node):
            node_text = node.text.decode()
            if node.child_count == 0 and node.type != node_text:
                node_str = f"{node.type} ({node_text})"
            else:
                node_str = f"{node.type}"
            return node_str

        tree.create_node(_format_node(current), current.id, parent=parent)
        for child in current.children:
            _build_treelib_tree(child, current.id)

    _build_treelib_tree(root)
    print(tree.show(key=lambda x: True, stdout=False))  # keep order of insertion


def indexes2sent(
    input_ids: List[List[int]] | List[int],
    vocab: Dict[str, int],
    pad_token_id: int = 0,
    eos_token_id: int = 2,
):

    def revert_sent(indexes, ivocab, ignore_tok=0):
        indexes = filter(lambda i: i != ignore_tok, indexes)
        toks, length = [], 0
        for idx in indexes:
            toks.append(ivocab.get(idx, "<unk>"))
            length += 1
            if idx == eos_token_id:
                break
        return " ".join(toks), length

    ivocab = {v: k for k, v in vocab.items()}
    # one sentence
    if isinstance(input_ids[0], int):
        return revert_sent(input_ids, ivocab, pad_token_id)
    else:  # dim > 1
        sentences, lens = [], []  # a batch of sentences
        for inds in input_ids:
            sentence, length = revert_sent(inds, ivocab, pad_token_id)
            sentences.append(sentence)
            lens.append(length)
        return sentences, lens


def setup_logger(config: Config, log_dir: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"seed{config.seed}.log"

    filename = f"{timestamp}-{filename}"
    filename = os.path.join(log_dir, filename)

    return _setup_training_logger(filename)


def _setup_training_logger(filename: str):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(message)s")

    file_handler = logging.FileHandler(filename=filename, mode="a", encoding="utf-8")
    file_formatter = logging.Formatter("[%(levelname)s %(asctime)s]: %(message)s")

    stream_handler.setFormatter(stream_formatter)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def save_model(model: nn.Module, ckpt_path: str):
    torch.save(model.state_dict(), ckpt_path)


def move_data_to_device(inputs: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in inputs.items()}
