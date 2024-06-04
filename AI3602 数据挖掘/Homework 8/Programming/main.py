import os
import json
import random
import numpy as np
import torch.optim as optim

import torch
import torch.nn.functional as F

from tqdm import tqdm
from logging import Logger
from datetime import datetime
from torch.utils.data import DataLoader
from configs import get_config
from model import CodeNN
from loss import SimilarityLoss
from data_loading import load_dataset, CodeSearchDataCollator
from utils import setup_logger, save_model, move_data_to_device
from metrics import metric_acc, metric_mrr, metric_ndcg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
    eid: int,
    model: CodeNN,
    loss_fn: SimilarityLoss,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    logger: Logger,
    device: torch.device,
):
    model.train()

    progress = tqdm(data_loader)
    tot_loss = 0.0
    tot_pos_sim = 0.0
    tot_neg_sim = 0.0
    for bid, inputs in enumerate(progress):
        optimizer.zero_grad()

        inputs = move_data_to_device(inputs, device)
        code_features, pos_desc_features, neg_desc_features = model(**inputs)

        loss, pos_sim, neg_sim = loss_fn(
            code_features, pos_desc_features, neg_desc_features
        )
        loss.backward()

        optimizer.step()

        tot_loss += loss.item()
        tot_pos_sim += pos_sim.item()
        tot_neg_sim += neg_sim.item()

        avg_loss = tot_loss / (bid + 1)
        avg_pos_sim = tot_pos_sim / (bid + 1)
        avg_neg_sim = tot_neg_sim / (bid + 1)

        progress.set_description(
            f"| TRAIN {eid:3d} | loss: {avg_loss:.4f} "
            f"| pos_sim: {avg_pos_sim:.4f} | neg_sim: {avg_neg_sim:.4f} |"
        )

    logger.info(
        f"TRAIN {eid:3d} | loss: {avg_loss:.4f} "
        f"| pos_sim: {avg_pos_sim:.4f} | neg_sim: {avg_neg_sim:.4f}"
    )


def evaluate(
    eid: int,
    model: CodeNN,
    data_loader: DataLoader,
    logger: Logger,
    device: torch.device,
):
    """XXX: Do NOT modify this function."""

    POOL_SIZE = 1000
    TOPK = 10

    model.eval()

    accs, mrrs, ndcgs = [], [], []
    code_reprs, desc_reprs = [], []

    n_samples = 0
    with torch.no_grad():
        # get embeddings for all samples
        for batch in tqdm(data_loader):
            inputs = move_data_to_device(batch, device)
            code_features = model.code_features(
                inputs["method_ids"],
                inputs["method_padding_mask"],
                inputs["token_ids"],
                inputs["token_padding_mask"],
                inputs["apiseq_ids"],
                inputs["apiseq_padding_mask"],
            )
            desc_features = model.desc_features(
                inputs["pos_desc_ids"],
                inputs["pos_desc_padding_mask"],
            )

            code_features = F.normalize(code_features, p=2, dim=1)
            desc_features = F.normalize(desc_features, p=2, dim=1)

            code_reprs.append(code_features.detach().clone().cpu())
            desc_reprs.append(desc_features.detach().clone().cpu())

            n_samples += inputs["method_ids"].size(0)

        code_reprs = torch.cat(code_reprs, dim=0)
        desc_reprs = torch.cat(desc_reprs, dim=0)

        # perform code search by calculating similarity scores
        # since the dataset could be very large
        # we divide the validation set to smaller pools
        # by default POOL_SIZE = 1000
        for k in tqdm(range(0, n_samples, POOL_SIZE)):

            # perform code search within each pool
            # i.e., for each query, find the topk most similar code snippets in this pool
            code_pool = code_reprs[k : k + POOL_SIZE]
            desc_pool = desc_reprs[k : k + POOL_SIZE]

            # iterate over each query in the pool
            for i in range(POOL_SIZE):
                desc_vec = desc_pool[i].unsqueeze(0)  # [1, dim]

                # calculate cosine similarity between the query and all code snippets
                sims = torch.mm(code_pool, desc_vec.T).squeeze(1)  # [pool_size]

                # find the topk most similar code snippets
                _, indices = torch.topk(sims, k=TOPK)
                indices = indices.tolist()

                # calculate metrics
                accs.append(metric_acc([i], indices))
                mrrs.append(metric_mrr([i], indices))
                ndcgs.append(metric_ndcg([i], indices))

        acc = torch.mean(torch.tensor(accs)).item()
        mrr = torch.mean(torch.tensor(mrrs)).item()
        ndcg = torch.mean(torch.tensor(ndcgs)).item()

        logger.info(
            f"EVAL  {eid:3d} | ACC@{TOPK}: {acc:.4f} "
            f"| MRR: {mrr:.4f} | NDCG@{TOPK}: {ndcg:.4f}"
        )

    return {
        f"ACC@{TOPK}": acc,
        "MRR": mrr,
        f"NDCG@{TOPK}": ndcg,
    }


def main():
    config = get_config()
    set_seed(config.seed)

    # check output directories
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if config.run_name is not None:
        run_name = f"{timestamp}-{config.run_name}"
    else:
        run_name = timestamp

    os.makedirs(f"./output/{run_name}/models", exist_ok=True)
    os.makedirs(f"./output/{run_name}/logs", exist_ok=True)

    # set up logging
    logger = setup_logger(config, log_dir=f"./output/{run_name}/logs")

    logger.info(f"Config: {config}")
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Training on CPU.")
    device = torch.device(f"cuda:{config.gpu_id}" if torch.cuda.is_available() else "cpu")

    # load data
    logger.info("Loading data.")
    train_set = load_dataset(config.train_data_path, is_train=True)
    valid_set = load_dataset(config.valid_data_path, is_train=False)
    logger.info(f"Train set size: {len(train_set):,}")
    logger.info(f"Valid set size: {len(valid_set):,}")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.train_batch_size,
        collate_fn=CodeSearchDataCollator(),
        shuffle=True,
    )
    eval_loader = DataLoader(
        dataset=valid_set,
        batch_size=config.eval_batch_size,
        collate_fn=CodeSearchDataCollator(),
        shuffle=False,
    )

    # build model
    logger.info("Constructing model.")
    model = CodeNN(config)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_params:,}")

    loss_fn = SimilarityLoss(config.sim_loss_margin, return_sims=True)

    # prepare optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # training loop
    logger.info("Beginning training loop.")
    try:
        best_metric = 0
        for eid in range(config.n_epochs):
            train(eid, model, loss_fn, optimizer, train_loader, logger, device)
            eval_res = evaluate(eid, model, eval_loader, logger, device)

            # check and save best model
            if eval_res[config.primary_metric] > best_metric:
                best_metric = eval_res[config.primary_metric]
                logger.info(
                    f"New best at epoch {eid:3d}: "
                    f"{config.primary_metric}: {best_metric:.4f}"
                )
                save_model(model, f"./output/{run_name}/models/model_best.pt")
                logger.info(f"Saved to ./output/{run_name}/models/model_best.pt")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    except Exception as e:
        logger.error(f"Exception: {e}")
        raise e

    finally:
        # XXX: Do NOT modify this block.
        # run a final evaluation after training completed or interrupted
        logger.info("Running final evaluation.")

        # try to run a final evaluation using the best model
        best_ckpt_path = f"./output/{run_name}/models/model_best.pt"
        if not os.path.exists(best_ckpt_path):
            logger.warning("No best model found. Skip final evaluation.")
        else:
            model.load_state_dict(torch.load(best_ckpt_path))
            eval_res = evaluate(100, model, eval_loader, logger, device)
            logger.info(f"Number of trainable parameters: {n_params:,}")
            logger.info(f"Final eval result: {json.dumps(eval_res, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
