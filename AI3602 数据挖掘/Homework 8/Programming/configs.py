from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """This class defines the configuration for the model training.
    Add more configuration options as needed.

    By default, most configurations follow the original implementation of CodeNN
    But you are free to make any changes to the configurations as needed.
    """

    # dataset
    train_data_path: str = "./data/train.json"
    valid_data_path: str = "./data/valid.json"
    name_len: int = 6
    tokens_len: int = 50
    desc_len: int = 30
    pad_token_id: int = 0

    # vocabulary info
    n_tokens: int = 10000
    vocab_name: str = "./data/vocab.name.json"
    vocab_tokens: str = "./data/vocab.tokens.json"
    vocab_desc: str = "./data/vocab.comment.json"

    # misc
    gpu_id: int = 0
    run_name: Optional[str] = None  # change this to your preferred run name
    primary_metric: str = "ACC@10"

    # training hyperparameters
    seed: int = 42
    train_batch_size: int = 64
    eval_batch_size: int = 200
    n_epochs: int = 15
    learning_rate: float = 0.001
    # NOTE: 0.05 is the default value used in CodeNN's paper.
    # However, the original CodeNN is implemented with Keras
    # and this 0.05 value somehow does not work well with PyTorch.
    # You might want to increase it to 0.2 or even higher for better performance.
    sim_loss_margin: float = 0.2

    # model architecture
    # dimension for word/token embeddings
    embedding_dim: int = 128
    # hidden dimension for mlp
    mlp_hidden_dim: int = 256
    # hidden dimension for the fusion mlp
    fusion_hidden_dim: int = 512
    # hidden dimension for the lstm
    lstm_hidden_dim: int = 256
    lstm_n_layers: int = 1
    # activation function
    token_activation: str = "tanh"
    fusion_activation: str = "tanh"
    # dropout
    dropout: Optional[float] = None


def get_config() -> Config:
    """Instantiate the configuration object. Change the parameters as needed."""
    return Config()
