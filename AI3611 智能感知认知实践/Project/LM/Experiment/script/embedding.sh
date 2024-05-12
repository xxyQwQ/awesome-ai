# rnn architecture
python main.py checkpoint=checkpoint/embedding/rnn-32 model=rnn embedding_dims=32 hidden_dims=512 num_layers=4 # RNN-32
python main.py checkpoint=checkpoint/embedding/rnn-64 model=rnn embedding_dims=64 hidden_dims=512 num_layers=4 # RNN-64
python main.py checkpoint=checkpoint/embedding/rnn-128 model=rnn embedding_dims=128 hidden_dims=512 num_layers=4 # RNN-128
python main.py checkpoint=checkpoint/embedding/rnn-256 model=rnn embedding_dims=256 hidden_dims=512 num_layers=4 # RNN-256
python main.py checkpoint=checkpoint/embedding/rnn-512 model=rnn embedding_dims=512 hidden_dims=512 num_layers=4 # RNN-512

# lstm architecture
python main.py checkpoint=checkpoint/embedding/lstm-32 model=lstm embedding_dims=32 hidden_dims=512 num_layers=4 # LSTM-32
python main.py checkpoint=checkpoint/embedding/lstm-64 model=lstm embedding_dims=64 hidden_dims=512 num_layers=4 # LSTM-64
python main.py checkpoint=checkpoint/embedding/lstm-128 model=lstm embedding_dims=128 hidden_dims=512 num_layers=4 # LSTM-128
python main.py checkpoint=checkpoint/embedding/lstm-256 model=lstm embedding_dims=256 hidden_dims=512 num_layers=4 # LSTM-256
python main.py checkpoint=checkpoint/embedding/lstm-512 model=lstm embedding_dims=512 hidden_dims=512 num_layers=4 # LSTM-512

# gru architecture
python main.py checkpoint=checkpoint/embedding/gru-32 model=gru embedding_dims=32 hidden_dims=512 num_layers=4 # GRU-32
python main.py checkpoint=checkpoint/embedding/gru-64 model=gru embedding_dims=64 hidden_dims=512 num_layers=4 # GRU-64
python main.py checkpoint=checkpoint/embedding/gru-128 model=gru embedding_dims=128 hidden_dims=512 num_layers=4 # GRU-128
python main.py checkpoint=checkpoint/embedding/gru-256 model=gru embedding_dims=256 hidden_dims=512 num_layers=4 # GRU-256
python main.py checkpoint=checkpoint/embedding/gru-512 model=gru embedding_dims=512 hidden_dims=512 num_layers=4 # GRU-512

# transformer architecture
python main.py checkpoint=checkpoint/embedding/transformer-32 model=transformer embedding_dims=32 hidden_dims=1024 num_heads=16 num_layers=8 # Transformer-32
python main.py checkpoint=checkpoint/embedding/transformer-64 model=transformer embedding_dims=64 hidden_dims=1024 num_heads=16 num_layers=8 # Transformer-64
python main.py checkpoint=checkpoint/embedding/transformer-128 model=transformer embedding_dims=128 hidden_dims=1024 num_heads=16 num_layers=8 # Transformer-128
python main.py checkpoint=checkpoint/embedding/transformer-256 model=transformer embedding_dims=256 hidden_dims=1024 num_heads=16 num_layers=8 # Transformer-256
python main.py checkpoint=checkpoint/embedding/transformer-512 model=transformer embedding_dims=512 hidden_dims=1024 num_heads=16 num_layers=8 # Transformer-512

# gpt architecture
python main.py checkpoint=checkpoint/embedding/gpt-32 model=gpt embedding_dims=32 hidden_dims=1024 num_heads=16 num_layers=8 # GPT-32
python main.py checkpoint=checkpoint/embedding/gpt-64 model=gpt embedding_dims=64 hidden_dims=1024 num_heads=16 num_layers=8 # GPT-64
python main.py checkpoint=checkpoint/embedding/gpt-128 model=gpt embedding_dims=128 hidden_dims=1024 num_heads=16 num_layers=8 # GPT-128
python main.py checkpoint=checkpoint/embedding/gpt-256 model=gpt embedding_dims=256 hidden_dims=1024 num_heads=16 num_layers=8 # GPT-256
python main.py checkpoint=checkpoint/embedding/gpt-512 model=gpt embedding_dims=512 hidden_dims=1024 num_heads=16 num_layers=8 # GPT-512
