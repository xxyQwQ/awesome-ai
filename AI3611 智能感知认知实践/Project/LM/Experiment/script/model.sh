# rnn architecture
python main.py checkpoint=checkpoint/model/rnn-1-64 model=rnn hidden_dims=64 num_layers=1 # RNN-1-64
python main.py checkpoint=checkpoint/model/rnn-1-128 model=rnn hidden_dims=128 num_layers=1 # RNN-1-128
python main.py checkpoint=checkpoint/model/rnn-2-128 model=rnn hidden_dims=128 num_layers=2 # RNN-2-128
python main.py checkpoint=checkpoint/model/rnn-2-256 model=rnn hidden_dims=256 num_layers=2 # RNN-2-256
python main.py checkpoint=checkpoint/model/rnn-4-256 model=rnn hidden_dims=256 num_layers=4 # RNN-4-256
python main.py checkpoint=checkpoint/model/rnn-4-512 model=rnn hidden_dims=512 num_layers=4 # RNN-4-512

# lstm architecture
python main.py checkpoint=checkpoint/model/lstm-1-64 model=lstm hidden_dims=64 num_layers=1 # LSTM-1-64
python main.py checkpoint=checkpoint/model/lstm-1-128 model=lstm hidden_dims=128 num_layers=1 # LSTM-1-128
python main.py checkpoint=checkpoint/model/lstm-2-128 model=lstm hidden_dims=128 num_layers=2 # LSTM-2-128
python main.py checkpoint=checkpoint/model/lstm-2-256 model=lstm hidden_dims=256 num_layers=2 # LSTM-2-256
python main.py checkpoint=checkpoint/model/lstm-4-256 model=lstm hidden_dims=256 num_layers=4 # LSTM-4-256
python main.py checkpoint=checkpoint/model/lstm-4-512 model=lstm hidden_dims=512 num_layers=4 # LSTM-4-512

# gru architecture
python main.py checkpoint=checkpoint/model/gru-1-64 model=gru hidden_dims=64 num_layers=1 # GRU-1-64
python main.py checkpoint=checkpoint/model/gru-1-128 model=gru hidden_dims=128 num_layers=1 # GRU-1-128
python main.py checkpoint=checkpoint/model/gru-2-128 model=gru hidden_dims=128 num_layers=2 # GRU-2-128
python main.py checkpoint=checkpoint/model/gru-2-256 model=gru hidden_dims=256 num_layers=2 # GRU-2-256
python main.py checkpoint=checkpoint/model/gru-4-256 model=gru hidden_dims=256 num_layers=4 # GRU-4-256
python main.py checkpoint=checkpoint/model/gru-4-512 model=gru hidden_dims=512 num_layers=4 # GRU-4-512

# transformer architecture
python main.py checkpoint=checkpoint/model/transformer-2-128 model=transformer hidden_dims=128 num_heads=2 num_layers=2 # Transformer-2-128
python main.py checkpoint=checkpoint/model/transformer-2-256 model=transformer hidden_dims=256 num_heads=4 num_layers=2 # Transformer-2-256
python main.py checkpoint=checkpoint/model/transformer-4-256 model=transformer hidden_dims=256 num_heads=4 num_layers=4 # Transformer-4-256
python main.py checkpoint=checkpoint/model/transformer-4-512 model=transformer hidden_dims=512 num_heads=8 num_layers=4 # Transformer-4-512
python main.py checkpoint=checkpoint/model/transformer-8-512 model=transformer hidden_dims=512 num_heads=8 num_layers=8 # Transformer-8-512
python main.py checkpoint=checkpoint/model/transformer-8-1024 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 # Transformer-8-1024

# gpt architecture
python main.py checkpoint=checkpoint/model/gpt-2-128 model=gpt hidden_dims=128 num_heads=2 num_layers=2 # GPT-2-128
python main.py checkpoint=checkpoint/model/gpt-2-256 model=gpt hidden_dims=256 num_heads=4 num_layers=2 # GPT-2-256
python main.py checkpoint=checkpoint/model/gpt-4-256 model=gpt hidden_dims=256 num_heads=4 num_layers=4 # GPT-4-256
python main.py checkpoint=checkpoint/model/gpt-4-512 model=gpt hidden_dims=512 num_heads=8 num_layers=4 # GPT-4-512
python main.py checkpoint=checkpoint/model/gpt-8-512 model=gpt hidden_dims=512 num_heads=8 num_layers=8 # GPT-8-512
python main.py checkpoint=checkpoint/model/gpt-8-1024 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 # GPT-8-1024
