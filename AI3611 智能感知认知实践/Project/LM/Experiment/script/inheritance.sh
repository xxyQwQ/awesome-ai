# rnn architecture
python main.py checkpoint=checkpoint/inheritance/rnn-enabled model=rnn hidden_dims=512 num_layers=4 reuse_hidden=true # RNN-enabled
python main.py checkpoint=checkpoint/inheritance/rnn-disabled model=rnn hidden_dims=512 num_layers=4 reuse_hidden=false # RNN-disabled

# lstm architecture
python main.py checkpoint=checkpoint/inheritance/lstm-enabled model=lstm hidden_dims=512 num_layers=4 reuse_hidden=true # LSTM-enabled
python main.py checkpoint=checkpoint/inheritance/lstm-disabled model=lstm hidden_dims=512 num_layers=4 reuse_hidden=false # LSTM-disabled

# gru architecture
python main.py checkpoint=checkpoint/inheritance/gru-enabled model=gru hidden_dims=512 num_layers=4 reuse_hidden=true # GRU-enabled
python main.py checkpoint=checkpoint/inheritance/gru-disabled model=gru hidden_dims=512 num_layers=4 reuse_hidden=false # GRU-disabled

# transformer architecture
python main.py checkpoint=checkpoint/inheritance/transformer model=transformer hidden_dims=1024 num_heads=16 num_layers=8 # Transformer

# gpt architecture
python main.py checkpoint=checkpoint/inheritance/gpt model=gpt hidden_dims=1024 num_heads=16 num_layers=8 # GPT
