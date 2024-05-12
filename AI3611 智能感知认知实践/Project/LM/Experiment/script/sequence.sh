# rnn architecture
python main.py checkpoint=checkpoint/sequence/rnn-4 model=rnn hidden_dims=512 num_layers=4 sequence_length=4 # RNN-4
python main.py checkpoint=checkpoint/sequence/rnn-8 model=rnn hidden_dims=512 num_layers=4 sequence_length=8 # RNN-8
python main.py checkpoint=checkpoint/sequence/rnn-16 model=rnn hidden_dims=512 num_layers=4 sequence_length=16 # RNN-16
python main.py checkpoint=checkpoint/sequence/rnn-32 model=rnn hidden_dims=512 num_layers=4 sequence_length=32 # RNN-32
python main.py checkpoint=checkpoint/sequence/rnn-64 model=rnn hidden_dims=512 num_layers=4 sequence_length=64 # RNN-64

# lstm architecture
python main.py checkpoint=checkpoint/sequence/lstm-4 model=lstm hidden_dims=512 num_layers=4 sequence_length=4 # LSTM-4
python main.py checkpoint=checkpoint/sequence/lstm-8 model=lstm hidden_dims=512 num_layers=4 sequence_length=8 # LSTM-8
python main.py checkpoint=checkpoint/sequence/lstm-16 model=lstm hidden_dims=512 num_layers=4 sequence_length=16 # LSTM-16
python main.py checkpoint=checkpoint/sequence/lstm-32 model=lstm hidden_dims=512 num_layers=4 sequence_length=32 # LSTM-32
python main.py checkpoint=checkpoint/sequence/lstm-64 model=lstm hidden_dims=512 num_layers=4 sequence_length=64 # LSTM-64

# gru architecture
python main.py checkpoint=checkpoint/sequence/gru-4 model=gru hidden_dims=512 num_layers=4 sequence_length=4 # GRU-4
python main.py checkpoint=checkpoint/sequence/gru-8 model=gru hidden_dims=512 num_layers=4 sequence_length=8 # GRU-8
python main.py checkpoint=checkpoint/sequence/gru-16 model=gru hidden_dims=512 num_layers=4 sequence_length=16 # GRU-16
python main.py checkpoint=checkpoint/sequence/gru-32 model=gru hidden_dims=512 num_layers=4 sequence_length=32 # GRU-32
python main.py checkpoint=checkpoint/sequence/gru-64 model=gru hidden_dims=512 num_layers=4 sequence_length=64 # GRU-64

# transformer architecture
python main.py checkpoint=checkpoint/sequence/transformer-4 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=4 # Transformer-4
python main.py checkpoint=checkpoint/sequence/transformer-8 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=8 # Transformer-8
python main.py checkpoint=checkpoint/sequence/transformer-16 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=16 # Transformer-16
python main.py checkpoint=checkpoint/sequence/transformer-32 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=32 # Transformer-32
python main.py checkpoint=checkpoint/sequence/transformer-64 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=64 # Transformer-64

# gpt architecture
python main.py checkpoint=checkpoint/sequence/gpt-4 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=4 # GPT-4
python main.py checkpoint=checkpoint/sequence/gpt-8 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=8 # GPT-8
python main.py checkpoint=checkpoint/sequence/gpt-16 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=16 # GPT-16
python main.py checkpoint=checkpoint/sequence/gpt-32 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=32 # GPT-32
python main.py checkpoint=checkpoint/sequence/gpt-64 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 sequence_length=64 # GPT-64
