# rnn architecture
python main.py checkpoint=checkpoint/dropout/rnn-0.1 model=rnn hidden_dims=512 num_layers=4 dropout=0.1 # RNN-0.1
python main.py checkpoint=checkpoint/dropout/rnn-0.2 model=rnn hidden_dims=512 num_layers=4 dropout=0.2 # RNN-0.2
python main.py checkpoint=checkpoint/dropout/rnn-0.3 model=rnn hidden_dims=512 num_layers=4 dropout=0.3 # RNN-0.3
python main.py checkpoint=checkpoint/dropout/rnn-0.4 model=rnn hidden_dims=512 num_layers=4 dropout=0.4 # RNN-0.4
python main.py checkpoint=checkpoint/dropout/rnn-0.5 model=rnn hidden_dims=512 num_layers=4 dropout=0.5 # RNN-0.5

# lstm architecture
python main.py checkpoint=checkpoint/dropout/lstm-0.1 model=lstm hidden_dims=512 num_layers=4 dropout=0.1 # LSTM-0.1
python main.py checkpoint=checkpoint/dropout/lstm-0.2 model=lstm hidden_dims=512 num_layers=4 dropout=0.2 # LSTM-0.2
python main.py checkpoint=checkpoint/dropout/lstm-0.3 model=lstm hidden_dims=512 num_layers=4 dropout=0.3 # LSTM-0.3
python main.py checkpoint=checkpoint/dropout/lstm-0.4 model=lstm hidden_dims=512 num_layers=4 dropout=0.4 # LSTM-0.4
python main.py checkpoint=checkpoint/dropout/lstm-0.5 model=lstm hidden_dims=512 num_layers=4 dropout=0.5 # LSTM-0.5

# gru architecture
python main.py checkpoint=checkpoint/dropout/gru-0.1 model=gru hidden_dims=512 num_layers=4 dropout=0.1 # GRU-0.1
python main.py checkpoint=checkpoint/dropout/gru-0.2 model=gru hidden_dims=512 num_layers=4 dropout=0.2 # GRU-0.2
python main.py checkpoint=checkpoint/dropout/gru-0.3 model=gru hidden_dims=512 num_layers=4 dropout=0.3 # GRU-0.3
python main.py checkpoint=checkpoint/dropout/gru-0.4 model=gru hidden_dims=512 num_layers=4 dropout=0.4 # GRU-0.4
python main.py checkpoint=checkpoint/dropout/gru-0.5 model=gru hidden_dims=512 num_layers=4 dropout=0.5 # GRU-0.5

# transformer architecture
python main.py checkpoint=checkpoint/dropout/transformer-0.1 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.1 # Transformer-0.1
python main.py checkpoint=checkpoint/dropout/transformer-0.2 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.2 # Transformer-0.2
python main.py checkpoint=checkpoint/dropout/transformer-0.3 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.3 # Transformer-0.3
python main.py checkpoint=checkpoint/dropout/transformer-0.4 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.4 # Transformer-0.4
python main.py checkpoint=checkpoint/dropout/transformer-0.5 model=transformer hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.5 # Transformer-0.5

# gpt architecture
python main.py checkpoint=checkpoint/dropout/gpt-0.1 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.1 # GPT-0.1
python main.py checkpoint=checkpoint/dropout/gpt-0.2 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.2 # GPT-0.2
python main.py checkpoint=checkpoint/dropout/gpt-0.3 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.3 # GPT-0.3
python main.py checkpoint=checkpoint/dropout/gpt-0.4 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.4 # GPT-0.4
python main.py checkpoint=checkpoint/dropout/gpt-0.5 model=gpt hidden_dims=1024 num_heads=16 num_layers=8 dropout=0.5 # GPT-0.5
