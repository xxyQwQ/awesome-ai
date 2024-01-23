# ablation: no refinement (no correction)

# slu_transformer
python -X utf8 main.py --model slu_transformer --ckpt ./ckpt/refinement --device 0 --num_layer 6 --lr 5e-5 --optimizer AdamW
python -X utf8 main.py --model slu_transformer --ckpt ./ckpt/refinement --device 0 --num_layer 6 --testing

# slu_rnn_crf
python -X utf8 main.py --model slu_rnn_crf --ckpt ./ckpt/refinement --device 0 --encoder_cell LSTM
python -X utf8 main.py --model slu_rnn_crf --ckpt ./ckpt/refinement --device 0 --encoder_cell LSTM --testing

# slu_bert
python -X utf8 main.py --model slu_bert --ckpt ./ckpt/refinement --device 0 --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert --ckpt ./ckpt/refinement --device 0 --testing

# slu_bert_rnn
python -X utf8 main.py --model slu_bert_rnn --ckpt ./ckpt/refinement --device 0 --encoder_cell LSTM --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_rnn --ckpt ./ckpt/refinement --device 0 --encoder_cell LSTM --testing

# slu_bert_crf
python -X utf8 main.py --model slu_bert_crf --ckpt ./ckpt/refinement --device 0 --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_crf --ckpt ./ckpt/refinement --device 0 --testing

# slu_bert_rnn_crf
python -X utf8 main.py --model slu_bert_rnn_crf --ckpt ./ckpt/refinement --device 0 --encoder_cell LSTM --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_rnn_crf --ckpt ./ckpt/refinement --device 0 --encoder_cell LSTM --testing
