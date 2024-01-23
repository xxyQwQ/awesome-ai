# ablation: all tricks (including correction)

# slu_transformer
python -X utf8 main.py --model slu_transformer --ckpt ./ckpt/correction --device 0 --correction --refinement --num_layer 6 --lr 5e-5 --optimizer AdamW
python -X utf8 main.py --model slu_transformer --ckpt ./ckpt/correction --device 0 --correction --refinement --num_layer 6 --testing

# slu_rnn_crf
python -X utf8 main.py --model slu_rnn_crf --ckpt ./ckpt/correction --device 0 --correction --refinement --encoder_cell LSTM
python -X utf8 main.py --model slu_rnn_crf --ckpt ./ckpt/correction --device 0 --correction --refinement --encoder_cell LSTM --testing

# slu_bert
python -X utf8 main.py --model slu_bert --ckpt ./ckpt/correction --device 0 --correction --refinement --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert --ckpt ./ckpt/correction --device 0 --correction --refinement --testing

# slu_bert_rnn
python -X utf8 main.py --model slu_bert_rnn --ckpt ./ckpt/correction --device 0 --correction --refinement --encoder_cell LSTM --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_rnn --ckpt ./ckpt/correction --device 0 --correction --refinement --encoder_cell LSTM --testing

# slu_bert_crf
python -X utf8 main.py --model slu_bert_crf --ckpt ./ckpt/correction --device 0 --correction --refinement --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_crf --ckpt ./ckpt/correction --device 0 --correction --refinement --testing

# slu_bert_rnn_crf
python -X utf8 main.py --model slu_bert_rnn_crf --ckpt ./ckpt/correction --device 0 --correction --refinement --encoder_cell LSTM --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_rnn_crf --ckpt ./ckpt/correction --device 0 --correction --refinement --encoder_cell LSTM --testing
