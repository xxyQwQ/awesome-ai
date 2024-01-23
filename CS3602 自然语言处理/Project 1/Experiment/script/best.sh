# reproduction: best model

# slu_bert_crf
python -X utf8 main.py --model slu_bert_crf --ckpt ./ckpt/best --device 0 --correction --refinement --lr 2e-3 --optimizer AdamW
python -X utf8 main.py --model slu_bert_crf --ckpt ./ckpt/best --device 0 --refinement --testing
