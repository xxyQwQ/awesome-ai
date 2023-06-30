# convert the result of kaldi into the submission file
score_path = "exp/nnet3/tdnn_sp/decode_test/scoring_kaldi/best_cer"
with open(score_path, "r") as score:
    parameter = score.readline().strip().split('/')[-1]
weight, penalty = parameter.split('_')[1:]
source_path = "exp/nnet3/tdnn_sp/decode_test/scoring_kaldi/penalty_{}/{}.txt".format(penalty, weight)
target_path = "label.csv"
with open(source_path, "r") as source, open(target_path, "w") as target:
    target.write('uttid,result\n')
    for line in source:
        uttid, result = line.strip().split(' ', 1)
        target.write('{},{}\n'.format(uttid, result.replace(' ', '')))
