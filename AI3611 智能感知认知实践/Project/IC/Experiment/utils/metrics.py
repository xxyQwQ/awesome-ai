from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def bleu_score_fn(method_no=4, ref_type='corpus'):
    smoothing_method = getattr(SmoothingFunction(), f'method{method_no}')
    def bleu_score_corpus(reference_corpus, candidate_corpus, n=4):
        weights = [1 / n] * n
        return corpus_bleu(
            reference_corpus,
            candidate_corpus,
            weights=weights,
            smoothing_function=smoothing_method
        )
    def bleu_score_sentence(reference_sentences, candidate_sentence, n=4):
        weights = [1 / n] * n
        return sentence_bleu(
            reference_sentences,
            candidate_sentence,
            weights=weights,
            smoothing_function=smoothing_method
        )
    if ref_type == 'corpus':
        return bleu_score_corpus
    elif ref_type == 'sentence':
        return bleu_score_sentence
