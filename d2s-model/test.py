#!/usr/bin/env python
"""
    Test and validate
"""
import json
from nltk import PorterStemmer
from rouge import Rouge
from spacy.lang.en import English
from pathlib import Path
from tabulate import tabulate

from lfqa_utils import *


def base(mode, s2s_path, cache_path, result_path, file_name):
    predicted = []
    reference = []
    with open('{}/precomputed/{}_prefilter.json'.format(cache_path, mode), 'r') as f:
        cache = json.load(f)

    s2s_tokenizer, s2s_model = make_qa_s2s_model(
        model_name="facebook/bart-large-cnn",
        from_file=s2s_path,
        device="cuda:0"
    )

    for i in range(0, len(cache), 4):
        print(i)
        question_doc = [x[1] for x in cache[i:i+4]]

        answers = qa_s2s_generate_two(
            question_doc, s2s_model, s2s_tokenizer,
            num_answers=1,
            num_beams=8,
            min_len=64,
            max_len=200,
            max_input_length=1024,
            device="cuda:0"
        )
        predicted += answers
        reference += [x[2] for x in cache[i:i+4]]

    if file_name != "":
        Path(result_path).mkdir(parents=True, exist_ok=True)
        output_dic = {'predicted': predicted, 'references': reference}
        with open('{}/{}.json'.format(result_path, file_name), 'w') as f:
            json.dump(output_dic, f)
    rouge_eval(predicted, reference)


def validate(args):
    print("val")
    base('val', args.s2s_model, args.cache_path, args.result_path, args.file_name)


def test_model(args):
    print('test')
    base('test', args.s2s_model, args.cache_path, args.result_path, args.file_name)


def compute_rouge(compare_list):
    stemmer = PorterStemmer()
    rouge = Rouge()
    # tokenizer = English().Defaults.create_tokenizer()
    tokenizer = English().tokenizer

    preds = [" ".join([stemmer.stem(str(w))
                       for w in tokenizer(pred)])
             for gold, pred in compare_list]
    golds = [" ".join([stemmer.stem(str(w))
                       for w in tokenizer(gold)])
             for gold, pred in compare_list]
    scores = rouge.get_scores(preds, golds, avg=True)
    return scores


def rouge_eval(predicted, reference):
    compare_list = [(g, p) for p, g in zip(predicted, reference)]
    scores = compute_rouge(compare_list)
    df = pd.DataFrame({
        'rouge1': [scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']],
        'rouge2': [scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']],
        'rougeL': [scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']],
    }, index=['P', 'R', 'F'])
    print(tabulate(df, headers='keys'))


def only_rouge(result_file):
    with open(result_file, 'r') as f:
        stored_result = json.load(f)
    rouge_eval(stored_result['predicted'], stored_result['references'])