#!/usr/bin/env python
"""
    Information Retrieval
"""
import json
from pathlib import Path
import faiss
from fuzzywuzzy import fuzz

from lfqa_utils import *


###############
# Dense helper
###############
def load_data(slide_json, paper_jsons, split_path):
    with open(slide_json) as f:
        slide_data = json.load(f)
    paper_glob = Path(paper_jsons).glob('*')
    paper_data = {}
    for i in paper_glob:
        idx = str(i).split('.json')[0].split('/')[-1]
        with open(i) as f:
            paper_data[idx] = json.load(f)

    trainlist = [i.rstrip() for i in open(split_path + '/train.txt').readlines()]
    vallist = [i.rstrip() for i in open(split_path + '/val.txt').readlines()]
    testlist = [i.rstrip() for i in open(split_path + '/test.txt').readlines()]
    slide_dic = {}
    slide_dic['train'] = {'id': [], 'paper_title': [], 'slide_num': [], 'title': [], 'answers': []}
    slide_dic['val'] = {'id': [], 'paper_title': [], 'slide_num': [], 'title': [], 'answers': []}
    slide_dic['test'] = {'id': [], 'paper_title': [], 'slide_num': [], 'title': [], 'answers': []}

    for i in slide_data:
        for j in slide_data[i]['slides']:
            title = slide_data[i]['slides'][j]['title']
            text = slide_data[i]['slides'][j]['text']
            if len(text) == 0:
                continue
            papertitle = slide_data[i]['paper_title']
            if i in trainlist:
                category = 'train'
            elif i in vallist:
                category = 'val'
            elif i in testlist:
                category = 'test'
            slide_dic[category]['id'].append(i)
            slide_dic[category]['paper_title'].append(papertitle)
            slide_dic[category]['slide_num'].append(j)
            slide_dic[category]['title'].append(title)
            slide_dic[category]['answers'].append({'id': [0], 'text': ['\n'.join(text)], 'score': [0]})

    for i in slide_dic:
        slide_dic[i] = nlp.Dataset.from_dict(slide_dic[i])
    slide_dic = nlp.DatasetDict(slide_dic)
    slide_dic['train'] = slide_dic['train'].shuffle()

    return slide_dic, paper_data


def preprocess_data(paper_data):
    paper_snippets = {}
    for i in paper_data:
        paper_snippets[i] = {'article_title': [], 'section_title': [], 'passage_text': []}
        paper_snippets[i]['article_title'].append('')
        paper_snippets[i]['section_title'].append('Abstract')
        paper_snippets[i]['passage_text'].append(paper_data[i]['abstract'])
        for j in paper_data[i]['figures']:
            paper_snippets[i]['article_title'].append('')
            title = j['filename'].split('-')[1]
            if 'Table' in title:
                title = 'Table ' + title.split('Table')[1]
            elif 'Figure' in title:
                title = 'Figure ' + title.split("Figure")[1]
            else:
                print('error')
            paper_snippets[i]['section_title'].append(title)
            paper_snippets[i]['passage_text'].append(j['caption'])
        for h in paper_data[i]['headers']:
            sentences = paper_data[i]['text'][h['start']:h['end'] + 1]
            fours = zip(*(iter(sentences),) * 4)  # four sentences averages close to 100 words
            for j in fours:
                tmp = ' '.join([x['string'] for x in j])
                paper_snippets[i]['article_title'].append('')
                paper_snippets[i]['section_title'].append(h['section'])
                paper_snippets[i]['passage_text'].append(tmp)
        paper_snippets[i] = pd.DataFrame.from_dict(paper_snippets[i])
        paper_snippets[i] = nlp.Dataset.from_pandas(paper_snippets[i])
    return paper_snippets


def compute_embeddings(model, tokenizer, snippets, cache_path, is_filter):
    label = 'filter' if is_filter else 'prefilter'
    print(label)
    Path('{}/{}/'.format(cache_path, label)).mkdir(parents=True, exist_ok=True)
    for i in snippets:
        if not os.path.isfile('{}/{}/passages_{}.dat'.format(cache_path, label, i)):
            make_qa_dense_index(
                model, tokenizer, snippets[i], device='cuda:0',
                index_name='{}/{}/passages_{}.dat'.format(cache_path, label, i)
            )
        if not os.path.isfile('{}/{}/keywords_{}.dat'.format(cache_path, label, i)):
            make_keyword_dense_index(
                model, tokenizer, snippets[i], device='cuda:0',
                index_name='{}/{}/keywords_{}.dat'.format(cache_path, label, i)
            )
    faiss_res = faiss.StandardGpuResources()

    paper_passage_reps = {}
    paper_gpu_index = {}
    keyword_reps = {}
    keyword_gpu_index = {}

    for i in snippets:
        paper_passage_reps[i] = np.memmap(
            '{}/{}/passages_{}.dat'.format(cache_path, label, i),
            dtype='float32', mode='r',
            shape=(snippets[i].num_rows, 128)
        )
        keyword_reps[i] = np.memmap(
            '{}/{}/keywords_{}.dat'.format(cache_path, label, i),
            dtype='float32', mode='r',
            shape=(snippets[i].num_rows, 128)
        )
        paper_index_flat = faiss.IndexFlatIP(128)
        paper_gpu_index[i] = faiss.index_cpu_to_gpu(faiss_res, 0, paper_index_flat)
        paper_gpu_index[i].add(paper_passage_reps[i])

        keyword_index_flat = faiss.IndexFlatIP(128)
        keyword_gpu_index[i] = faiss.index_cpu_to_gpu(faiss_res, 0, keyword_index_flat)
        keyword_gpu_index[i].add(keyword_reps[i])
    return paper_gpu_index, keyword_gpu_index


def build_cache(model, tokenizer, slide_dic, snippets, paper_data,
                paper_index, keyword_index, cache_path, is_filter):
    label = 'filter' if is_filter else 'prefilter'
    Path(cache_path + '/precomputed/').mkdir(parents=True, exist_ok=True)
    cache_dic = {'train': [], 'val': [], 'test': []}
    for key in cache_dic.keys():
        for i in range(slide_dic[key].num_rows):
            question = slide_dic[key][i]['title']
            idx = slide_dic[key][i]['id']
            doc, res_list = query_mix_dense_index(
                question, model, tokenizer,
                snippets[idx], paper_index[idx], keyword_index[idx], weight=0.75, device='cuda:0'
            )
            headers = paper_data[idx]['headers']
            headers = [(elem['section'], elem['n']) for elem in headers]
            scrs = [fuzz.ratio(question.lower(), elem[0].lower()) for elem in headers]
            keywords = []
            if len(scrs) > 0 and max(scrs) >= 90:
                begin = headers[scrs.index(max(scrs))]
                for elem in headers:
                    if elem == begin:
                        continue
                    if begin[1] == elem[1][:len(begin[1])]:
                        keywords.append(elem[0])
                if len(keywords) > 0:
                    question = question + ', ' + ', '.join(keywords)
            question_doc = "question: {} context: {}".format(question, doc.strip())
            cache_dic[key].append(('{}_{}'.format(idx, slide_dic[key][i]['slide_num']),
                                   question_doc, slide_dic[key][i]['answers']['text'][0]))
        with open('{}/precomputed/{}_{}.json'.format(cache_path, key, label), 'w') as f:
            json.dump(cache_dic[key], f)


###############
# Main methods
###############
def dense_ir(args):
    print('dense ir')
    slide_dic, paper_data = load_data(args.slide_json, args.paper_path, args.split_path)
    snippets = preprocess_data(paper_data)
    tokenizer, model = make_qa_retriever_model(
        model_name="google/bert_uncased_L-8_H-768_A-12",
        from_file=args.ir_model,
        device="cuda:0"
    )
    paper_index, keyword_index = compute_embeddings(model, tokenizer, snippets,
                                                    args.cache_path, args.filter)
    build_cache(model, tokenizer, slide_dic, snippets, paper_data,
                paper_index, keyword_index, args.cache_path, args.filter)


def idf_recall(args):
    print('eval')
    if args.filter:
        label = 'filter'
    else:
        label = 'prefilter'
    print(label)
    precomputed = json.load(open('{}/precomputed/test_{}.json'
                                 .format(args.cache_path, label)))
    answer_doc = {}
    for idx, d, a in precomputed:
        if idx not in answer_doc:
            answer_doc[idx] = {}
        for w in a.lower().split():
            answer_doc[idx][w] = answer_doc[idx].get(w, 0) + 1

    def scorer(doc, answer, idx):
        d_words = dict([(w, True) for w in doc.lower().split()])
        a_words = answer.lower().split()
        recall = sum([1. / math.log(1 + answer_doc[idx].get(w, 1)) for w in a_words if w in d_words]) / \
                 sum([1. / math.log(1 + answer_doc[idx].get(w, 1)) for w in a_words])
        return recall

    total_retriever_score = 0.0
    for idx, d, a in precomputed:
        d = d.split('context: ', 1)[1]
        total_retriever_score += scorer(d, a, idx)

    print("idf_recall:", total_retriever_score / len(precomputed))
