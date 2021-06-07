# Extract text from slides with builtin unix pdftotext
import os
import json
from glob import glob
import string
from unidecode import unidecode
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz
import nlp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def pdf_to_text(pdf_in, txt_out):
    num_slides = len(glob(pdf_in + '/slides/*'))
    for i in range(num_slides):
        os.system("pdftotext {}/slides/{}.pdf {}/{}.txt".format(pdf_in, i, txt_out, i))


def txt_to_json(txt_dir):
    all_slides_json = {}
    num_slides = len(glob(txt_dir + '/*.txt'))
    for i in range(num_slides):
        with open('slide_txts/{}.txt'.format(i), 'r', encoding='utf-8') as f:
            deck = f.read().split('\f')
            all_slides_json[i] = {'titles': [], 'texts': []}
            for page in deck:
                to_keep = []
                lines = page.split('\n')
                title = ""
                for idx, line in enumerate(lines):
                    if "sha1" in line:
                        continue
                    line = unidecode(line)
                    if len(line) < 2:
                        continue
                    nospace = line.replace(' ', '')
                    if len(nospace) == 0:
                        continue
                    if sum(c in string.ascii_letters for c in nospace) / len(nospace) < 0.75:
                        continue
                    if idx == 0 or title == "":
                        title = line
                        continue
                    if len(to_keep) > 0 and line[0] in string.ascii_lowercase:
                        to_keep[-1] += " " + line
                        continue
                    to_keep.append(line)
                to_keep = [sent for sent in to_keep if len(sent.split()) > 3]
                if title != "" and len(to_keep) > 0:
                    all_slides_json[i]['titles'].append(title)
                    all_slides_json[i]['texts'].append(to_keep)
    return all_slides_json


def merge_titles(json_in):
    df = pd.read_pickle('papers.pkl')
    json_tmp = {}
    for i in json_in:
        counter = 0
        json_tmp[i] = {}
        paper_title = df.loc[df.index[df['ID'] == str(i)][0], 'Title']
        json_tmp[i]['paper_title'] = paper_title
        last_title = None
        for idx in range(len(json_in[i]['titles'])):
            title = json_in[i]['titles'][idx]
            text = json_in[i]['texts'][idx]
            if idx == 0:
                continue
            if json_in[i] == '':
                continue
            result = title
            result = re.sub(r'[^A-Za-z0-9 ]+', '', result)
            if result.isupper():
                result = result.capitalize()
            result = ' '.join(result.split())
            if result == '':
                continue
            if result not in json_tmp[i]:
                json_tmp[i][result] = {'id': counter, 'page_nums': [], 'text': []}
                counter += 1
            json_tmp[i][result]['page_nums'].append(idx)
            for line in text:
                insert = True
                for enum, oldtext in enumerate(json_tmp[i][result]['text']):
                    if fuzz.partial_ratio(line, oldtext) > 90:
                        if len(line) > len(oldtext):
                            json_tmp[i][result]['text'][enum] = line
                        insert = False
                        break
                if insert:
                    json_tmp[i][result]['text'].append(line)
    json_out = {}
    for i in json_tmp:
        json_out[i] = {'slides': {}}
        for j in json_tmp[i]:
            if j == 'paper_title':
                json_out[i][j] = json_tmp[i][j]
                continue
            idx = json_tmp[i][j]['id']
            json_out[i]['slides'][idx] = {'title': j, 'text': json_tmp[i][j]['text'],
                                          'page_nums': json_tmp[i][j]['page_nums']}
    return json_out


def clean_up(json_in):
    for deck in json_in:
        for slide in json_in[deck]['slides']:
            text = json_in[deck]['slides'][slide]['text']
            cleaned = []
            for line in text:
                line = line.strip()
                if len(line) < 3:
                    continue
                if line[1] == ' ':
                    line = line[2:]
                line = line.strip()
                cleaned.append(line)
            json_in[deck]['slides'][slide]['text'] = cleaned
    return json_in


def random_forest(json_in):
    df = pd.read_excel('deriv_rouge.xlsx')

    X = df[['r_1', 'r_2', 'r_L', 'f_1', 'f_2', 'f_L', '3_r', '3_f', 'allsum']]
    y = df['majority']
    X_train_g, X_test_g, y_train_g, y_test_g = X[:50], X[50:], y[:50], y[50:]

    bestacc = 0
    while bestacc < 0.62:
        clf = RandomForestClassifier(max_depth=1)
        clf = clf.fit(X_train_g, y_train_g)
        pred = clf.predict(X_test_g)
        bestacc = accuracy_score(y_test_g, pred)
    print(bestacc)

    nlp_rouge = nlp.load_metric('rouge')
    data = json_in

    for i in data:
        with open('paper_jsons/{}.json'.format(i)) as f:
            paper = json.load(f)

        # Preparing paper data
        full_text = paper['text']
        full_text.append({'id': 0, 'string': paper['abstract']})
        for fig in paper['figures']:
            full_text.append({'id': 0, 'string': fig['caption']})
        textonly = [x['string'] for x in full_text]

        # Preparing slide data
        for j in data[i]['slides']:
            text_arr = data[i]['slides'][j]['text']
            to_keep = []
            for line in text_arr:
                ref = [re.sub(r'[^A-Za-z0-9 ]+', '', line)]
                score = nlp_rouge.compute(textonly, ref * len(textonly), rouge_types=['rouge1', 'rouge2', 'rougeL'],
                                          use_stemmer=True, use_agregator=False)
                arr1 = []
                arr2 = []
                for x in ['rouge1', 'rouge2', 'rougeL']:
                    r_arr = np.array([z.recall for z in score[x]])
                    f_arr = np.array([z.fmeasure for z in score[x]])
                    arr1.append(r_arr.max())
                    arr2.append(f_arr.max())
                arr3 = [arr1 + arr2 + [sum(arr1)] + [sum(arr2)] + [sum(arr1) + sum(arr2)]]
                if clf.predict(arr3)[0]:
                    to_keep.append(line)

            data[i]['slides'][j]['text'] = to_keep
        print(f"{i} Done")
    return data


def main():
    slide_dir = 'data'
    txt_dir = 'slide_txts'
    pdf_to_text(slide_dir, txt_dir)

    prefilter_json = txt_to_json(txt_dir)
    prefilter_json = merge_titles(prefilter_json)
    prefilter_json = clean_up(prefilter_json)

    print(len(prefilter_json))
    with open('suppl_slides_prefilter.json', 'w') as f:
        json.dump(prefilter_json, f)

    filter_json = random_forest(prefilter_json)
    with open('suppl_slides_filter.json', 'w') as f:
        json.dump(filter_json, f)


if __name__ == "__main__":
    main()
