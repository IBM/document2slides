# Convert Grobid tei.xml + PDFFigures into individual json
import os
from os.path import basename, splitext
from os import path
import json
from dataclasses import dataclass
from multiprocessing.pool import Pool
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import nltk


def basename_without_ext(path):
    base_name = basename(path)
    stem, ext = splitext(base_name)
    if stem.endswith('.tei'):
        # Return base name without tei file
        return stem[0:-4]
    else:
        return stem


@dataclass
class Person:
    firstname: str
    middlename: str
    surname: str


def read_tei(tei_file):
    with open(tei_file, 'r', encoding='utf-8') as tei:
        soup = BeautifulSoup(tei, 'xml')
        return soup


def elem_to_text(elem, default=''):
    if elem:
        return elem.getText()
    else:
        return default


class TEIFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.soup = read_tei(filename)
        self._text = None
        self._title = ''
        self._abstract = ''
        self._headers = None
        self._figures = None

    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText()

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    @property
    def authors(self):
        authors_in_header = self.soup.analytic.find_all('author')

        result = []
        for author in authors_in_header:
            persname = author.persname
            if not persname:
                continue
            firstname = elem_to_text(persname.find("forename", type="first"))
            middlename = elem_to_text(persname.find("forename", type="middle"))
            surname = elem_to_text(persname.surname)
            person = Person(firstname, middlename, surname)
            result.append(person)
        return result

    @property
    def text(self):
        if not self._text:
            self._headers = []
            headerlist = self.soup.body.find_all("head")
            sections = []
            for head in headerlist:
                if head.parent.name == 'div':
                    txt = head.parent.get_text(separator=' ', strip=True)
                    if head.get("n"):
                        sections.append([head.text, head.get('n'), txt])
                    else:
                        if len(sections) == 0:
                            print("Grobid processing error: " + self.filename)
                        sections[-1][2] += txt
            start = 0
            for i in sections:
                sent = nltk.tokenize.sent_tokenize(i[2])
                sec_dic = {'section': i[0], 'n': i[1], 'start': start, 'end': start + len(sent) - 1}
                self._headers.append(sec_dic)
                start += len(sent)
            plain_text = " ".join([i[2] for i in sections])
            self._text = [{'id': i, 'string': s} for i, s in enumerate(nltk.tokenize.sent_tokenize(plain_text))]
        return self._text

    @property
    def headers(self):
        if not self._headers:
            self.text()
        return self._headers

    @property
    def figures(self):
        if not self._figures:
            base_name = basename_without_ext(self.filename)
            self._figures = []
            fn = 'figures/{}.json'.format(base_name)  # link to figures dir
            if not path.isfile(fn):
                return []
            with open(fn) as f:
                data = json.load(f)
            for i in data:
                elem = {'filename': i['renderURL'], 'caption': i['caption'], 'page': i['page'],
                        'bbox': i['regionBoundary']}
                self._figures.append(elem)
        return self._figures


def single_entry(tei_file):
    tei = TEIFile(tei_file)
    # print(f"Start {tei_file}")
    base_name = basename_without_ext(tei_file)
    return base_name, tei.title, tei.abstract, tei.text, tei.headers, tei.figures


def main():
    # Step 1: Generating pickle

    pool = Pool()
    papers = sorted(Path('teidir').glob('*.tei.xml'))
    entries = pool.map(single_entry, papers)
    result_df = pd.DataFrame(entries, columns=['ID', 'Title', 'Abstract', 'Text', 'Headers', 'Figs'])
    result_df.to_pickle("papers.pkl")
    print(result_df.head())

    # Step 2: Generating individual json

    # result_pkl = pd.read_pickle('papers.pkl')
    if not os.path.exists('paper_jsons'):
        os.makedirs('paper_jsons')

    for i, row in result_df.iterrows():
        a = row['ID']
        data = {}
        data['title'] = row['Title']
        data['abstract'] = row['Abstract']
        data['text'] = row['Text']
        data['headers'] = row['Headers']
        data['figures'] = row['Figs']
        with open('paper_jsons/' + a + '.json', 'w') as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
