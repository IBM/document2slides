# Scrape paper-slide pairs from recent ML conferences
import requests
import pandas as pd
from bs4 import BeautifulSoup

pair_all = {'titles': [], 'slides': [], 'papers': []}
count = 0

"""
Scrape ICML 2019
"""

url_icml = 'https://icml.cc/Conferences/2019/Schedule'
base_icml = 'https://icml.cc'
pmlr_icml = 'http://proceedings.mlr.press/v97/'

slides_icml = {}
papers_icml = {}

r = requests.get(url_icml)

soup = BeautifulSoup(r.content, "lxml")
filtered = soup.find_all('a', title="Slides")

print(len(filtered))
for i in filtered:
    parent = i.parent.parent.parent.parent
    title = parent.find('div', class_='maincardBody').text
    author = parent.find('div', class_='maincardFooter').text
    link = base_icml + i.attrs['href']
    slides_icml[title] = link

r2 = requests.get(pmlr_icml)

soup2 = BeautifulSoup(r2.content, 'lxml')
filtered2 = soup2.find_all('div', class_='paper')
for i in filtered2:
    title = i.find('p', class_='title').text
    paper = i.find('a', href=True, text='Download PDF')
    papers_icml[title] = paper['href']

print("ICML, slides:", len(slides_icml), "papers:", len(papers_icml))
for i in slides_icml:
    if i in papers_icml:
        count += 1
        pair_all['titles'].append(i)
        pair_all['slides'].append(slides_icml[i])
        pair_all['papers'].append(papers_icml[i])

"""
Scrape NIPS 2018/2019
"""

url_nips18 = 'https://nips.cc/Conferences/2018/Schedule'
url_nips19 = 'https://nips.cc/Conferences/2019/Schedule'
base_nips = 'https://nips.cc'
base2_nips = 'https://proceedings.neurips.cc'

r = requests.get(url_nips18)
soup = BeautifulSoup(r.content, "lxml")
filtered1 = soup.find_all('a', title="Paper")

r = requests.get(url_nips19)
soup = BeautifulSoup(r.content, "lxml")
filtered2 = soup.find_all('a', title="Paper")

filtered = [filtered2, filtered1]

print("NIPS, 2018:", len(filtered1), "2019:", len(filtered2))
for idx, year in enumerate(filtered):
    for i in year:
        link = None
        parent = i.parent.parent.parent.parent
        if idx == 1:
            tmp = parent.find_all('a', title='Spotlight Slides')
        else:
            tmp = parent.find_all('a', title='Slides')
        for j in tmp:
            if j['href'][-4:] == '.pdf':
                if j['href'][0] == '/':
                    link = base_nips + j['href']
                else:
                    link = j['href']
        if link:
            try:
                r1 = requests.get(link)
                content_type = r1.headers.get('content-type')
                if 'application/pdf' not in content_type:
                    continue
                redirect = i['href']
                r2 = requests.get(redirect)
                tmpsoup = BeautifulSoup(r2.content, 'lxml')
                paper = base2_nips + tmpsoup.find('a', text='Paper »')['href']
                title = tmpsoup.find_all('h4')[0].text

                pair_all['titles'].append(title)
                pair_all['slides'].append(link)
                pair_all['papers'].append(paper)
                count += 1
            except:
                print('error:', link)

"""
Output to csv
"""

pd.DataFrame(pair_all).to_csv("external_urls.csv")
print('total external:', count)
