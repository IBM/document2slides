# Collect files from external_urls.csv
import requests
from pathlib import Path
import pandas as pd

df = pd.read_csv('external_urls.csv')
Path("data/papers").mkdir(parents=True, exist_ok=True)
Path("data/slides").mkdir(parents=True, exist_ok=True)
for i, row in df.iterrows():
    r1 = requests.get(row['papers'])
    r2 = requests.get(row['slides'])
    f1 = Path('data/papers/{}.pdf'.format(i))
    f1.write_bytes(r1.content)
    f2 = Path('data/slides/{}.pdf'.format(i))
    f2.write_bytes(r2.content)
