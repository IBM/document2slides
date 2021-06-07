import shutil
import json

with open('suppl_slides_prefilter.json', 'r') as f:
    suppl_data_prefilter = json.load(f)
with open('suppl_slides_filter.json', 'r') as f:
    suppl_data_filter = json.load(f)

with open('../input/acl_slides_prefilter.json','r') as f:
    acl_data_prefilter = json.load(f)
    acl_data_filter = json.load(f)

acl_data_prefilter.update(suppl_data_prefilter)
acl_data_filter.update(suppl_data_filter)

with open('../input/sciduet_slides_prefilter.json','w') as f:
    json.dump(acl_data_prefilter, f)

with open('../input/sciduet_slides_filter.json','w') as f:
    json.dump(acl_data_filter, f)

train_file = open('../input/split/train.txt','r')
train_split = train_file.readlines()
print(train_split)

for i in suppl_data_prefilter:
    shutil.copy('paper_jsons/{}.json'.format(i), '../input/sciduet_papers/')
    train_split.append(i+'\n')

new_file = open('../input/split/train.txt','w')
new_file.writelines(train_split)



