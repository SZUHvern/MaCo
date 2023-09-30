import pandas as pd
import numpy as np
import json

def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj

path = '/mnt/disk2/hwj/MIMIC-DATA-Final/MIMIC-CXR/BASE-MIMIC.csv'
csv = pd.read_csv(path)
values = csv._values
head = csv.head()

splits = csv['split_with_MS']
report = csv['report_ori']
path_imgs = csv


paths = values[:, 4]
# findings = values[:, 5]
# impression = values[:, 6]
texts = values[:, 8]
views = values[:, 10]
cls = values[:, 11:-1]
delete_list = [13, 10, 6, 5, 4]
cls = cls.transpose().tolist()
for i in delete_list:
    cls.pop(i)
cls = np.array(cls).transpose()

cls_pos = []
cls_neg = []
for i in range(len(cls)):
    cls_instance = []
    for ins in cls[i]:
        if ins == 1.0:
            cls_instance.append(1)
        else:
            cls_instance.append(0)
    if sum(cls_instance) == 0:
        cls_instance[5] = 1
    cls_pos.append(cls_instance)
cls_neg = (1 - np.array(cls_pos)).tolist()

all_train = []
all_valid = []
all_test = []

for i in range(len(views)):
    if views[i] == 'Frontal':
        # if splits[i] == 'train' and 'files/' + paths[i] not in MS_images:
        if splits[i] == 'train' and 'files/':
            all_train.append([paths[i], texts[i], cls_pos[i], cls_neg[i]])
        elif splits[i] == 'validate':
            all_valid.append([paths[i], texts[i], cls_pos[i], cls_neg[i]])
        else:
            all_test.append([paths[i], texts[i], cls_pos[i], cls_neg[i]])

cases = json.dumps(all_train)
F = open('MIMIC_train.json', 'w')
F.write(cases)
F.close()

cases = json.dumps(all_valid)
F = open('MIMIC_valid.json', 'w')
F.write(cases)
F.close()

cases = json.dumps(all_test)
F = open('MIMIC_test.json', 'w')
F.write(cases)
F.close()
