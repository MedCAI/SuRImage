import os
import numpy as np
import cv2
import imageio
import random
import json

root_path = '/data3/ceiling/datasets/Gross-Combined/ROI/'
json_path = 'ROI_404_standard.json'
print(root_path)

level = ['AIS', 'MIA', '1', '2', '3']
for l in level:
    path = root_path + l
    patient = {}
    total = 0
    for name in os.listdir(path):
        name = name[0:4]
        total += 1
        if name in patient.keys():
            patient[name] += 1
        else:
            patient[name] = 1
    print("Level:{}, Patient Num:{}, Image Num:{}".format(l, len(patient), total))

split_dict = {'train': {},
              'val': {},
              'test': {}}

train = 0.6
val = 0.2
test = 0.2
random.seed(404)

for l in level:
    patient = {}
    path = root_path + l
    total = 0
    for name in os.listdir(path):
        name = name[0:4]  # Notice Name
        total += 1
        if name in patient.keys():
            patient[name] += 1
        else:
            patient[name] = 1
    print("Level:{}, Patient Num:{}, Image Num:{}".format(l, len(patient), total))
    name_list = list(patient.keys())
    random.shuffle(name_list)
    split_dict['train'][l] = name_list[: int(train * len(patient))]
    split_dict['val'][l] = name_list[int(train * len(patient)): int((train + val) * len(patient))]
    split_dict['test'][l] = name_list[int((train + val) * len(patient)):]
    print("Training Patient Num:{}, Validation Patient Num:{}, Testing Patient Num:{}".format(len(split_dict['train'][l]),
                                                   len(split_dict['val'][l]),
                                                   len(split_dict['test'][l])))

json_data = json.dumps(split_dict)
with open(json_path, 'w') as f:
    f.write(json_data)