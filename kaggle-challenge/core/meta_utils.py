import numpy as np
from collections import defaultdict


def extract_names(df, label2ind, ind2label):
    all_names_cnt = dict()
    name_dict = defaultdict(list)
    ambiguous_names = list()
    for i in range(len(df)):
        name = df.iloc[i]['name'][2:-1].lower()
        label = df.iloc[i]['label']
        if name in all_names_cnt:
            all_names_cnt[name][label2ind[label]] += 1
        else:
            all_names_cnt[name] = [0] * len(label2ind)
            all_names_cnt[name][label2ind[label]] += 1
    for name, cnts in all_names_cnt.items():
        max_occ = np.max(cnts)
        if np.sum(np.array(cnts) == max_occ) > 1:
            if cnts[0]==max_occ and cnts[1]==max_occ and ('bar' in name or 'grill' in name):
                name_dict['american (traditional)'].append(name)
                continue
            ambiguous_names.append('{} | {}'.format(name, str(cnts)))
            continue
        max_idx = np.argmax(cnts)
        name_dict[ind2label[max_idx]].append(name)
    
    return name_dict, ambiguous_names