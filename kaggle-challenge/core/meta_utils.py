import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier

def extract_ambience(line):
    text = line['attributes.Ambience']
    ambience2ind = {'casual': 0,
                    'classy': 1,
                    'divey': 2,
                    'hipster': 3,
                    'intimate': 4,
                    'romantic': 5,
                    'touristy': 6,
                    'trendy': 7,
                    'upscale': 8}
    res = [0] * len(ambience2ind)
    # extract the content within " "
    p = re.compile(r'["](.*?)["]')
    if isinstance(text, str):
        try:
            tmp_dict = eval(re.findall(p, text)[0])
        except:
            return None
        for amb in tmp_dict:
            if tmp_dict[amb]:
                res[ambience2ind[amb]] = 1
        return res
    return None

def extract_parking(line):
    text = line['attributes.BusinessParking']
    place2ind = {'garage': 0,
                 'street': 1,
                 'validated': 2,
                 'lot': 3,
                 'valet': 4}
    res = [0] * len(place2ind)
    # extract the content within " "
    p = re.compile(r'["](.*?)["]')
    if isinstance(text, str):
        try:
            tmp_dict = eval(re.findall(p, text)[0])
        except:
            return None
        for place in tmp_dict:
            if tmp_dict[place]:
                res[place2ind[place]] = 1
        return res
    return None

def extract_forMeal(line):
    text = line['attributes.GoodForMeal']
    meal2ind = {'dessert': 0, 
                'latenight': 1, 
                'lunch': 2, 
                'dinner': 3, 
                'brunch': 4, 
                'breakfast': 5}
    res = [0] * len(meal2ind)
    # extract the content within " "
    p = re.compile(r'["](.*?)["]')
    if isinstance(text, str):
        try:
            tmp_dict = eval(re.findall(p, text)[0])
        except:
            return None
        for meal in tmp_dict:
            if tmp_dict[meal]:
                res[meal2ind[meal]] = 1
        return res
    return None

def extract_price(line):
    text = line['attributes.RestaurantsPriceRange2']
    res = [0] * 4
    if isinstance(text, str):
        try:
            price = int(text[2]) - 1
        except:
            return None
        res[price] = 1
        return res
    return None

def extract_noise(line):
    text = line['attributes.NoiseLevel']
    res = [0] * 3
    if isinstance(text, str):
        text = text.lower()
        if 'quiet' in text:
            res[0] = 1
            return res
        if 'average' in text:
            res[1] = 1
            return res
        if 'loud' in text:
            res[2] = 1
            return res
    return None

def extract_forGroup(line):
    text = line['attributes.RestaurantsGoodForGroups']
    if isinstance(text, str):
        text = text.lower()
        if 'true' in text:
            return [1, 0]
        if 'false' in text:
            return [0, 1]
    return None

def extract_forKids(line):
    text = line['attributes.GoodForKids']
    if isinstance(text, str):
        text = text.lower()
        if 'true' in text:
            return [1, 0]
        if 'false' in text:
            return [0, 1]
    return None

def extract_caters(line):
    text = line['attributes.Caters']
    if isinstance(text, str):
        text = text.lower()
        if 'true' in text:
            return [1, 0]
        if 'false' in text:
            return [0, 1]
    return None

def extract_hours(line):
    text = line['hours']
    res = list()
    if isinstance(text, str):
        hours = eval(text)
        longest_day, longest_ind = 0, 0
        shortet_day, shortest_ind = 24, 0
        total_hours, days = 0, 0
        for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']):
            hour = hours.get(day, None)
            if hour is None:
                res.extend([0, 0, 0])
                continue
            start = int(hour.strip().split('-')[0].split(':')[0])
            end = int(hour.strip().split('-')[1].split(':')[0])
            if end == 0:
                end = 24
            dur = end - start
            total_hours += dur
            days += 1
            if dur > longest_day:
                longest_ind = i
                longest_day = dur
            if dur < shortet_day:
                shortest_ind = i
                shortest_day = dur
            res.extend([start, end, dur])
        res.extend([shortest_ind, longest_ind, total_hours/days])
        return res
    return None

def extract_takeout_delivery(line):
    text1 = line['attributes.RestaurantsDelivery']
    text2 = line['attributes.RestaurantsTakeOut']
    res = list()
    if isinstance(text1, str) and isinstance(text2, str):
        text1 = text1.lower()
        text2 = text2.lower()
        if 'true' in text1:
            res.extend([1, 0])
        if 'false' in text1:
            res.extend([0, 1])
        if 'true' in text2:
            res.extend([1, 0])
        if 'false' in text2:
            res.extend([0, 1])
        if len(res) == 4:
            return res
    return None

def extract_position(line):
    lon = line['longitude']
    lat = line['latitude']
    if isinstance(lon, float) and isinstance(lat, float):
        return [lon, lat]
    return None

def extrat_clean_data(df, cate):
    label2ind = {'american (new)': 0,
                'american (traditional)': 1,
                'asian fusion': 2,
                'canadian (new)': 3,
                'chinese': 4,
                'italian': 5,
                'japanese': 6,
                'mediterranean': 7,
                'mexican': 8,
                'thai': 9}
    vectors, labels = list(), list()
    for i in range(len(df)):
        vec = eval('extract_{}'.format(cate))(df.iloc[i])
        if vec is None:
            continue
        vectors.append(vec)
        labels.append(label2ind[df.iloc[i]['label']])
    return vectors, labels

def train_multi_clf(df, feats, class_weight):
    clfs = dict()
    label_list = ['american (new)', 'american (traditional)']
    df = df[[label in label_list for label in df['label']]]
    for cate in feats:
        X, y = extrat_clean_data(df, cate)
        print (np.sum(np.array(y)==0), np.sum(np.array(y)==1))
        clf = RandomForestClassifier(n_estimators=200, class_weight=class_weight)
        clf.fit(X, y)
        print ('{}: {}'.format(cate, clf.score(X, y)))
        clfs[cate] = clf
    return clfs

def test_multi_clf(df, clfs, feats, save_file='multi_clf.csv'):
    ind2label = {-1: 'None', 0: 'american (new)', 1: 'american (traditional)'}
    dic = {"Id": [], "Predicted": []}
    for i in range(len(df)):
        dic['Id'].append(i)
        line = df.iloc[i]
        preds = list()
        for cate in feats:
            vec = eval('extract_{}'.format(cate))(line)
            if vec is None:
                preds.append(-1)
            else:
                pred = clfs[cate].predict([vec])[0]
                preds.append(pred)
        hour_pred = preds[-1]
        preds = Counter(preds)
        
        if preds[1] > preds[0] and preds[1] >= 4:
            final_pred = 1
        elif preds[0] > preds[1] and preds[0] >= 4:
            final_pred = 0
        elif preds[0] == preds[1] and preds[0] == 4:
            final_pred = hour_pred
        else:
            final_pred = -1

        dic["Predicted"].append(ind2label[final_pred])
    dic_df = pd.DataFrame.from_dict(dic)
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    dic_df.to_csv(os.path.join(out_path, save_file), index=False)



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