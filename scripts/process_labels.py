import os
import pandas as pd
import numpy as np
import csv
import math
from collections import Counter
BASE_PATH = '/data/amazon'
TRAIN_CSV = 'train_v2.csv'

LABEL_ALL = [
    'blow_down',
    'conventional_mine',
    'slash_burn',
    'blooming',
    'artisinal_mine',
    'selective_logging',
    'bare_ground',
    'cloudy',
    'haze',
    'habitation',
    'cultivation',
    'partly_cloudy',
    'water',
    'road',
    'agriculture',
    'clear',
    'primary',
]

LABEL_GROUND_COVER = [
    'blow_down',
    'conventional_mine',
    'slash_burn',
    'blooming',
    'artisinal_mine',
    'selective_logging',
    'bare_ground',
    'habitation',
    'cultivation',
    'water',
    'road',
    'agriculture',
    'primary',
]

LABEL_SKY_COVER = [
    'cloudy',
    'haze',
    'partly_cloudy',
    'clear',
]


def main():
    train_df = pd.read_csv(os.path.join(BASE_PATH, TRAIN_CSV))
    train_df.tags = train_df.tags.map(lambda x: set(x.split()))

    count = Counter()
    train_df.tags.apply(lambda x: count.update(x))

    for k in count:
        train_df[k] = [1 if k in tag else 0 for tag in train_df.tags]

    train_df = train_df[(train_df[LABEL_SKY_COVER].T != 0).any()]

    with open('tags_count.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['tag', 'count'])
        for k, v in count.items():
            w.writerow([k, v])
        f.close()

    tags_only = train_df[list(count.keys())]
    corr = tags_only.corr()
    corr.to_csv("corr.csv")

    attempt = 0
    num_folds = 12
    target_counts = {k: (v / num_folds) for k, v in count.items()}
    target_thresh = {k: max(1., v * .20) for k, v in target_counts.items()}
    print(target_counts, target_thresh)
    furthest_fold = 0
    fold_counts = []
    while attempt < 1000000:
        train_df['fold'] = np.random.randint(0, num_folds, size=len(train_df.index))
        valid = True
        ss = train_df.groupby('fold').sum()
        for f in range(num_folds):
            sr = ss.ix[f]
            fold_counts.append(sr)
            for k, v in sr.items():
                target = target_counts[k]
                thresh = target_thresh[k]
                diff = math.floor(abs(v - target))
                thresh = 3.0 if k == 'conventional_mine' else math.ceil(thresh)
                if diff > thresh:
                    valid = False
                    if f > furthest_fold:
                        furthest_fold = f
                        print(f, abs(v - target), math.ceil(thresh), k)
                    break
            if not valid:
                break
        if valid:
            break
        else:
            fold_counts = []
            attempt += 1
    print(attempt, furthest_fold)
    for i, x in enumerate(fold_counts):
        print(i)
        for k, v in x.items():
            print(k, v)
        print()
    labels_df = train_df[['image_name', 'fold'] + list(count.keys())]
    labels_df.to_csv("labels.csv", index=False)


if __name__ == '__main__':
    main()

