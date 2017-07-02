import os
import pandas as pd
import numpy as np
import csv
from collections import Counter
BASE_PATH = '/data/x/amazon'
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

    #arr = train_df.as_matrix(columns=list(count.keys()))
    #arr = arr.astype(np.float32)

    train_df = train_df[(train_df[LABEL_SKY_COVER].T != 0).any()]
    #print(len(blorb.index), len(train_df.index))

    with open('tags_count.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['tag', 'count'])
        for k, v in count.items():
            w.writerow([k, v])
        f.close()

    tags_only = train_df[list(count.keys())]
    corr = tags_only.corr()
    corr.to_csv("corr.csv")

    train_df['fold'] = np.random.randint(0, 5, size=len(train_df.index))
    labels_df = train_df[['image_name', 'fold'] + list(count.keys())]
    labels_df.to_csv("labels.csv", index=False)

    lda = labels_df.as_matrix(columns=LABEL_GROUND_COVER)
    ldas = np.argmax(lda, axis=1)
    for i in range(0, 5):
        print(lda[i], ldas[i])

if __name__ == '__main__':
    main()

