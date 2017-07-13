import argparse
import os
import time
import numpy as np
import pandas as pd
import dataset

parser = argparse.ArgumentParser(description='PyTorch Sealion count inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-t, --type', default='vote', type=str, metavar='TYPE',
                    help='Type of ensemble: vote, geometric, arithmetic (default: "vote"')
parser.add_argument('--multi-label', action='store_true', default=False,
                    help='multi-label target')
parser.add_argument('--tif', action='store_true', default=False,
                    help='Use tif dataset')


submission_col = ['image_name', 'tags']


def find_inputs(folder, types=['.csv'], match=''):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            print(rel_filename)
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                if match and base == match:
                    abs_filename = os.path.join(root, rel_filename)
                    inputs.append((base, abs_filename))
    return inputs


def vector_to_tags(v, tags):
    idx = np.nonzero(v)
    t = [tags[i] for i in idx[0]]
    return ' '.join(t)


def main():
    args = parser.parse_args()

    subs = find_inputs(args.data, types=['.csv'], match='results_thr')
    dfs = []
    for s in subs:
        df = pd.read_csv(s[1], index_col=None)
        df = df.set_index('image_name')
        dfs.append(df)

    assert len(dfs)
    d = dfs[0]
    for o in dfs[1:]:
        d = d.add(o)
    d = d / len(dfs)
    b = (d >= 0.5).astype(int)

    tags = dataset.LABEL_ALL
    m = b.as_matrix()
    out = []
    for i, x in enumerate(m):
        t = vector_to_tags(x, tags)
        out.append([b.index[i]] + [t])

    results_sub_df = pd.DataFrame(out, columns=submission_col)
    results_sub_df.to_csv('submission-e.csv', index=False)


if __name__ == '__main__':
    main()
