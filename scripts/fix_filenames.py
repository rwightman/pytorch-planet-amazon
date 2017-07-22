import os
import shutil
import pandas as pd

BASEPATH = './'
WORKING = './working'

CSVPATH = os.path.join(BASEPATH, 'test_v2_file_mapping.csv')
JPGPATH = os.path.join(BASEPATH, 'test-jpg-v2')
TIFPATH = os.path.join(BASEPATH, 'test-tif-v2')


def find_images(folder, types=['.jpg', '.jpeg']):
    results = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            if os.path.splitext(rel_filename)[1].lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                basename = os.path.splitext(rel_filename)[0]
                results.append((basename, abs_filename))
    return results


def get_outdir(path, *paths):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

FIXEDPATH = get_outdir(WORKING, 'fixed-test')
BADPATH = get_outdir(WORKING, 'bad-test')


def main():

    df = pd.read_csv(CSVPATH)
    n = 0
    for index, row in df.iterrows():
        old = os.path.join(TIFPATH, row['old'])
        new = os.path.join(FIXEDPATH, row['new'])
        shutil.move(old, new)
        n += 1
        if n % 500 == 0:
            print('Copied {}'.format(n))


if __name__ == '__main__':
    main()
