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
        # Remove the following lines to unleash the full power of
        # this kernel and copy & rename all the misnamed tif files!

    jpg_images = find_images(JPGPATH)
    tif_images = find_images(TIFPATH, types=['.tif'])
    jpg_set = {x[0] for x in jpg_images}
    tif_mismatch = [x for x in tif_images if x[0] not in jpg_set]
    print(len(tif_mismatch))
    for x in tif_mismatch:
        shutil.move(x[1], os.path.join(BADPATH, x[0] + '.tif'))

    #FIXME move fixed-test back into original test folder

if __name__ == '__main__':
    main()
