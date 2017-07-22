import os
import cv2
import numpy as np


BASEPATH = '/data/amazon'
JPGPATH = os.path.join(BASEPATH, 'test-jpg')
TIFPATH = os.path.join(BASEPATH, 'old-test-tif-v2')
PREFIX='test'


def find_inputs(folder, types=('.jpg', '.tif'), prefix=''):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if prefix and base.startswith(prefix) and ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                inputs.append((base, abs_filename))
    return inputs


def main():

    jpg_inputs = find_inputs(JPGPATH, types=('.jpg',), prefix=PREFIX)
    tif_inputs = find_inputs(TIFPATH, types=('.tif',), prefix=PREFIX)

    jpg_stats = []
    for f in jpg_inputs:
        img = cv2.imread(f[1])
        mean, std = cv2.meanStdDev(img)
        jpg_stats.append(np.array([mean[::-1] / 255, std[::-1] / 255]))
    jpg_vals = np.mean(jpg_stats, axis=0)
    print(jpg_vals)

    tif_stats = []
    for f in tif_inputs:
        img = cv2.imread(f[1], -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        mean, std = cv2.meanStdDev(img)
        tif_stats.append(np.array([mean, std]))
    tif_vals = np.mean(tif_stats, axis=0)
    print(tif_vals)


if __name__ == '__main__':
    main()

