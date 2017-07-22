import numbers
import math
import numpy as np
import os


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_crop_size(target_w, target_h, angle=0.0, scale=1.0):
    crop_w = target_w
    crop_h = target_h
    if angle:
        corners = np.array(
            [[target_w/2, -target_w/2, -target_w/2, target_w/2],
            [target_h/2, target_h/2, -target_h/2, -target_h/2]])
        s = np.sin(angle * np.pi/180)
        c = np.cos(angle * np.pi/180)
        M = np.array([[c, -s], [s, c]])
        rotated_corners = np.dot(M, corners)
        crop_w = 2 * np.max(np.abs(rotated_corners[0, :]))
        crop_h = 2 * np.max(np.abs(rotated_corners[1, :]))
    crop_w = int(np.ceil(crop_w / scale))
    crop_h = int(np.ceil(crop_h / scale))
    return crop_w, crop_h


def crop_center(img, cx, cy, crop_w, crop_h):
    img_h, img_w = img.shape[:2]
    trunc_top = trunc_bottom = trunc_left = trunc_right = 0
    left = cx - crop_w//2
    if left < 0:
        trunc_left = 0 - left
        left = 0
    right = left - trunc_left + crop_w
    if right > img_w:
        trunc_right = right - img_w
        right = img_w
    top = cy - crop_h//2
    if top < 0:
        trunc_top = 0 - top
        top = 0
    bottom = top - trunc_top + crop_h
    if bottom > img_h:
        trunc_bottom = bottom - img_h
        bottom = img_h
    if trunc_left or trunc_right or trunc_top or trunc_bottom:
        img_new = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)
        trunc_bottom = crop_h - trunc_bottom
        trunc_right = crop_w - trunc_right
        img_new[trunc_top:trunc_bottom, trunc_left:trunc_right] = img[top:bottom, left:right]
        return img_new
    else:
        return img[top:bottom, left:right]


def crop_points_center(points, cx, cy, crop_w, crop_h):
    xl = cx - crop_w // 2
    xu = xl + crop_w
    yl = cy - crop_h // 2
    yu = yl + crop_h
    mask = (points[:, 0] >= xl) & (points[:, 0] < xu) & (points[:, 1] >= yl) & (points[:, 1] < yu)
    return points[mask]


def crop_points(points, x, y, crop_w, crop_h):
    xu = x + crop_w
    yu = y + crop_h
    mask = (points[:, 0] >= x) & (points[:, 0] < xu) & (points[:, 1] >= y) & (points[:, 1] < yu)
    return points[mask]


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir
