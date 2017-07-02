import numbers
import math
import numpy as np
import os
from sklearn.feature_extraction.image import extract_patches
from contextlib import contextmanager


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


@contextmanager
def measure_time(title='unknown'):
    t1 = time.clock()
    yield
    t2 = time.clock()
    print('%s: %0.2f seconds elapsed' % (title, t2-t1))


def calc_crop_size(target_w, target_h, angle, scale):
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
    #print(crop_w, crop_h)
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


def calc_num_patches(img_w, img_h, patch_size, stride):
    if isinstance(patch_size, numbers.Number):
        pw = ph = patch_size
    else:
        pw, ph = patch_size
    patches_rows = (img_h - ph) // stride + 1
    patches_cols = (img_w - pw) // stride + 1
    return patches_cols * patches_rows, patches_cols, patches_rows


def index_to_rc(index, ncols):
    row = index // ncols
    col = index - ncols * row
    return col, row


def rc_to_index(row, col, ncols):
    return row * ncols + col


def merge_patches(output_img, patches, patches_cols, patch_size, stride, agg_fn='mean'):
    # This is INCREDIBLY slow in pure Python. There is likely a better approach, but in
    # lieu of that, the Cython version in utils_cython is fast enough for this purpose.
    oh, ow = output_img.shape[:2]
    if isinstance(patch_size, numbers.Number):
        pw = ph = patch_size, patch_size
    else:
        pw, ph = patch_size
    oh = (oh - ph) // stride * stride + ph
    ow = (ow - pw) // stride * stride + pw
    patches_rows = patches.shape[0] // patches_cols
    print(patches_rows, patches_cols, oh, ow, patches.shape)
    for y in range(0, oh):
        pjl = max((y - ph) // stride + 1, 0)
        pju = min(y // stride + 1, patches_rows)
        for x in range(0, ow):
            pil = max((x - pw) // stride + 1,  0)
            piu = min(x // stride + 1, patches_cols)
            agg = np.zeros(output_img.shape[-1], dtype=np.uint32)
            agg_count = 0
            for pj in range(pjl, pju):
                for pi in range(pil, piu):
                    px = x - pi * stride
                    py = y - pj * stride
                    agg += patches[pi + pj * patches_cols][py, px, :]
                    agg_count += 1
            pa = agg // agg_count
            output_img[y, x, :] = pa.astype(output_img.dtype)


def patch_view(input_img, patch_size, stride, flatten=True):
    num_chan = input_img.shape[-1]
    if isinstance(patch_size, numbers.Number):
        patch_shape = (patch_size, patch_size, num_chan)
    else:
        patch_shape = (patch_size[1], patch_size[0], num_chan)
    # shape should be (h, w, c)
    assert patch_shape[-1] == input_img.shape[-1]
    patches = extract_patches(input_img, patch_shape, stride)
    patch_rowcol = patches.shape[:2]
    if flatten:
        # Note, this causes data in view to be copied to a new array
        patches = patches.reshape([-1] + list(patch_shape))
    return patches, patch_rowcol


def get_outdir(path, *paths):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
