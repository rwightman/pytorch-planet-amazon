import torch
import torch.nn.functional as F


def pooling_factor(pool_type='avg'):
    return 2 if pool_type == 'avgmax' else 1


def adaptive_avgmax_pool(x, pool_type='avg', padding=0, count_include_pad=False):
    if pool_type == 'avgmax':
        x = torch.cat([
            F.avg_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad),
            F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        ], dim=1)
    elif pool_type == 'max':
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
    else:
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
    return x
