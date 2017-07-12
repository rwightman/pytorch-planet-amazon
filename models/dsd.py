import torch
import torch.nn as nn
import math


def is_sparseable(m):
    return True if hasattr(m, 'weight') and isinstance(m, (
            nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.Linear)) else False


def sparsify(module, sparsity=0.5):
    for m in module.modules():
        if is_sparseable(m):
            print(m)
            wv = m.weight.data.view(-1)
            k = int(math.floor(sparsity*wv.numel()))
            smallest_idx = wv.abs().topk(k, dim=0, largest=False)[1]
            mask = torch.zeros(m.weight.size()).byte()
            if m.weight.is_cuda:
                mask = mask.cuda()
            mask.view(-1)[smallest_idx] = 1
            m.register_buffer('sparsity_mask', mask)
            m.weight.data.masked_fill_(mask, 0.)


def densify(module):
    for m in module.modules():
        if hasattr(m, 'sparsity_mask'):
            del m.sparsity_mask


def apply_sparsity_mask(module):
    for m in module.modules():
        if hasattr(m, 'sparsity_mask'):
            m.weight.data.masked_fill_(m.sparsity_mask, 0.)


