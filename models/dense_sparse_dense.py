""" Dense-Sparse-Dense Training
Trying ideas from https://arxiv.org/abs/1607.04381 by Song Han
"""
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
            wv = m.weight.data.view(-1)
            mask = torch.zeros(m.weight.size()).byte()
            if m.weight.is_cuda:
                mask = mask.cuda()
            k = int(math.floor(sparsity*wv.numel()))
            if k > 0:
                smallest_idx = wv.abs().topk(k, dim=0, largest=False)[1]
                mask.view(-1)[smallest_idx] = 1
                m.weight.data.masked_fill_(mask, 0.)
            m.register_buffer('sparsity_mask', mask)


def densify(module):
    for m in module.modules():
        if hasattr(m, 'sparsity_mask'):
            del m.sparsity_mask


def apply_sparsity_mask(module):
    for m in module.modules():
        if hasattr(m, 'sparsity_mask'):
            m.weight.data.masked_fill_(m.sparsity_mask, 0.)


