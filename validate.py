import argparse
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from dataset import AmazonDataset, get_tags_size, get_tags
from utils import AverageMeter, get_outdir
from sklearn.metrics import fbeta_score

import torch
import torch.nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.data as data
import torch.optim as optim
import torchvision.utils
from models import create_model

parser = argparse.ArgumentParser(description='PyTorch Amazon satellite training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--loss', default='nll', type=str, metavar='LOSS',
                    help='Loss function (default: "nll"')
parser.add_argument('--multi-label', action='store_true', default=True,
                    help='Multi-label target')
parser.add_argument('--no-multi-label', action='store_false', dest='multi_label', default=False,
                    help='No multi-label target')
parser.add_argument('--gp', default='avg', type=str, metavar='POOL',
                    help='Type of global pool, "avg", "max", "avgmax"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument('--tif', action='store_true', default=False,
                    help='Use tif dataset')
parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='Train/valid fold #. (default: 0')
parser.add_argument('--labels', default='all', type=str, metavar='NAME',
                    help='Label set (default: "all"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--restore-checkpoint', default=None,
                    help='path to restore checkpoint, e.g. ./checkpoint-1.tar')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')
parser.add_argument('--train', action='store_true', default=False,
                    help='Run on training data')

def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data)
    train_labels_file = './data/labels.csv'
    output_dir = get_outdir('./output', 'eval', datetime.now().strftime("%Y%m%d-%H%M%S"))

    batch_size = args.batch_size
    num_epochs = 1000
    if args.tif:
        img_type = '.tif'
    else:
        img_type = '.jpg'
    img_size = (args.img_size, args.img_size)
    num_classes = get_tags_size(args.labels)
    debug_model = False

    torch.manual_seed(args.seed)

    if args.train:
        dataset_train = AmazonDataset(
            train_input_root,
            train_labels_file,
            train=False,
            train_fold=True,
            tags_type=args.labels,
            multi_label=args.multi_label,
            img_type=img_type,
            img_size=img_size,
            fold=args.fold,
        )

        loader_train = data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_processes
        )

    dataset_eval = AmazonDataset(
        train_input_root,
        train_labels_file,
        train=False,
        tags_type=args.labels,
        multi_label=args.multi_label,
        img_type=img_type,
        img_size=img_size,
        test_aug=args.tta,
        fold=args.fold,
    )

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_processes
    )

    model = create_model(args.model, pretrained=args.pretrained, num_classes=num_classes, global_pool=args.gp)

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model.cuda()

    if False:
        class_weights = torch.from_numpy(dataset_train.get_class_weights()).float()
        class_weights_norm = class_weights / class_weights.sum()
        if not args.no_cuda:
            class_weights = class_weights.cuda()
            class_weights_norm = class_weights_norm.cuda()
    else:
        class_weights = None
        class_weights_norm = None

    if args.loss.lower() == 'nll':
        #assert not args.multi_label and 'Cannot use crossentropy with multi-label target.'
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss.lower() == 'mlsm':
        assert args.multi_label
        loss_fn = torch.nn.MultiLabelSoftMarginLoss(weight=class_weights)
    else:
        assert False and "Invalid loss function"

    if not args.no_cuda:
        loss_fn = loss_fn.cuda()

    # load a checkpoint
    if args.restore_checkpoint is not None:
        assert os.path.isfile(args.restore_checkpoint), '%s not found' % args.restore_checkpoint
        checkpoint = torch.load(args.restore_checkpoint)
        print(checkpoint['arch'])
        model.load_state_dict(checkpoint['state_dict'])
        if 'threshold' in checkpoint:
            threshold = checkpoint['threshold']
            threshold = torch.FloatTensor(threshold)
            print('Using thresholds:', threshold)
            if not args.no_cuda:
                threshold = threshold.cuda()
        else:
            threshold = 0.5
        print('Model restored from file: %s' % args.restore_checkpoint)
    else:
        assert False and "No checkpoint specified"

    if args.train:
        print('Validating training data...')
        validate(
            model, loader_train, loss_fn, args, threshold, prefix='train', output_dir=output_dir)

    print('Validating validation data...')
    validate(
        model, loader_eval, loss_fn, args, threshold, prefix='eval', output_dir=output_dir)


def validate(model, loader, loss_fn, args, threshold, prefix='', output_dir=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    acc_m = AverageMeter()
    f2_m = AverageMeter()

    model.eval()

    end = time.time()
    index_list = []
    output_list = []
    output_thr_list = []
    target_list = []
    for batch_idx, (input, target, index) in enumerate(loader):
        if not args.no_cuda:
            input, target = input.cuda(), target.cuda()
        if args.multi_label and args.loss == 'nll':
            # pick one of the labels for validation loss, should we randomize like in train?
            target_var = autograd.Variable(target.max(dim=1)[1].squeeze())
        else:
            target_var = autograd.Variable(target, volatile=True)
        input_var = autograd.Variable(input, volatile=True)

        if args.save_batches:
            torchvision.utils.save_image(
                input,
                os.path.join(output_dir, 'input-batch-%d.jpg' % batch_idx),
                padding=0,
                normalize=True)

        # compute output
        output = model(input_var)

        # augmentation reduction
        reduce_factor = loader.dataset.get_aug_factor()
        if reduce_factor > 1:
            output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2).squeeze(dim=2)
            target_var.data = target_var.data[0:target_var.size(0):reduce_factor]
            index = index[0:index.size(0):reduce_factor]

        # output non-linearities, thresholding, and metrics
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], output.size(0))
        if isinstance(threshold, torch.FloatTensor) or isinstance(threshold, torch.cuda.FloatTensor):
            threshold_b = torch.unsqueeze(threshold, 0).expand_as(output.data)
        else:
            threshold_b = threshold
        output_thr = (output.data > threshold_b).byte()
        if args.multi_label:
            if args.loss == 'nll':
                output = F.softmax(output)
            else:
                output = torch.sigmoid(output)
            a, p, _, f2 = scores(output.data, target_var.data, threshold_b)
            acc_m.update(a, output.size(0))
            prec1_m.update(p, output.size(0))
            f2_m.update(f2, output.size(0))
        else:
            prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 3))
            prec1_m.update(prec1[0], output.size(0))
            prec5_m.update(prec5[0], output.size(0))

        # copy data to CPU and collect
        output_thr_list.append(output_thr.cpu().numpy())
        output_list.append(output.data.cpu().numpy())
        target_list.append(target_var.data.cpu().numpy())
        index_list.append(index.cpu().numpy().flatten())

        batch_time_m.update(time.time() - end)
        end = time.time()
        if batch_idx % args.print_freq == 0:
            if args.multi_label:
                print('Test ({0}): [{1}/{2}]  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Acc {acc.val:.4f} ({acc.avg:.4f})  '
                      'Prec {prec.val:.4f} ({prec.avg:.4f})  '
                      'F2 {f2.val:.4f} ({f2.avg:.4f})  '.format(
                    prefix, batch_idx, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    acc=acc_m, prec=prec1_m, f2=f2_m))
            else:
                print('Test ({0}): [{1}/{2}]  '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                      'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    prefix, batch_idx, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    top1=prec1_m, top5=prec5_m))

    index_total = np.concatenate(index_list, axis=0)
    if args.multi_label:
        output_raw_total = np.vstack(output_list)
        output_thr_total = np.vstack(output_thr_list)
        target_total = np.vstack(target_list)
        print(output_raw_total.shape, target_total.shape)
        new_threshold, f2 = optimise_f2_thresholds(target_total, output_raw_total)
        print(f2, new_threshold)
        score = f2
    else:
        output_raw_total = np.concatenate(output_list, axis=0)
        target_total = np.concatenate(target_list, axis=0)
        print(output_raw_total.shape, target_total.shape)
        f2 = f2_score(output_raw_total, target_total, threshold=0.5)
        print(f2)
        score = prec1_m.val
        new_threshold = []

    if prefix:
        prefix += '_'
    tags = get_tags()
    output_col = ['image_name'] + tags
    results_raw_df = pd.DataFrame(output_raw_total, index=index_total, columns=tags)
    image_name_col = results_raw_df.index.map(
        lambda x: os.path.splitext(os.path.basename(loader.dataset.inputs[x]))[0])
    results_raw_df['image_name'] = image_name_col
    results_raw_df.to_csv(
        os.path.join(output_dir, '%sresults_raw.csv' % prefix), index=False, columns=output_col)

    results_thr_df = pd.DataFrame(output_thr_total, index=index_total, columns=tags)
    results_thr_df['image_name'] = image_name_col
    results_thr_df.to_csv(
        os.path.join(output_dir, '%sresults_thr.csv' % prefix), index=False, columns=output_col)

    return score, new_threshold


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def scores(output, target, threshold=0.5):
    # Count true positives, true negatives, false positives and false negatives.
    outputr = (output > threshold).long()
    target = target.long()
    a_sum = 0.0
    p_sum = 0.0
    r_sum = 0.0
    f2_sum = 0.0

    def _safe_size(t, n=0):
        if n < len(t.size()):
            return t.size(n)
        else:
            return 0

    count = 0
    for o, t in zip(outputr, target):
        tp = _safe_size(torch.nonzero(o * t))
        tn = _safe_size(torch.nonzero((o - 1) * (t - 1)))
        fp = _safe_size(torch.nonzero(o * (t - 1)))
        fn = _safe_size(torch.nonzero((o - 1) * t))
        a = (tp + tn) / (tp + fp + fn + tn)
        if tp == 0 and fp == 0 and fn == 0:
            p = 1.0
            r = 1.0
            f2 = 1.0
        elif tp == 0 and (fp > 0 or fn > 0):
            p = 0.0
            r = 0.0
            f2 = 0.0
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f2 = (5 * p * r) / (4 * p + r)
        a_sum += a
        p_sum += p
        r_sum += r
        f2_sum += f2
        count += 1
    accuracy = a_sum / count
    precision = p_sum / count
    recall = r_sum / count
    fmeasure = f2_sum / count
    return accuracy, precision, recall, fmeasure


def f2_score(output, target, threshold):
    output = (output > threshold)
    return fbeta_score(target, output, beta=2, average='samples')


def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
    """ Find optimal threshold values for f2 score. Thanks Anokas
    https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
    """
    size = y.shape[1]

    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(size):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2] * size
    for i in range(size):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)

    return x, best_score


if __name__ == '__main__':
    main()
