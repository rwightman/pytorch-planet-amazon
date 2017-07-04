import argparse
import os
import time
import shutil
import numpy as np
from datetime import datetime
from dataset import AmazonDataset, get_label_size
from utils import AverageMeter, get_outdir
from sklearn.metrics import fbeta_score

import torch
import torch.nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.utils.data as data
import torch.optim as optim
import torchvision.utils
from torchvision.models import *
from models import *

parser = argparse.ArgumentParser(description='PyTorch Sealion count training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--loss', default='nll', type=str, metavar='LOSS',
                    help='Loss function (default: "nll"')
parser.add_argument('--multi-label', action='store_true', default=False,
                    help='multi-label target')
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
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M',
                    help='weight decay (default: 0.0001)')
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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')


def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data)
    train_labels_file = './data/labels.csv'
    output_dir = get_outdir('./output', 'train', datetime.now().strftime("%Y%m%d-%H%M%S"))

    batch_size = args.batch_size
    num_epochs = 1000
    if args.tif:
        img_type = '.tif'
    else:
        img_type = '.jpg'
    img_size = (args.img_size, args.img_size)
    num_classes = get_label_size(args.labels)
    debug_model = False

    torch.manual_seed(args.seed)

    dataset_train = AmazonDataset(
        train_input_root,
        train_labels_file,
        train=True,
        label_type=args.labels,
        multi_label=args.multi_label,
        img_type=img_type,
        img_size=img_size,
        fold=args.fold,
    )

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_processes
    )

    dataset_eval = AmazonDataset(
        train_input_root,
        train_labels_file,
        train=False,
        label_type=args.labels,
        multi_label=args.multi_label,
        img_type=img_type,
        img_size=img_size,
        fold=args.fold,
    )

    loader_eval = data.DataLoader(
        dataset_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_processes
    )

    if args.model == 'resnet50':
        if args.pretrained:
            model = resnet50(pretrained=True)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = resnet50(num_classes=num_classes)
    elif args.model == 'resnet101':
        if args.pretrained:
            model = resnet101(pretrained=True)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = resnet101(num_classes=num_classes)
    elif args.model == 'resnet152':
        if args.pretrained:
            model = resnet152(pretrained=True)
            model.fc = torch.nn.Linear(2048, num_classes)
        else:
            model = resnet152(num_classes=num_classes)
    elif args.model == 'densenet121':
        if args.pretrained:
            model = densenet121(pretrained=True)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet121(num_classes=num_classes)
    elif args.model == 'densenet161':
        if args.pretrained:
            model = densenet161(pretrained=True)
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        else:
            model = densenet161(num_classes=num_classes)
    elif args.model == 'inception_resnet_v2':
        if args.pretrained:
            model = inception_resnet_v2(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = inception_resnet_v2(num_classes=num_classes)
        assert False and "Invalid model"
    elif args.model == 'inception_v4':
        if args.pretrained:
            model = inception_v4(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = inception_v4(num_classes=num_classes)
        assert False and "Invalid model"
    elif args.model == 'resnext_101_32x4d':
        activation_fn = torch.nn.LeakyReLU(0.1)
        if args.pretrained:
            model = resnext_101_32x4d(pretrained=True, activation_fn=activation_fn)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = resnext_101_32x4d(num_classes=num_classes, activation_fn=activation_fn)
    elif args.model == 'wrn_50_2':
        if args.pretrained:
            model = wrn_50_2(pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            model = wrn_50_2(num_classes=num_classes)
    else:
        assert False and "Invalid model"

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model.cuda()

    if args.opt.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'adadelta':
        optimizer = optim.Adadelta(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        assert False and "Invalid optimizer"

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

    # optionally resume from a checkpoint
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    best_score = None
    #threshold = np.array([0.5] * num_classes)
    threshold = 0.5 # pytorch gt broadcasting not working as expected
    for epoch in range(start_epoch, num_epochs + 1):
        adjust_learning_rate(optimizer, epoch, initial_lr=args.lr, decay_epochs=3)

        train_epoch(
            epoch, model, loader_train, optimizer, loss_fn, args, class_weights_norm, output_dir)

        score, latest_threshold = validate(
            model, loader_eval, loss_fn, args, threshold, output_dir)

        best = False
        if best_score is None or score > best_score:
            best_score = score
            best = True

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'state_dict':  model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'threshold': latest_threshold,
            },
            is_best=best,
            filename='checkpoint-%d.pth.tar' % epoch,
            output_dir=output_dir)


def train_epoch(epoch, model, loader, optimizer, loss_fn, args, class_weights=None, output_dir=''):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
        data_time_m.update(time.time() - end)
        if not args.no_cuda:
            input, target = input.cuda(), target.cuda()
        input_var = autograd.Variable(input)

        if args.multi_label and args.opt == 'nll':
            # if multi-label AND nll set, train network by sampling an index using class weights
            if class_weights is not None:
                target_weights = target * torch.unsqueeze(class_weights, 0).expand_as(target)
                target_weights = target_weights.div(target_weights.sum(dim=1).expand_as(target_weights))
            else:
                target_weights = target
            target_var = autograd.Variable(torch.multinomial(target_weights, 1).squeeze().long())
        else:
            target_var = autograd.Variable(target)

        output = model(input_var)
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input_var.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time_m.update(time.time() - end)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  '
                  'Loss: {loss.val:.6f} ({loss.avg:.4f})  '
                  'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                  '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                epoch,
                batch_idx * len(input), len(loader.sampler),
                100. * batch_idx / len(loader),
                loss=losses_m,
                batch_time=batch_time_m,
                rate=input_var.size(0) / batch_time_m.val,
                rate_avg=input_var.size(0) / batch_time_m.avg,
                data_time=data_time_m))

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'input-batch-%d.jpg' % batch_idx),
                    normalize=True)

        end = time.time()


def validate(model, loader, loss_fn, args, threshold, output_dir=''):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    acc_m = AverageMeter()
    f2_m = AverageMeter()

    #if isinstance(threshold, np.ndarray):
    #    threshold = torch.from_numpy(threshold).float()
    #    if not args.no_cuda:
    #        threshold = threshold.cuda()
    #FIXME broadcasting this doesn't flippin work for some reason

    model.eval()

    end = time.time()
    output_list = []
    target_list = []
    for i, (input, target, _) in enumerate(loader):
        if not args.no_cuda:
            input, target = input.cuda(), target.cuda()
        if args.multi_label and args.opt == 'nll':
            # pick one of the labels for validation loss, should we randomize like in train?
            target_var = autograd.Variable(target.max(dim=1)[1].squeeze())
        else:
            target_var = autograd.Variable(target, volatile=True)
        input_var = autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input.size(0))

        target_np = target.cpu().numpy()
        target_list.append(target_np)
        if args.multi_label:
            if args.opt == 'nll':
                output = F.softmax(output)
            else:
                output = torch.sigmoid(output)
            a, p, _, f2 = scores(output.data, target, threshold)
            acc_m.update(a, input.size(0))
            prec1_m.update(p, input.size(0))
            f2_m.update(f2, input.size(0))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
            prec1_m.update(prec1[0], input.size(0))
            prec5_m.update(prec5[0], input.size(0))
        output_np = output.data.cpu().numpy()
        output_list.append(output_np)

        batch_time_m.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if args.multi_label:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Acc {acc.val:.4f} ({acc.avg:.4f})  '
                      'Prec {prec.val:.4f} ({prec.avg:.4f})  '
                      'F2 {f2.val:.4f} ({f2.avg:.4f})  '.format(
                    i, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    acc=acc_m, prec=prec1_m, f2=f2_m))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                      'Prec@1 {top1.val:.4f} ({top1.avg:.4f})  '
                      'Prec@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                    i, len(loader),
                    batch_time=batch_time_m, loss=losses_m,
                    top1=prec1_m, top5=prec5_m))

    if args.multi_label:
        output_total = np.vstack(output_list)
        target_total = np.vstack(target_list)
        print(output_total.shape, target_total.shape)
        new_threshold, f2 = optimise_f2_thresholds(target_total, output_total)
        print(new_threshold)
        print(f2)
        return f2, new_threshold
    else:
        output_total = np.concatenate(output_list, axis=0)
        target_total = np.concatenate(target_list, axis=0)
        print(target_total.shape, output_total.shape)
        f2 = f2_score(output_total, target_total, threshold=0.5)
        print(f2)
        return prec1_m.val, []


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output_dir=''):
    save_path = os.path.join(output_dir, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, os.path.join(output_dir, 'model_best.pth.tar'))


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
