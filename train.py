import argparse
import csv
import os
import shutil
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils
from sklearn.metrics import fbeta_score

from dataset import AmazonDataset, get_tags_size
from models import model_factory, dense_sparse_dense
from opt.yellowfin import YFOptimizer
from opt.lr_scheduler import ReduceLROnPlateau
from utils import AverageMeter, get_outdir

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


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
                    help='Type of global pool, "avg", "max", "avgmax (default: "avg")')
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
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--decay-epochs', type=int, default=15, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--ft-epochs', type=float, default=0., metavar='LR',
                    help='Number of finetuning epochs (final layer only)')
parser.add_argument('--ft-opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--ft-lr', type=float, default=0.01, metavar='N',
                    help='Finetune learning rates.')
parser.add_argument('--drop', type=float, default=0.1, metavar='DROP',
                    help='Dropout rate (default: 0.1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='M',
                    help='weight decay (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=1, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-tb', action='store_true', default=False,
                    help='disables tensorboard')
parser.add_argument('--tbh', default='127.0.0.1:8009', type=str, metavar='IP',
                    help='Tensorboard (Crayon) host')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-batches', action='store_true', default=False,
                    help='save images of batch inputs and targets every log interval for debugging/verification')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--sparse', action='store_true', default=False,
                    help='enable sparsity masking for DSD training')


def main():
    args = parser.parse_args()

    train_input_root = os.path.join(args.data)
    train_labels_file = './data/labels.csv'

    if args.output:
        output_base = args.output
    else:
        output_base = './output'

    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(args.img_size),
        'f'+str(args.fold),
        'tif' if args.tif else 'jpg'])
    output_dir = get_outdir(output_base, 'train', exp_name)

    batch_size = args.batch_size
    num_epochs = args.epochs
    img_type = '.tif' if args.tif else '.jpg'
    img_size = (args.img_size, args.img_size)
    num_classes = get_tags_size(args.labels)

    torch.manual_seed(args.seed)

    dataset_train = AmazonDataset(
        train_input_root,
        train_labels_file,
        train=True,
        tags_type=args.labels,
        multi_label=args.multi_label,
        img_type=img_type,
        img_size=img_size,
        fold=args.fold,
    )

    #sampler = WeightedRandomOverSampler(dataset_train.get_sample_weights())

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        #sampler=sampler,
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

    model = model_factory.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=num_classes,
        drop_rate=args.drop,
        global_pool=args.gp)

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
    elif args.opt.lower() == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, alpha=0.9, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt.lower() == 'yellowfin':
        optimizer = YFOptimizer(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, clip_thresh=2)
    else:
        assert False and "Invalid optimizer"

    if not args.decay_epochs:
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=8)

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
            sparse_checkpoint = True if 'sparse' in checkpoint and checkpoint['sparse'] else False
            if sparse_checkpoint:
                print("Loading sparse model")
                dense_sparse_dense.sparsify(model, sparsity=0.)  # ensure sparsity_masks exist in model definition
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            start_epoch = checkpoint['epoch']
            if args.sparse and not sparse_checkpoint:
                print("Sparsifying loaded model")
                dense_sparse_dense.sparsify(model, sparsity=0.5)
            elif sparse_checkpoint and not args.sparse:
                print("Densifying loaded model")
                dense_sparse_dense.densify(model)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(-1)
    else:
        if args.sparse:
            dense_sparse_dense.sparsify(model, sparsity=0.5)

    use_tensorboard = not args.no_tb and CrayonClient is not None
    if use_tensorboard:
        hostname = '127.0.0.1'
        port = 8889
        host_port = args.tbh.split(':')[:2]
        if len(host_port) == 1:
            hostname = host_port[0]
        elif len(host_port) >= 2:
            hostname, port = host_port[:2]
        try:
            cc = CrayonClient(hostname=hostname, port=port)
            try:
                cc.remove_experiment(exp_name)
            except ValueError:
                pass
            exp = cc.create_experiment(exp_name)
        except Exception as e:
            exp = None
            print("Error (%s) connecting to Tensoboard/Crayon server. Giving up..." % str(e))
    else:
        exp = None

    # Optional fine-tune of only the final classifier weights for specified number of epochs (or part of)
    if not args.resume and args.ft_epochs > 0.:
        if args.opt.lower() == 'adam':
            finetune_optimizer = optim.Adam(
                model.get_fc().parameters(), lr=args.ft_lr, weight_decay=args.weight_decay)
        else:
            finetune_optimizer = optim.SGD(
                model.get_fc().parameters(), lr=args.ft_lr, momentum=args.momentum, weight_decay=args.weight_decay)

        finetune_epochs_int = int(np.ceil(args.ft_epochs))
        finetune_final_batches = int(np.ceil((1 - (finetune_epochs_int - args.ft_epochs)) * len(loader_train)))
        print(finetune_epochs_int, finetune_final_batches)
        for fepoch in range(1, finetune_epochs_int + 1):
            if fepoch == finetune_epochs_int and finetune_final_batches:
                batch_limit = finetune_final_batches
            else:
                batch_limit = 0
            train_epoch(
                fepoch, model, loader_train, finetune_optimizer, loss_fn, args,
                class_weights_norm, output_dir, batch_limit=batch_limit)
            step = fepoch * len(loader_train)
            score, _ = validate(step, model, loader_eval, loss_fn, args, 0.3, output_dir)

    score_metric = 'f2'
    best_loss = None
    best_f2 = None
    threshold = 0.3
    try:
        for epoch in range(start_epoch, num_epochs + 1):
            if args.decay_epochs:
                adjust_learning_rate(optimizer, epoch, initial_lr=args.lr, decay_epochs=args.decay_epochs)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, loss_fn, args, class_weights_norm, output_dir, exp=exp)

            step = epoch * len(loader_train)
            eval_metrics, latest_threshold = validate(
                step, model, loader_eval, loss_fn, args, threshold, output_dir, exp=exp)

            if lr_scheduler is not None:
                lr_scheduler.step(eval_metrics['eval_loss'])

            rowd = OrderedDict(epoch=epoch)
            rowd.update(train_metrics)
            rowd.update(eval_metrics)
            with open(os.path.join(output_dir, 'summary.csv'), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=rowd.keys())
                if best_loss is None:  # first iteration (epoch == 1 can't be used)
                    dw.writeheader()
                dw.writerow(rowd)

            best = False
            if best_loss is None or eval_metrics['eval_loss'] < best_loss[1]:
                best_loss = (epoch, eval_metrics['eval_loss'])
                if score_metric == 'loss':
                    best = True
            if best_f2 is None or eval_metrics['eval_f2'] > best_f2[1]:
                best_f2 = (epoch, eval_metrics['eval_f2'])
                if score_metric == 'f2':
                    best = True

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'sparse': args.sparse,
                'state_dict':  model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'threshold': latest_threshold,
                'args': args,
                },
                is_best=best,
                filename='checkpoint-%d.pth.tar' % epoch,
                output_dir=output_dir)

    except KeyboardInterrupt:
        pass
    print('*** Best loss: {0} (epoch {1})'.format(best_loss[1], best_loss[0]))
    print('*** Best f2: {0} (epoch {1})'.format(best_f2[1], best_f2[0]))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        class_weights=None, output_dir='', exp=None, batch_limit=0):

    epoch_step = (epoch - 1) * len(loader)
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (input, target, index) in enumerate(loader):
        step = epoch_step + batch_idx
        data_time_m.update(time.time() - end)
        if not args.no_cuda:
            input, target = input.cuda(), target.cuda()
        input_var = autograd.Variable(input)

        if args.multi_label and args.loss == 'nll':
            # if multi-label AND nll set, train network by sampling an index using class weights
            if class_weights is not None:
                target_weights = target * torch.unsqueeze(class_weights, 0).expand_as(target)
                ss = target_weights.sum(dim=1, keepdim=True).expand_as(target_weights)
                target_weights = target_weights.div(ss)
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

        if args.sparse:
            dense_sparse_dense.apply_sparsity_mask(model)

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

            if exp is not None:
                exp.add_scalar_value('loss_train', losses_m.val, step=step)
                exp.add_scalar_value('learning_rate', optimizer.param_groups[0]['lr'], step=step)

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                    padding=0,
                    normalize=True)

        end = time.time()

        if batch_limit and batch_idx >= batch_limit:
            break

    return OrderedDict([('train_loss', losses_m.avg)])


def validate(step, model, loader, loss_fn, args, threshold, output_dir='', exp=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    acc_m = AverageMeter()
    f2_m = AverageMeter()

    model.eval()

    end = time.time()
    output_list = []
    target_list = []
    for i, (input, target, _) in enumerate(loader):
        if not args.no_cuda:
            input, target = input.cuda(), target.cuda()
        if args.multi_label and args.loss == 'nll':
            # pick one of the labels for validation loss, should we randomize like in train?
            target_var = autograd.Variable(target.max(dim=1)[1].squeeze(), volatile=True)
        else:
            target_var = autograd.Variable(target, volatile=True)
        input_var = autograd.Variable(input, volatile=True)

        # compute output
        output = model(input_var)

        # augmentation reduction
        reduce_factor = loader.dataset.get_aug_factor()
        if reduce_factor > 1:
            output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
            target_var.data = target_var.data[0:target_var.size(0):reduce_factor]

        # calc loss
        loss = loss_fn(output, target_var)
        losses_m.update(loss.data[0], input.size(0))

        # output non-linearities and metrics
        if args.multi_label:
            if args.loss == 'nll':
                output = F.softmax(output)
            else:
                output = torch.sigmoid(output)
            a, p, _, f2 = scores(output.data, target_var.data, threshold)
            acc_m.update(a, output.size(0))
            prec1_m.update(p, output.size(0))
            f2_m.update(f2, output.size(0))
        else:
            prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
            prec1_m.update(prec1[0], output.size(0))
            prec5_m.update(prec5[0], output.size(0))

        # copy to CPU and collect
        target_list.append(target.cpu().numpy())
        output_list.append(output.data.cpu().numpy())

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

            if args.save_batches:
                torchvision.utils.save_image(
                    input,
                    os.path.join(output_dir, 'validate-batch-%d.jpg' % i),
                    padding=0,
                    normalize=True)

    output_total = np.concatenate(output_list, axis=0)
    target_total = np.concatenate(target_list, axis=0)
    if args.multi_label:
        new_threshold, f2 = optimise_f2_thresholds(target_total, output_total)
        metrics = [('eval_loss', losses_m.avg), ('eval_f2', f2)]
    else:
        f2 = f2_score(output_total, target_total, threshold=0.5)
        new_threshold = []
        metrics = [('eval_loss', losses_m.avg), ('eval_f2', f2), ('eval_prec1', prec1_m.avg)]
    print(f2, new_threshold)

    if exp is not None:
        exp.add_scalar_value('loss_eval', losses_m.avg, step=step)
        exp.add_scalar_value('prec@1_eval', prec1_m.avg, step=step)
        exp.add_scalar_value('f2_eval', f2, step=step)

    return OrderedDict(metrics), new_threshold


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if isinstance(optimizer, YFOptimizer):
        return
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
