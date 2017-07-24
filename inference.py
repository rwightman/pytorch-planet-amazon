import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
from dataset import AmazonDataset, get_tags
from utils import AverageMeter, get_outdir
import torch
import torch.autograd as autograd
import torch.utils.data as data
from models import create_model, dense_sparse_dense

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='countception', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
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
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: 224)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('-r', '--restore-checkpoint', default=None,
                    help='path to restore checkpoint, e.g. ./checkpoint-1.tar')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')


def main():
    args = parser.parse_args()

    batch_size = args.batch_size
    img_size = (args.img_size, args.img_size)
    num_classes = 17
    if args.tif:
        img_type = '.tif'
    else:
        img_type = '.jpg'

    dataset = AmazonDataset(
        args.data,
        train=False,
        multi_label=args.multi_label,
        tags_type='all',
        img_type=img_type,
        img_size=img_size,
        test_aug=args.tta,
    )

    tags = get_tags()
    output_col = ['image_name'] + tags
    submission_col = ['image_name', 'tags']

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_processes)

    model = create_model(args.model, pretrained=False, num_classes=num_classes, global_pool=args.gp)

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model.cuda()

    if args.restore_checkpoint is not None:
        assert os.path.isfile(args.restore_checkpoint), '%s not found' % args.restore_checkpoint
        checkpoint = torch.load(args.restore_checkpoint)
        print('Restoring model with %s architecture...' % checkpoint['arch'])
        sparse_checkpoint = True if 'sparse' in checkpoint and checkpoint['sparse'] else False
        if sparse_checkpoint:
            print("Loading sparse model")
            dense_sparse_dense.sparsify(model, sparsity=0.)  # ensure sparsity_masks exist in model definition
        model.load_state_dict(checkpoint['state_dict'])
        if 'args' in checkpoint:
            train_args = checkpoint['args']
        if 'threshold' in checkpoint:
            threshold = checkpoint['threshold']
            threshold = torch.FloatTensor(threshold)
            print('Using thresholds:', threshold)
            if not args.no_cuda:
                threshold = threshold.cuda()
        else:
            threshold = 0.5
        if 'gp' in checkpoint and checkpoint['gp'] != args.gp:
            print("Warning: Model created with global pooling (%s) different from checkpoint (%s)"
                  % (args.gp, checkpoint['gp']))
        csplit = os.path.normpath(args.restore_checkpoint).split(sep=os.path.sep)
        if len(csplit) > 1:
            exp_name = csplit[-2] + '-' + csplit[-1].split('.')[0]
        else:
            exp_name = ''
        print('Model restored from file: %s' % args.restore_checkpoint)
    else:
        assert False and "No checkpoint specified"

    if args.output:
        output_base = args.output
    else:
        output_base = './output'
    if not exp_name:
        exp_name = '-'.join([
            args.model,
            str(train_args.img_size),
            'f'+str(train_args.fold),
            'tif' if args.tif else 'jpg'])
    output_dir = get_outdir(output_base, 'predictions', exp_name)

    model.eval()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    results_raw = []
    results_thr = []
    results_sub = []
    try:
        end = time.time()
        for batch_idx, (input, target, index) in enumerate(loader):
            data_time_m.update(time.time() - end)
            if not args.no_cuda:
                input = input.cuda()
            input_var = autograd.Variable(input, volatile=True)
            output = model(input_var)

            # augmentation reduction
            reduce_factor = loader.dataset.get_aug_factor()
            if reduce_factor > 1:
                output.data = output.data.unfold(0, reduce_factor, reduce_factor).mean(dim=2).squeeze(dim=2)
                index = index[0:index.size(0):reduce_factor]

            # output non-linearity and thresholding
            output = torch.sigmoid(output)
            if isinstance(threshold, torch.FloatTensor) or isinstance(threshold, torch.cuda.FloatTensor):
                threshold_m = torch.unsqueeze(threshold, 0).expand_as(output.data)
                output_thr = (output.data > threshold_m).byte()
            else:
                output_thr = (output.data > threshold).byte()

            # move data to CPU and collect
            output = output.cpu().data.numpy()
            output_thr = output_thr.cpu().numpy()
            index = index.cpu().numpy().flatten()
            for i, o, ot in zip(index, output, output_thr):
                #print(dataset.inputs[i], o, ot)
                image_name = os.path.splitext(os.path.basename(dataset.inputs[i]))[0]
                results_raw.append([image_name] + list(o))
                results_thr.append([image_name] + list(ot))
                results_sub.append([image_name] + [vector_to_tags(ot, tags)])
                # end iterating through batch

            batch_time_m.update(time.time() - end)
            if batch_idx % args.log_interval == 0:
                print('Inference: [{}/{} ({:.0f}%)]  '
                      'Time: {batch_time.val:.3f}s, {rate:.3f}/s  '
                      '({batch_time.avg:.3f}s, {rate_avg:.3f}/s)  '
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    batch_idx * len(input), len(loader.sampler),
                    100. * batch_idx / len(loader),
                    batch_time=batch_time_m,
                    rate=input_var.size(0) / batch_time_m.val,
                    rate_avg=input_var.size(0) / batch_time_m.avg,
                    data_time=data_time_m))

            end = time.time()
            #end iterating through dataset
    except KeyboardInterrupt:
        pass
    results_raw_df = pd.DataFrame(results_raw, columns=output_col)
    results_raw_df.to_csv(os.path.join(output_dir, 'results_raw.csv'), index=False)
    results_thr_df = pd.DataFrame(results_thr, columns=output_col)
    results_thr_df.to_csv(os.path.join(output_dir, 'results_thr.csv'), index=False)
    results_sub_df = pd.DataFrame(results_sub, columns=submission_col)
    results_sub_df.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)


def vector_to_tags(v, tags):
    idx = np.nonzero(v)
    t = [tags[i] for i in idx[0]]
    return ' '.join(t)

if __name__ == '__main__':
    main()
