import argparse
import os
import time
import cv2
import numpy as np
import pandas as pd
from dataset import AmazonDataset, get_tags
from utils import AverageMeter
import torch
import torch.autograd as autograd
import torch.utils.data as data
from models import create_model

parser = argparse.ArgumentParser(description='PyTorch Sealion count inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='countception', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--multi-label', action='store_true', default=False,
                    help='multi-label target')
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


def main():
    args = parser.parse_args()

    batch_size = args.batch_size
    img_size = (args.img_size, args.img_size)
    num_classes = 17

    dataset = AmazonDataset(
        args.data,
        train=False,
        multi_label=args.multi_label,
        tags_type='all',
        img_type='.jpg',
        img_size=img_size,
    )

    tags = get_tags()
    output_col = ['image_name'] + tags
    submission_col = ['image_name', 'tags']

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_processes)

    model = create_model(args.model, pretrained=False, num_classes=num_classes)

    if not args.no_cuda:
        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        else:
            model.cuda()

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

    model.eval()

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    results = []
    results_thresh = []
    results_sub = []
    try:
        end = time.time()
        for batch_idx, (input, target, index) in enumerate(loader):
            data_time_m.update(time.time() - end)
            if not args.no_cuda:
                input = input.cuda()
            input_var = autograd.Variable(input)
            output = model(input_var)
            output = torch.sigmoid(output)
            if isinstance(threshold, torch.FloatTensor) or isinstance(threshold, torch.cuda.FloatTensor):
                threshold_m = torch.unsqueeze(threshold, 0).expand_as(output.data)
                output_thresh = (output.data > threshold_m).byte()
            else:
                output_thresh = (output.data > threshold).byte()
            output = output.cpu().data.numpy()
            output_thresh = output_thresh.cpu().numpy()
            index = index.cpu().numpy().flatten()
            for i, o, ot in zip(index, output, output_thresh):
                #print(dataset.inputs[i], o, ot)
                image_name = os.path.splitext(os.path.basename(dataset.inputs[i]))[0]
                results.append([image_name] + list(o))
                results_thresh.append([image_name] + list(ot))
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
    results_df = pd.DataFrame(results, columns=output_col)
    results_df.to_csv('output.csv', index=False)
    results_thresh_df = pd.DataFrame(results_thresh, columns=output_col)
    results_thresh_df.to_csv('output_thresh.csv', index=False)
    results_sub_df = pd.DataFrame(results_sub, columns=submission_col)
    results_sub_df.to_csv('submission.csv', index=False)


def vector_to_tags(v, tags):
    idx = np.nonzero(v)
    t = [tags[i] for i in idx[0]]
    return ' '.join(t)

if __name__ == '__main__':
    main()
