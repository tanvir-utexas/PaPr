# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
import sys
import builtins
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
import pprint
from datasets import build_dataset
from engine import evaluate
from samplers import RASampler
from functools import partial

from models.papr_convnext import PaPrConvNeXt
from models.papr_swin import PaPrSwinTransformer

import utils
from calc_flops import calc_flops, throughput

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='deit_small', type=str, help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default='', help='resume from checkpoint')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--z', type=float, default=1.0)
    parser.add_argument('--profiling', action='store_true', default=False)
    parser.add_argument('--mask_block', type=int, default=4)

    parser.add_argument('--work_dir', default='./work_dir', type=str,
                        help='result path')
    parser.add_argument('--mobileone_weights', default='./mobileone_weights', type=str,
                        help='result path')
    parser.add_argument('--in22k', default=False, action='store_true')
    parser.add_argument('--cnn_size', default=None, type=int, help='images input size')
    parser.add_argument('--distilled', default=False, action='store_true')

    return parser


def main(args):

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    log_file = os.path.join(args.work_dir, 'evaluation.log')

    if os.path.isfile(log_file):
        os.remove(log_file)

    def print_and_log(*content, **kwargs):
        msg = ' '.join([str(ct) for ct in content])
        sys.stdout.write(msg+'\n')
        sys.stdout.flush()
        with open(log_file, 'a') as f:
            f.write(msg+'\n')

    builtins.print = print_and_log

    print("All arguments are given below:")
    
    all_args = vars(args)
    
    for k, v in all_args.items():
        print(f"{k}: {v}")

    print("#"*100)
    print("\n")
    

    if args.mask_block == 1:
        mask = [2]
    elif args.mask_block == 2:
        mask = [2, 3]
    elif args.mask_block == 3:
        mask = [1, 2, 3]
    elif args.mask_block == 4:
        mask = [0, 1, 2, 3]

    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")

    if args.model == 'papr_convnext-b':
        model = PaPrConvNeXt(
            fraction=args.z, mask_block=mask,
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024],
            mobileone_weights=args.mobileone_weights,
            cnn_size = args.input_size
        )
        if args.input_size == 224:
            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'convnext_base_22k_1k_224.pth')
            else:
                args.model_path = os.path.join(args.model_path, 'convnext_base_1k_224_ema.pth')
        else:
            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'convnext_base_22k_1k_384.pth')
            else:
                args.model_path = os.path.join(args.model_path, 'convnext_base_1k_384.pth')


    elif args.model == 'papr_convnext-l':
        model = PaPrConvNeXt(
            fraction=args.z, mask_block=mask,
            depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536],
            mobileone_weights=args.mobileone_weights,
            cnn_size=args.input_size
        )

        if args.input_size == 224:
            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'convnext_large_22k_1k_224.pth')
            else:
                args.model_path = os.path.join(args.model_path, 'convnext_large_1k_224_ema.pth')
        else:
            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'convnext_large_22k_1k_384.pth')
            else:
                args.model_path = os.path.join(args.model_path, 'convnext_large_1k_384.pth')

    elif args.model == 'papr_swin-b':
        if args.input_size == 224:
            model = PaPrSwinTransformer(
                img_size=args.input_size,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                drop_rate=0.0,
                drop_path_rate=args.drop_path,
                fraction = args.z,
                mask_block=mask,
                mobileone_weights=args.mobileone_weights,
                cnn_size=args.input_size
            )
            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'swin_base_patch4_window7_224_22kto1k.pth')
            else:
                args.model_path = os.path.join(args.model_path, 'swin_base_patch4_window7_224.pth')
        else:
            model = PaPrSwinTransformer(
                img_size=args.input_size,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                drop_rate=0.0,
                drop_path_rate=args.drop_path,
                fraction = args.z,
                mask_block=mask,
                mobileone_weights=args.mobileone_weights,
                cnn_size=args.input_size
            )


            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'swin_base_patch4_window12_384_22kto1k.pth')
            else:
                args.model_path = os.path.join(args.model_path, 'swin_base_patch4_window12_384.pth')
    
    elif args.model == 'papr_swin-l':
        if args.input_size == 224:
            model = PaPrSwinTransformer(
                img_size=args.input_size,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=7,
                drop_rate=0.0,
                drop_path_rate=args.drop_path,
                fraction = args.z,
                mask_block=mask,
                mobileone_weights=args.mobileone_weights,
                cnn_size=args.input_size
            )

            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'swin_large_patch4_window7_224_22kto1k.pth')
            else:
                raise Exception("Sorry, only ImageNet22k Pretraining is available.")
        else:
            model = PaPrSwinTransformer(
                img_size=args.input_size,
                embed_dim=192,
                depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48],
                window_size=12,
                drop_rate=0.0,
                drop_path_rate=args.drop_path,
                fraction = args.z,
                mask_block=mask,
                mobileone_weights=args.mobileone_weights,
                cnn_size=args.input_size
            )

            if args.in22k:
                args.model_path = os.path.join(args.model_path, 'swin_large_patch4_window12_384_22kto1k.pth')
            else:
                raise Exception("Sorry, only ImageNet22k Pretraining is available.")

    else:
        raise NotImplementedError

    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.update()
    print('## model has been successfully loaded')

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    if args.profiling:
        flops = calc_flops(model, args.input_size, True)
        print('FLOPs: {}'.format(flops))
        
        throughput(model, args.input_size)
        return

    criterion = torch.nn.CrossEntropyLoss().cuda()
    acc1, acc5 = validate(data_loader_val, model, criterion)

    flops = calc_flops(model, args.input_size)
    print('FLOPs: {}'.format(flops))
    
    thr, memory = throughput(model, args.input_size, batch_size=64)

    data = {'Acc@1': acc1, 'Acc@5': acc5, 'Flops(G)': flops, 'Throughput(Img/s)': thr, 'Memory(MB)': memory}

    with open(os.path.join(args.work_dir, 'results.json'), 'w') as f:
        json.dump(data, f, indent=4)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg.item(), top5.avg.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)