import argparse
import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models.resnet20 import resnet20
from models.resnet32 import resnet32
import sys

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-path', dest='evaluate_path',
                    help='The path used to load the trained model',
                    default='save_temp', type=str)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--seed', default=0, type=int,
                    help='random seed of numpy,random,torch')
parser.add_argument('--arch', dest='arch',
                    help='The architecture to be trained',
                    default='resnet20', type=str)
parser.add_argument('--Ther_len', default=32, type=int, metavar='N',
                    help='the length of ther coding')
best_prec1 = 0
print('python  ' + '  '.join(sys.argv))


def main():
    global args, best_prec1, flag_matrix_RGB
    args = parser.parse_args()
    flag_matrix_RGB = Flag_Matrix_RGB()
    flag_matrix_RGB = flag_matrix_RGB.cuda()
    set_random_seed(args.seed)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.arch == 'resnet20':
        model = torch.nn.DataParallel(resnet20())
    elif args.arch == 'resnet32':
        model = torch.nn.DataParallel(resnet32())
    else:
        print("No such arch {}".format(args.arch))
        exit()
    model.cuda()
    writer = SummaryWriter(os.path.join(args.save_dir,'log'))
    print(model.module)
    print(args)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
    #        normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
    #        normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr': args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[200, 260, 320 ], last_epoch=args.start_epoch - 1)  

    if args.evaluate:
        print("=> loading checkpoint '{}'".format(args.evaluate_path))
        checkpoint = torch.load(args.evaluate_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer,  epoch, writer)
        lr_scheduler.step()

        # evaluate on validation set
        if epoch%5==0 or epoch>110:
            prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        if epoch >100 and is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, filename=os.path.join(args.save_dir, 'model_best.th'))           

    save_checkpoint({
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, filename=os.path.join(args.save_dir, 'model.th'))


def train(train_loader, model, criterion, optimizer, epoch, writer):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    precess_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()

        input_var = input.cuda()
        input_var=Thermometer_Input(input_var)
        target_var = target

        end4 = time.time()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        precess_time.update(time.time() - end4)
        end = time.time()

    print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Process_time {precess_time.val:.3f} ({precess_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, precess_time=precess_time,
                loss=losses, top1=top1))
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc' , top1.avg, epoch)

best_val_acc = 0
def validate(val_loader, model, criterion, epoch=0, writer=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            input_var = Thermometer_Input(input_var)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    if writer is not None:
        writer.add_scalar('val_acc', top1.avg, epoch)
        writer.add_scalar('val_loss', losses.avg, epoch)
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    global best_val_acc
    if top1.avg > best_val_acc:
        best_val_acc = top1.avg
        if epoch >= 300:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_val_acc,
            }, filename=os.path.join(args.save_dir, 'model_best.th'))
    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    if epoch % 20 == 0 or epoch >= 299:
        print("best_val_acc:  ", best_val_acc)

    return top1.avg

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
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

def Flag_Matrix_RGB():
    W = H = 32
    global args
    Ther_len = int(args.Ther_len)
    Ther_advisor = int(256/Ther_len)
    flag_matrix = torch.zeros(int(Ther_len*3), W, H )

    for i in range(Ther_len):
        flag_matrix[ int(i*3):int(i*3+3), :, :] = int( (i+0.5)*Ther_advisor )
    flag_matrix /=255
    return flag_matrix

def Thermometer_Input(input):
    # input: [N, C, W, H]
    global args
    Ther_len = int(args.Ther_len)
    Ther_advisor = int(256/Ther_len)
    global flag_matrix_RGB

    input = input.repeat(1,Ther_len,1,1)
    thermometer_input = (input > flag_matrix_RGB).float()
 
    return thermometer_input

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

def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    main()
