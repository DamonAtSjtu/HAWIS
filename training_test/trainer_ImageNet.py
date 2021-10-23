import os
import time
import argparse
from tqdm import tqdm
from PIL import ImageFile
from datetime import datetime
from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision
from models.resnet18 import resnet18
from models.resnet18_3x3 import resnet18_3x3


import sys

from utils_imgnet.utils import DisablePrint
from utils_imgnet.preprocessing import Lighting
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms


from apex import amp
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='classification_baselines')

parser.add_argument('--dist', action='store_true')
parser.add_argument('--local_rank', type=int, default=0)

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/data/benchmark/ILSVRC2012/imagenet-data/')
parser.add_argument('--log_name', type=str, default='alexnet_baseline')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/')

parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--wd', type=float, default=1e-4)

parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=200)
parser.add_argument('--max_epochs', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--resume', default='', type=str, help='Resuming model path for testing')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--Ther_len', type=int, default=8)
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate_path', dest='evaluate_path',
                    help='The path used to load the trained model',
                    default='save_temp', type=str)
cfg = parser.parse_args()

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
if'resnet18' in cfg.arch:
      cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt/ImageNet-Resnet18', cfg.log_name)
elif 'resnet32' in cfg.arch:
  cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt/CIFAR10-Resnet32', cfg.log_name)
else:
  cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt/CIFAR10-Resnet20', cfg.log_name)
print('python  ' + '  '.join(sys.argv))
print(cfg)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus


def load_checkpoint(model, filename):
    state_dict = torch.load(filename)
    model.load_state_dict(state_dict, strict=True)
    return 

def main():
  num_gpus = torch.cuda.device_count()
  if cfg.dist:
    device = torch.device('cuda:%d' % cfg.local_rank)
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    device = torch.device('cuda')

  print('==> Preparing data ...')                       
  train_loader, val_loader = load_dataset()
  val_dataset_len =  len(val_loader) * cfg.test_batch_size
  print("len(train_dataset): ", len(train_loader), "len(val_dataset)", len(val_loader))

  # create model
  print('==> Building model ...')
  if cfg.arch == 'resnet18':
    model = resnet18()
  elif cfg.arch == 'resnet18_3x3':
    model = resnet18_3x3()
  else:
    print("arch wrong")
    exit(-1)
  model = model.to(device)
  optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr': cfg.lr}], cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  model, optimizer = amp.initialize(model, optimizer, opt_level="O1") ###
  
  model = torch.nn.DataParallel(model)

  if cfg.dist:
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)
  else:
    model = torch.nn.DataParallel(model)
  print(model.module)

  if cfg.resume != '':
    print("Load checkpoint from {}, the start_epoch is {}".format(cfg.resume, cfg.start_epoch))
    load_checkpoint(model, cfg.resume)
    
  lr_schedulr = optim.lr_scheduler.MultiStepLR(optimizer, [30, 50, 65], gamma=0.1,  last_epoch=cfg.start_epoch - 1)  

  #criterion = torch.nn.CrossEntropyLoss()
  criterion = torch.nn.KLDivLoss(reduction='batchmean').cuda()
  
  summary_writer = SummaryWriter(cfg.log_dir)
  best_val_acc = 0
  val_acc = 0

  def train(epoch):
    # switch to train mode
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

      inputs, targets = inputs.to(device), targets.to(device)
      
      # compute output
      outputs, lessons = model(inputs)
      loss = criterion(outputs.log_softmax(dim=1), lessons.softmax(dim=1))
      for name,param in model.named_parameters():
        if ('conv' in name or 'fc' in name) and 'teacher' not in name:
          loss+= (0.0001 * torch.norm(torch.abs(param)-0.01)* torch.norm(torch.abs(param)-0.01))            
      ###loss = criterion(outputs, targets)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      ###loss.backward()
      with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
      optimizer.step()

      if cfg.local_rank == 0 and batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
              (datetime.now(), epoch, batch_idx, loss.item(),
               cfg.train_batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def validate(epoch):
    # switch to evaluate mode
    model.eval()
    top1 = 0
    top5 = 0
    with torch.no_grad():
      for i, (inputs, targets) in tqdm(enumerate(val_loader)):
        
        inputs, targets = inputs.to(device), targets.to(device)
        # compute output
        output, _ = model(inputs)

        # measure accuracy and record loss
        _, pred = output.data.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
        top5 += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()

    top1 *= 100 / val_dataset_len
    top5 *= 100 / val_dataset_len
    print('%s Precision@1 ==> %.2f%%  Precision@5: %.2f%%\n' % (datetime.now(), top1, top5))

    summary_writer.add_scalar('Precision@1', top1, epoch)
    summary_writer.add_scalar('Precision@5', top5, epoch)
    return top1

  if cfg.evaluate:
      print("=> loading checkpoint '{}'".format(cfg.evaluate_path))
      checkpoint = torch.load(cfg.evaluate_path)
      model.load_state_dict(checkpoint)
      validate(epoch=-1)
      return

  for epoch in range(cfg.max_epochs):
    lr_schedulr.step(epoch+ cfg.start_epoch)
    train(epoch+ cfg.start_epoch)
    if epoch%1==0 or epoch>40:
      val_acc = validate(epoch+ cfg.start_epoch)
    ##lr_schedulr.step(epoch+ cfg.start_epoch)
    if epoch %3 == 0 or (epoch+1)%5==0:
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint_e{}.t7'.format(epoch+ cfg.start_epoch)))
      print('checkpoint saved to %s !' % os.path.join(cfg.ckpt_dir, 'checkpoint_e{}.t7'.format(epoch+ cfg.start_epoch)))       

    if val_acc > best_val_acc:
       torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint_best.t7'))
       print('checkpoint saved to %s !' % os.path.join(cfg.ckpt_dir, 'checkpoint_best.t7'))
       best_val_acc = val_acc

  summary_writer.close()

def load_dataset():
    traindir = os.path.join(cfg.data_dir, 'train')
    valdir = os.path.join(cfg.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transform=train_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.test_batch_size, shuffle=False,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=True
    )

    return train_loader, val_loader


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

if __name__ == '__main__':
  with ExitStack() as stack:
    if cfg.local_rank != 0:
      stack.enter_context(DisablePrint())
    main()
