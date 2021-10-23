from __future__ import print_function

import argparse
import os
import sys
import shutil
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append(os.getcwd())
from python.torx.module_Binary.layer import crxb_Conv2d
from python.torx.module_Binary.layer import crxb_Linear
from models.Resnet18_Binary import resnet18
from torch.utils.tensorboard import SummaryWriter

from PIL import ImageFile
import torchvision.datasets as datasets
import torchvision
from utils_imgnet.preprocessing import Lighting
from apex import amp
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    Ther_len = int(32)
    Ther_advisor = int(256/Ther_len)
    flag_matrix = torch.zeros(int(Ther_len*3), W, H )

    for i in range(Ther_len):
        flag_matrix[ int(i*3):int(i*3+3), :, :] = int( (i+0.5)*Ther_advisor )
    flag_matrix /=255
    return flag_matrix

def Thermometer_Input(input):
    # input: [N, C, W, H]
    Ther_len = int(32)
    Ther_advisor = int(256/Ther_len)
    global flag_matrix_RGB

    input = input.repeat(1,Ther_len,1,1)
    thermometer_input = (input > flag_matrix_RGB).float()
 
    return thermometer_input

global flag_matrix_RGB
flag_matrix_RGB = Flag_Matrix_RGB()
flag_matrix_RGB = flag_matrix_RGB.cuda()


def train(model, device, criterion, optimizer, train_loader, epoch, writer=None):
    losses = AverageMeter()

    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        for name, module in model.named_modules():
            if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
                module._reset_delta()

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        #loss = criterion(output.log_softmax(dim=1), lessons.softmax(dim=1))
        for name,param in model.named_parameters():
            if ('conv' in name or 'fc' in name) and 'teacher' not in name:
                loss+= (0.0001 * torch.norm(torch.abs(param)-0.01)* torch.norm(torch.abs(param)-0.01))
        losses.update(loss.item(), data.size(0))
        #loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            print("scaled_loss", scaled_loss)
            scaled_loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_loader.sampler.__len__(),
                       100. * batch_idx / len(train_loader), loss.item()))

    print('\nTrain set: Accuracy: {}/{} ({:.4f}%)\n'.format(
        correct, train_loader.sampler.__len__(),
        100. * correct / train_loader.sampler.__len__()))
    if writer is not None:
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('train_acc' , 100. * correct / train_loader.sampler.__len__(), epoch)

    return losses.avg


def validate(args, model, device, criterion, val_loader, epoch=0, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            #test_loss += criterion(output.log_softmax(dim=1), lessons.softmax(dim=1))

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            if args.ir_drop:
                print('\nTest set: Accuracy: {}/{} ({:.4f}%)\n'.format(
                    correct, val_loader.batch_sampler.__dict__['batch_size'] * (batch_idx + 1),
                             100. * correct / (val_loader.batch_sampler.__dict__['batch_size'] * (batch_idx + 1))))

            if (batch_idx+1)%100==0 or batch_idx==0:
                N_ = target.shape[0]
                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
                    test_loss, correct, N_*(batch_idx+1),
                    100. * correct / (N_*(batch_idx+1))))  
     
        test_loss /= len(val_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, val_loader.sampler.__len__(),
            100. * correct / val_loader.sampler.__len__()))

        if writer is not None:
            writer.add_scalar('val_acc', 100. * correct / val_loader.sampler.__len__(), epoch)
            writer.add_scalar('val_loss', test_loss, epoch) 
        test_acc = 100. * correct / val_loader.sampler.__len__()           
        return test_loss, test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--modelfile', dest='modelfile',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--crxb_size', type=int, default=64, help='corssbar size')
    parser.add_argument('--vdd', type=float, default=3.3, help='supply voltage')
    parser.add_argument('--gwire', type=float, default=0.0357,
                        help='wire conductacne')
    parser.add_argument('--gload', type=float, default=0.25,
                        help='load conductance')
    parser.add_argument('--gmax', type=float, default=0.000333,
                        help='maximum cell conductance')
    parser.add_argument('--gmin', type=float, default=0.000000333,
                        help='minimum cell conductance')
    parser.add_argument('--ir_drop', action='store_true', default=False,
                        help='switch to turn on ir drop analysis')
    parser.add_argument('--scaler_dw', type=float, default=1,
                        help='scaler to compress the conductance')
    parser.add_argument('--test', action='store_true', default=False,
                        help='switch to turn inference mode')
    parser.add_argument('--enable_resistance_variance', action='store_true', default=False,
                        help='switch to turn on resistance variance analysis')
    parser.add_argument('--resistance_variance_gamma', type=float, default=0.1,
                        help='wire conductacne')    
    parser.add_argument('--enable_noise', action='store_true', default=False,
                        help='switch to turn on noise analysis')
    parser.add_argument('--enable_SAF', action='store_true', default=False,
                        help='switch to turn on SAF analysis')
    parser.add_argument('--enable_ec_SAF', action='store_true', default=False,
                        help='switch to turn on SAF error correction')
    parser.add_argument('--freq', type=float, default=10e6,
                        help='scaler to compress the conductance')
    parser.add_argument('--temp', type=float, default=300,
                        help='scaler to compress the conductance')

    parser.add_argument('--data_dir', type=str, default='/data/benchmark/ILSVRC2012/imagenet-data/')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--arch', type=str, default='resnet18')
    



    args = parser.parse_args()

    best_error = 0

    if args.ir_drop and (not args.test):
        warnings.warn("We don't recommend training with IR drop, too slow!")

    if args.ir_drop and args.test_batch_size > 150:
        warnings.warn("Reduce the batch size, IR drop is memory hungry!")

    if not args.test and args.enable_noise:
        raise KeyError("Noise can cause unsuccessful training!")

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    train_loader, test_loader = load_dataset(cfg=args)
    if args.arch == 'resnet18':
        model = resnet18(crxb_size=args.crxb_size, gmax=args.gmax, gmin=args.gmin, gwire=args.gwire, gload=args.gload,
                vdd=args.vdd, ir_drop=args.ir_drop, device=device, scaler_dw=args.scaler_dw, freq=args.freq, temp=args.temp,
                enable_resistance_variance=args.enable_resistance_variance, resistance_variance_gamma=args.resistance_variance_gamma,
                enable_SAF=args.enable_SAF, enable_noise=args.enable_noise, enable_ec_SAF=args.enable_ec_SAF).to(device)            
    writer = SummaryWriter(os.path.join(args.save_dir,'log'))
    print('python  ' + '  '.join(sys.argv))
    if not args.test:
        print(model)
    print(args)   
    optimizer = torch.optim.SGD([{'params':model.parameters(),'initial_lr': args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[30, 50, 65 ], last_epoch=args.start_epoch - 1)  
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1") ###

    #resume_path = '/data/tqd/from580_pytorx-CIFAR/save-HAWS/mine5/checkpoint_e44.t7'
    if args.arch == 'resnet18':
        resume_path = '../training_test/ckpt/ImageNet-Resnet18/HAWIS-B/checkpoint_best.t7'
    print('--------------------')
    params = torch.load(resume_path)
    params_new = {}
    for key in params.keys():
        key_new = key[14:]
        params_new[key_new] = params[key]

    del params
    for key in model.state_dict().keys():
        if key in params_new.keys():
            pass
        else:
            params_new[key] = model.state_dict()[key]
    
    params_noTeacher ={}
    for key in params_new.keys():
        if key  in model.state_dict().keys():
            params_noTeacher[key] = params_new[key]
    # model.load_state_dict(checkpoint['state_dict'],strict=False)
    del params_new
    model.load_state_dict(params_noTeacher)#,strict=False)



    loss_log = []
    best_acc = 0 
    if not args.test:
        for epoch in range(args.epochs):
            print("epoch {0}, and now lr = {1:.4f}\n".format(epoch, optimizer.param_groups[0]['lr']))
            train_loss = train(model=model, device=device, criterion=criterion,
                               optimizer=optimizer, train_loader=train_loader,
                               epoch=epoch, writer=writer)
            val_loss,test_acc = validate(args=args, model=model, device=device, criterion=criterion,
                                val_loader=test_loader, epoch=epoch, writer=writer)

            scheduler.step()

            loss_log += [(epoch, train_loss, val_loss)]
            is_best = val_loss > best_error
            best_error = min(val_loss, best_error)

            filename = 'checkpoint_' + str(args.crxb_size) + '.pth.tar'
            filename = os.path.join(args.save_dir, filename)
            save_checkpoint(state={
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_error,
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename=filename)

            is_best_acc = test_acc> best_acc
            best_acc = max(test_acc, best_acc)
            filename = 'checkpoint_' + str(args.crxb_size) + '_acc_best.pth.tar'
            filename = os.path.join(args.save_dir, filename)
            if epoch>=0 and is_best_acc:
                save_checkpoint(state={
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_error,
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best, filename=filename)

    elif args.test:
        if not args.enable_SAF and not args.enable_resistance_variance:
            print("No  SAF and No noise!")        
            for j in np.arange(start = 0.01, stop = 0.11, step = 0.01):
                test_p_SA0 =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
                test_p_SA1 =nn.Parameter(torch.Tensor([j]), requires_grad=False)
                test_p_SA0 = test_p_SA0.cuda()
                test_p_SA1 = test_p_SA1.cuda()
                
                for M in model.named_modules():
                    if 'SAF' in M[0]:
                        M[1].p_SA0.data = test_p_SA0
                        M[1].p_SA1.data = test_p_SA1

                
                print("conv1.w2g.SAF_pos.p_SA0",model.conv1_1.w2g.SAF_pos.p_SA0.data   )
                print("conv1.w2g.SAF_pos.p_SA1 ", model.conv1_1.w2g.SAF_pos.p_SA1.data  )

                test_loss, test_acc = validate(args=args, model=model, device=device, criterion=criterion,
                        val_loader=test_loader)
                print(test_acc)

        if args.enable_SAF:
            print("SAF enabled!")        
            result = []
            #for i in [ 0,0.001, 0.002, 0.003, 0.004, 0.005, 0.007, 0.008, 0.009, 0.01, 0.0125, 0.015, 0.0175, 0.02]:
            for j in np.arange(start = 0, stop = 0.11, step = 0.01):
                test_p_SA0 =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
                test_p_SA1 =nn.Parameter(torch.Tensor([j]), requires_grad=False)
                test_p_SA0 = test_p_SA0.cuda()
                test_p_SA1 = test_p_SA1.cuda()
                
                for M in model.named_modules():
                    if 'SAF' in M[0]:
                        M[1].p_SA0.data = test_p_SA0
                        M[1].p_SA1.data = test_p_SA1

                
                print("conv1.w2g.SAF_pos.p_SA0",model.conv1_1.w2g.SAF_pos.p_SA0.data   )
                print("conv1.w2g.SAF_pos.p_SA1 ", model.conv1_1.w2g.SAF_pos.p_SA1.data  )

                test_loss, test_acc = validate(args=args, model=model, device=device, criterion=criterion,
                        val_loader=test_loader)
                result.append(test_acc)
            print("result SA1: ",result)        

        if args.enable_resistance_variance:
            print("resistance variance enabled!")
            result = []
            print(" resistance variance gamma = ",args.resistance_variance_gamma )
            # test_loss, test_acc = validate(args=args, model=model, device=device, criterion=criterion,
            #             val_loader=test_loader)
            
            for g in np.arange(start=0, stop=1.1, step=0.05):
                if args.arch == 'resnet18':
                    model = resnet18(crxb_size=args.crxb_size, gmax=args.gmax, gmin=args.gmin, gwire=args.gwire, gload=args.gload,
                        vdd=args.vdd, ir_drop=args.ir_drop, device=device, scaler_dw=args.scaler_dw, freq=args.freq, temp=args.temp,
                        enable_SAF=args.enable_SAF, enable_noise=args.enable_noise, 
                        enable_resistance_variance=args.enable_resistance_variance, resistance_variance_gamma=g,
                        enable_ec_SAF=args.enable_ec_SAF).to(device)   
                if args.arch == 'resnet18':
                    resume_path = '../training_test/ckpt/ImageNet-Resnet18/HAWIS-B/checkpoint_best.t7'   
                print('--------------------')
                params = torch.load(resume_path)

                params_new = {}
                for key in params.keys():
                    key_new = key[14:]
                    params_new[key_new] = params[key]

                print('======================')
                del params
                for key in model.state_dict().keys():
                    if key in params_new.keys():
                        pass
                    else:
                        params_new[key] = model.state_dict()[key]

                params_noTeacher ={}
                for key in params_new.keys():
                    if key  in model.state_dict().keys():
                        params_noTeacher[key] = params_new[key]
                del params_new
                model.load_state_dict(params_noTeacher)#,strict=False)
                
                print(" resistance variance gamma = ",g )
                test_loss, test_acc = validate(args=args, model=model, device=device, criterion=criterion,
                        val_loader=test_loader)
                result.append(test_acc)
            print("resistance variance result: ", result)

        if not args.enable_SAF and not args.enable_resistance_variance: 
            print("No SAF and No resistance variance!")
            test_loss, test_acc = validate(args=args, model=model, device=device, criterion=criterion,
                        val_loader=test_loader)


def load_dataset(cfg):
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

if __name__ == '__main__':
    main()
