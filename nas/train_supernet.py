import argparse
import logging
import os
import sys
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import numpy as np
from torchinfo import summary
import utils
from models.fixed_Singlepath_search import SinglePath_Search
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

parser = argparse.ArgumentParser("Single_Path_One_Shot")
parser.add_argument('--exp_name', type=str, default='spos_fixed_updated', help='experiment name')
# Supernet Settings
parser.add_argument('--layers', type=int, default=8, help='batch size')
parser.add_argument('--num_choices', type=int, default=4, help='number choices per layer')
# Training Settings
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=229, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--print_freq', type=int, default=10, help='print frequency of training')
#parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/', help='checkpoints direction')
parser.add_argument('--seed', type=int, default=0, help='training seed')
# Dataset Settings
parser.add_argument('--data_root', type=str, default='./dataset/', help='dataset dir')
parser.add_argument('--classes', type=int, default=10, help='dataset classes')
parser.add_argument('--dataset', type=str, default='cifar10', help='path to the dataset')
#parser.add_argument('--cutout', action='store_true', help='use cutout')
#parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
#parser.add_argument('--auto_aug', action='store_true', default=False, help='use auto augmentation')
#parser.add_argument('--resize', action='store_true', default=False, help='use resize')
#GPU
parser.add_argument('--gpu', type=int, default=0, help='CUDA device')
args = parser.parse_args()
# Set device
args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)
utils.set_seed(args.seed)


def train(args, epoch, train_loader, model, criterion, optimizer):
    model.train()
    lr = optimizer.param_groups[0]["lr"]
    train_acc = utils.AverageMeter()
    train_loss = utils.AverageMeter()
    steps_per_epoch = len(train_loader)
    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        choice = utils.random_choice(args.num_choices, args.layers)
        kchoice = list(np.random.randint(args.num_choices-1, size=args.layers))
        kernel_choice = list(np.random.randint(args.num_choices, size=args.layers))
        outputs, _ = model(inputs, choice, kchoice, kernel_choice)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        train_loss.update(loss.item(), n)
        train_acc.update(prec1.item(), n)
        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            logging.info(
                '[Supernet Training] lr: %.5f epoch: %03d/%03d, step: %03d/%03d, '
                'train_loss: %.3f(%.3f), train_acc: %.3f(%.3f)'
                % (lr, epoch+1, args.epochs, step+1, steps_per_epoch,
                   loss.item(), train_loss.avg, prec1, train_acc.avg)
            )
    return train_loss.avg, train_acc.avg


def validate(args, val_loader, model, criterion):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            choice = utils.random_choice(args.num_choices, args.layers)
            kchoice = list(np.random.randint(args.num_choices-1, size=args.layers))
            kernel_choice = list(np.random.randint(args.num_choices, size=args.layers))
            # print("kernel choice", kernel_choice)
            #print("choice = ",choice)
            #print("kchoice = ",kchoice)
            outputs, _ = model(inputs, choice, kchoice, kernel_choice)
            loss = criterion(outputs, targets)
            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_loss.update(loss.item(), n)
            val_acc.update(prec1.item(), n)
    return val_loss.avg, val_acc.avg


def main():
    # Check Checkpoints Direction
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)

    # Define Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    #generator1 = torch.Generator().manual_seed(42)
    trainset = datasets.CIFAR10(
        root=os.path.join(args.data_root, args.dataset), train=True, download=True, transform=transform_train)

    trainset, valset = torch.utils.data.random_split(trainset, [42500, 7500])
    
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root=os.path.join(args.data_root, args.dataset), train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
    
    # Define Supernet
    model = SinglePath_Search(args.dataset, args.classes, args.layers).to(args.device)
    print("summary")
    summary(model)
    #best_supernet_weights = './checkpoints/spos_c10_k32_64_128_train_supernet_best.pth'
    #checkpoint = torch.load(best_supernet_weights, map_location=args.device)
    #model.load_state_dict(checkpoint, strict=True)
    #logging.info('Finish loading checkpoint from %s', best_supernet_weights)
    logging.info(model)
    model = model.to(args.device)
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    print('\n')

    # Running
    start = time.time()
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Supernet Training
        train_loss, train_acc = train(args, epoch, train_loader, model, criterion, optimizer)
        scheduler.step()
        logging.info(
            '[Supernet Training] epoch: %03d, train_loss: %.3f, train_acc: %.3f' %
            (epoch + 1, train_loss, train_acc)
        )
        # Supernet Validation
        val_loss, val_acc = validate(args, val_loader, model, criterion)
        # Save Best Supernet Weights
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(args.ckpt_dir, '%s_%s' % (args.exp_name, 'best.pth'))
            torch.save(model.state_dict(), best_ckpt)
            logging.info('Save best checkpoints to %s' % best_ckpt)
        logging.info(
            '[Supernet Validation] epoch: %03d, val_loss: %.3f, val_acc: %.3f, best_acc: %.3f'
            % (epoch + 1, val_loss, val_acc, best_val_acc)
        )
        print('\n')

    # Record Time
    utils.time_record(start)


if __name__ == '__main__':
    main()
