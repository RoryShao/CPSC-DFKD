from __future__ import print_function

import argparse
import os
import torch
import random
import network
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from utils.log_util import Logger
from dataloader import get_dataloader
import warnings
warnings.filterwarnings("ignore")


def train(args, model, device, train_loader, optimizer, cur_epoch, log):
    model.train()
    loss_all = []
    correct = 0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss = F.cross_entropy(output, target)
        loss_all.append(loss)
        loss.backward()
        optimizer.step()
    log.logger.info('Train Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        cur_epoch, torch.Tensor(loss_all).mean(), correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))


def test(args, model, device, test_loader, cur_epoch, log):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            if args.epochs == 1 and i == 1:
                print('test image:', data.shape)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log.logger.info('Test Epoch: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        cur_epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct / len(test_loader.dataset)
    return acc



def get_model(args):
    if args.model.lower() == 'resnet18':
        return network.resnet.resnet18(num_classes=args.num_classes)
    elif args.model.lower() == 'resnet34':
        return network.resnet.resnet34(num_classes=args.num_classes)
    elif args.model.lower() == 'vgg11':
        return network.vgg.vgg11_bn(num_classes=args.num_classes)
    elif args.model.lower() == 'wrn_16_2':
        return network.wresnet.wrn_16_2(num_classes=args.num_classes)
    elif args.model.lower() == 'wrn_40_2':
        return network.wresnet.wrn_40_2(num_classes=args.num_classes)
    elif args.model.lower() == 'wrn_40_1':
        return network.wresnet.wrn_40_1(num_classes=args.num_classes)


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--data_root', type=str, default=r'/data/xxx/datasets/tiny-imagenet')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    # parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
    #                     help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                        help='dataset name (default: mnist)')
    parser.add_argument('--model', type=str, default='wrn_16_1', choices=['resnet18', 'resnet34', 'vgg11', 'wrn_40_2', 'wrn_40_1', 'wrn_16_2'],
                        help='model name (default: mnist)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--step_size', type=int, default=10, metavar='S', help='(default: 10)')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--scheduler', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--operator', type=str, default='Teacher')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    log_path = 'log/' + args.dataset + '/'
    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path + '%s_%s_log.txt' %(args.dataset, args.gpu_id),
                 level='info')
    log.logger.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")
    os.makedirs('checkpoint/teacher', exist_ok=True)

    train_loader, test_loader = get_dataloader(args)
    model = get_model(args)
    # print(model)
    model = model.to(device)

    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    best_acc = 0
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.test_only:
        acc = test(args, model, device, test_loader, 0, log)
        print(acc)
        return

    for epoch in range(1, args.epochs + 1):
        if args.scheduler:
            scheduler.step()
        print("Lr = %.6f"%(optimizer.param_groups[0]['lr']))
        train(args, model, device, train_loader, optimizer, epoch, log)
        acc = test(args, model, device, test_loader, epoch, log)
        if acc>best_acc:
            best_acc = acc
            torch.save(model.state_dict(),"checkpoint/teacher/add_%s-%s.pt" % (args.dataset, args.model))
    log.logger.info('Best Acc={:.2f}'.format(best_acc * 100.))


if __name__ == '__main__':
    main()