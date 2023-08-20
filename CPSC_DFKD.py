from __future__ import print_function

import argparse
import os
import random
import warnings

import matplotlib.image as im
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler

import network
from dataloader import get_dataloader
from metric.DeepInversion import DeepInversionFeatureHook
from metric.loss import SupConLoss
from metric.loss import kdloss
from network.adapter import WRNAdapter
from utils.log_util import Logger
from utils.misc import pack_images, denormalize

warnings.filterwarnings("ignore")


def get_sim_criterion(args, device):
    sim_criterion = SupConLoss(temperature=args.temp, device=device)
    return sim_criterion


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def train(args, teacher, student, generator, device, optimizer, epoch, hooks, log):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_S, optimizer_G = optimizer
    loss_all_G = []
    loss_all_S = []

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for i in range(args.epoch_itrs):
        for k in range(args.G_steps):
            with torch.cuda.amp.autocast():
                loss_G = 0
                z = nn.Parameter(torch.randn((args.batch_size, args.nz, 1, 1)).to(device))
                labels = torch.randint(0, args.num_classes, (args.batch_size,), dtype=torch.long).to(device)
                optimizer_G = torch.optim.Adam([{'params': generator.parameters()}, {'params': [z]}], args.lr_G, betas=[0.5, 0.999])
                optimizer_G.zero_grad()
                generator.train()
                # generator.apply(fix_bn)
                fake = generator(z, labels)
                if epoch == 1 and args.epoch_itrs == 0:
                    print('fake:', fake.shape)
                t_fea, t_logit = teacher(fake, out_feature=True)
                s_fea, s_logit = student(fake, out_feature=True)

                loss_KD = - kdloss(s_logit, t_logit)
                loss_G += loss_KD
                if args.loss_IKD:
                    if args.dataset == 'cifar10':
                        loss_l2 = - torch.log(F.mse_loss(s_logit, t_logit.detach()))
                        # or loss_l2 = - torch.log(torch.norm(t_logit - s_logit, 2))
                    else:
                        loss_l2 = - F.mse_loss(s_logit, t_logit.detach())
                        # or loss_l2 = - torch.log(torch.norm(t_logit - s_logit, 2))
                    loss_IKD = args.co_alpha * loss_l2
                    loss_G += loss_IKD
                if args.loss_BN:
                    loss_BN = sum([h.r_feature for h in hooks])
                    loss_G += args.co_beta * loss_BN
                if args.loss_SCL:
                    features = torch.cat([s_fea.unsqueeze(1), t_fea.unsqueeze(1)], dim=1)
                    sim_criterion = get_sim_criterion(args, device)
                    loss_SCL = sim_criterion(features, labels=labels)
                    loss_G += args.co_gamma * loss_SCL
                if args.loss_CE:
                    loss_CE = F.cross_entropy(F.softmax(t_logit, dim=1), labels, reduction='mean')

                    loss_G += args.co_eta * loss_CE
                loss_all_G.append(loss_G)
            if args.use_amp:
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
            else:
                loss_G.backward()
                optimizer_G.step()
        for k in range(args.S_steps):
            with torch.cuda.amp.autocast():
                loss_S = 0
                z = nn.Parameter(torch.randn((args.batch_size, args.nz, 1, 1)).to(device))
                labels = torch.randint(0, args.num_classes, (args.batch_size,), dtype=torch.long).to(device)
                optimizer_S.zero_grad()
                fake = generator(z, labels).detach()
                with torch.no_grad():
                    t_fea, t_logit = teacher(fake, out_feature=True)
                s_fea, s_logit = student(fake, out_feature=True)
                base_KD = kdloss(s_logit, t_logit.detach())
                loss_S += base_KD
                if args.loss_IKD:
                    if args.dataset == 'cifar10':
                        loss_l2 = torch.log(torch.norm(t_logit.detach() - s_logit, 2))
                    else:
                        loss_l2 = F.mse_loss(s_logit, t_logit.detach())
                    loss_S += args.co_alpha * loss_l2

                if args.loss_CE:
                    loss_CE = F.cross_entropy(F.softmax(s_logit), labels, reduction='mean')

                    loss_S += args.co_eta * loss_CE

                loss_all_S.append(loss_S)

            if args.use_amp:
                scaler.scale(loss_S).backward()
                scaler.step(optimizer_S)
                scaler.update()
            else:
                loss_S.backward()
                optimizer_S.step()
    log.logger.info('Train Epoch: {}/{} G_Loss: {:.6f} S_loss: {:.6f}'.format(epoch, args.epochs, torch.Tensor(loss_all_G).mean(),
                                                    torch.Tensor(loss_all_S).mean()))


def test(args, student, generator, device, test_loader, epoch, log):
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if epoch == 1 and i == 1:
                print('test image:', data.shape)
            output = student(data)
            if epoch % 50 == 0:
                z = torch.randn((data.shape[0], args.nz, 1, 1), device=data.device, dtype=data.dtype)
                labels = torch.randint(0, args.num_classes, (data.shape[0],), dtype=torch.long).to(device)
                fake = generator(z, labels)
                pic_fake = pack_images(denormalize(fake, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)).clamp(0, 1).detach().cpu().numpy())
                pic_fake = np.transpose(pic_fake, (1, 2, 0))
                img_path = 'images/' + args.dataset + '/' + args.teacher + '_' + args.student + '/' + str(epoch) + '/'
                os.makedirs(img_path, exist_ok=True)
                im.imsave(img_path+'epoch_'+str(epoch) + '.png', pic_fake)

            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    log.logger.info('Test Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(epoch,
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    acc = correct/len(test_loader.dataset)
    return acc


def main():
    ''' Parameters setting '''
    parser = argparse.ArgumentParser(description='CPSC-DFKD')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--num_classes', type=int, default=100, metavar='N',
                        help='input num_classes for dataset (default: 100)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 600)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-3,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--data_root', type=str, default='/data/xxx/datasets/cifar100')
    parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tiny-imagenet'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--teacher', type=str, default='resnet34', choices=['resnet34', 'vgg11', 'wrn_40_2'],
                        help='model name (default: resnet18)')
    parser.add_argument('--student', type=str, default='resnet18', choices=['resnet18', 'wrn_16_2', 'wrn_40_1'],
                        help='model name (default: resnet18)')

    parser.add_argument('--loss_IKD', action='store_true', default=False)
    parser.add_argument('--loss_CE', action='store_true', default=False)
    parser.add_argument('--loss_BN', action='store_true', default=False)
    parser.add_argument('--loss_SCL', action='store_true', default=False)
    parser.add_argument('--co_alpha', type=float, default='1', help='IKD hyper-parameter')
    parser.add_argument('--co_beta', type=float, default='1',  help='BN hyper-parameter')
    parser.add_argument('--co_gamma', type=float, default='1',  help='SCL hyper-parameter')
    parser.add_argument('--co_eta', type=float, default='1',  help='BN hyper-parameter')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str, default='checkpoint/teacher/cifar100-resnet34.pt')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--download', action='store_true', default=False)
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--step_size', type=int, default=100, metavar='S')
    parser.add_argument('--G_steps', type=int, default=1, metavar='G')
    parser.add_argument('--S_steps', type=int, default=5, metavar='S')
    parser.add_argument('--scheduler', action='store_true', default=True)
    parser.add_argument('--operator', type=str, default='DFAD')
    parser.add_argument('--use_amp', action='store_true', default=False)
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if use_cuda else "cpu")

    ''' Log setting '''
    log_path = 'log/'+args.dataset+'/'
    os.makedirs(log_path, exist_ok=True)
    log = Logger(log_path+'%s_%s_%s-%s_log.txt' % (args.operator, args.dataset, args.teacher, args.student), level='info')
    log.logger.info(args)

    ''' Data setting '''
    _, test_loader = get_dataloader(args)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'tiny-imagenet':
        num_classes = 200
    else:
        assert 'Teacher Not Implmentation'

    ''' Model setting '''
    '''  Teacher  '''
    if args.teacher == 'resnet34':
        teacher = network.resnet.resnet34(num_classes=num_classes)
    elif args.teacher == 'vgg11':
        teacher = network.vgg.vgg11_bn(num_classes=num_classes)
    elif args.teacher == 'wrn_40_2':
        teacher = network.wresnet.wrn_40_2(num_classes=num_classes)
    else:
        assert 'Teacher Not Implmentation'

    '''  Student  '''
    if args.student == 'resnet18':
        student = network.resnet.resnet18(num_classes=num_classes)
    elif args.student == 'wrn_16_2':
        student = network.wresnet.wrn_16_2(num_classes=num_classes)
    elif args.student == 'wrn_16_1':
        student = network.wresnet.wrn_16_1(num_classes=num_classes)
        student = WRNAdapter(model=student, feat_in=student.fc.in_features, feat_out=teacher.fc.in_features)
    elif args.student == 'wrn_40_2':
        student = network.wresnet.wrn_40_2(num_classes=num_classes)
    elif args.student == 'wrn_40_1':
        student = network.wresnet.wrn_40_1(num_classes=num_classes)
        student = WRNAdapter(model=student, feat_in=student.fc.in_features, feat_out=teacher.fc.in_features)
    else:
        assert 'Student Not Implmentation'

    '''  Generator  '''

    generator = network.gan.GeneratorD(nz=args.nz, nc=3, img_size=32, num_classes=num_classes)
    if args.dataset == 'tiny-imagenet':
        generator = network.gan.GeneratorD(nz=args.nz, nc=3, img_size=64, num_classes=num_classes)
    teacher.load_state_dict(torch.load(args.ckpt, map_location=device))   #   ['state_dict']
    print("Teacher restored from %s"%(args.ckpt))

    teacher = teacher.to(device)
    student = student.to(device)
    generator = generator.to(device)
    teacher.eval()

    '''  Optimization  '''
    hooks = []
    for m in teacher.modules():
        if isinstance(m, nn.BatchNorm2d):
            hooks.append(DeepInversionFeatureHook(m))
    assert len(hooks) > 0, 'input model should contains at least one BN layer for DeepInversion'

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G, betas=(0.5, 0.999))
    
    if args.scheduler:
        scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_S, T_max=args.epochs)
        scheduler_warmup = GradualWarmupScheduler(optimizer_S, multiplier=1, total_epoch=10, after_scheduler=scheduler_S)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, [50, 150], 0.1)

    ''' Evaluation '''
    best_acc = 0
    if args.test_only:
        acc = test(args, student, generator, device, test_loader, log)
        print(acc)
        return

    ''' Training stage'''
    acc_list = []
    for epoch in range(1, args.epochs + 1):
        # Train
        if args.scheduler:
            scheduler_S.step()
            scheduler_G.step()
            scheduler_warmup.step(epoch)
        log.logger.info("Generator  Lr = %.6f, Student Lr = %.6f"%(optimizer_G.param_groups[0]['lr'], optimizer_S.param_groups[0]['lr']))
        train(args, teacher=teacher, student=student, generator=generator, device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch, hooks=hooks, log=log)
        # Test
        acc = test(args, student, generator, device, test_loader, epoch, log)
        acc_list.append(acc)
        if acc>best_acc:
            best_acc = acc
            s_path = r'checkpoint/student/'
            os.makedirs(s_path, exist_ok=True)
            torch.save(student.state_dict(),"checkpoint/student/%s-%s.pt"%(args.dataset, 'resnet18'))
            torch.save(generator.state_dict(),"checkpoint/student/%s-%s-generator.pt"%(args.dataset, 'resnet18'))
    log.logger.info("Best Acc=%.4f" % best_acc)


if __name__ == '__main__':
    main()