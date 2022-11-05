from torchvision import datasets, transforms
import torch


def get_dataloader(args):
    if args.dataset.lower()=='mnist':
        train_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(args.data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.batch_size, shuffle=True, num_workers=2)

    elif args.dataset.lower()=='cifar10':
        if args.operator == 'Teacher':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_root, train=True, download=False,
                           transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_root, train=False, download=False,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
        elif args.operator == 'DFAD':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.data_root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                  ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)

    elif args.dataset.lower()=='cifar100':
        if args.operator == 'Teacher':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_root, train=True, download=True,
                           transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                            ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_root, train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                            ])),
                batch_size=args.batch_size, shuffle=False, num_workers=8)
        elif args.operator == 'DFAD':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR100(args.data_root, train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                  ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)
    elif args.dataset.lower() == 'tiny-imagenet':
        if args.operator == 'Teacher':
            train_loader =torch.utils.data.DataLoader(datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/train/',
                                                 transform=transforms.Compose([
                                                  transforms.RandomCrop(64, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])),
               batch_size=args.batch_size, shuffle=True, num_workers=8)
            test_loader = torch.utils.data.DataLoader(datasets.ImageFolder(root=args.data_root+'/tiny-imagenet-200/val/',
                                               transform=transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                 ])),
              batch_size=args.batch_size, shuffle=True, num_workers=8)

        elif args.operator == 'DFAD':
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(root=args.data_root + '/tiny-imagenet-200/val/',
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])),
                batch_size=args.batch_size, shuffle=True, num_workers=8)

    return train_loader, test_loader


if __name__ == '__main__':
    pass