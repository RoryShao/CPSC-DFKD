import torch
import torch.nn as nn
import torch.nn.functional as F
from network.conditional_batchnorm import CategoricalConditionalBatchNorm2d


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        # self.opt=opt
        nz=nz
        ngf=ngf
        nc=nc
        img_size=img_size
        super(Generator, self).__init__()

        self.init_size = img_size//8
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*8*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU()
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU()
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = nn.functional.interpolate(out, scale_factor=2)
        img = self.conv_blocks0(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        super(GeneratorA, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorB(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, slope=0.2):
        super(GeneratorB, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3,1,1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output


class GeneratorC(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
        Conditional GAN with CBN for tiny-ImageNet or ImageNet
        Generated image-size: 224*224*3
    """

    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, slope=0.2, num_classes=100):
        super(GeneratorC, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = (img_size[0] // 16, img_size[1] // 16)
        else:
            self.init_size = (img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf * 8 * self.init_size[0] * self.init_size[1]),
        )

        self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(num_classes, ngf * 8)

        self.conv_blocks1_0 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(num_classes, ngf * 4, 0.8)
        self.conv_blocks1_2 = nn.LeakyReLU(slope, inplace=True)
        # 2x

        self.conv_blocks2_0 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(num_classes, ngf * 2, 0.8)
        self.conv_blocks2_2 = nn.LeakyReLU(slope, inplace=True)
        # 4x

        self.conv_blocks3_0 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.conv_blocks3_1 = CategoricalConditionalBatchNorm2d(num_classes, ngf, 0.8)
        self.conv_blocks3_2 = nn.LeakyReLU(slope, inplace=True)
        # 8x

        self.conv_blocks4_0 = nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False)
        # self.conv_blocks4_1 = nn.BatchNorm2d(ngf)
        self.conv_blocks4_1 = CategoricalConditionalBatchNorm2d(num_classes, ngf, 0.8)
        self.conv_blocks4_2 = nn.LeakyReLU(slope, inplace=True)
        # 16x

        self.conv_blocks5_0 = nn.Conv2d(ngf, nc, 3, 1, 1)
        self.conv_blocks5_1 = nn.Tanh()

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # if isinstance(m, (nn.BatchNorm2d)):
            #     nn.init.normal_(m.weight, 1.0, 0.02)
            #     nn.init.constant_(m.bias, 0)

    # @torch.cuda.amp.autocast()
    def forward(self, z, labels):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        img = self.conv_blocks0_0(proj, labels)
        img = self.conv_blocks1_0(img)
        img = self.conv_blocks1_1(img, labels)
        img = self.conv_blocks1_2(img)
        img = self.conv_blocks2_0(img)
        img = self.conv_blocks2_1(img, labels)
        img = self.conv_blocks2_2(img)
        img = self.conv_blocks3_0(img)
        img = self.conv_blocks3_1(img, labels)
        img = self.conv_blocks3_2(img)
        img = self.conv_blocks4_0(img)
        img = self.conv_blocks4_1(img, labels)
        # img = self.conv_blocks4_1(img)
        img = self.conv_blocks4_2(img)
        img = self.conv_blocks5_0(img)
        img = self.conv_blocks5_1(img)
        return img




class GeneratorD(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
        Conditional GAN for CIFAR-10 and CIFAR-100
        Generated image-size: 32*32*3
    """
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, num_classes=10):
        super(GeneratorD, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(num_classes, ngf * 2)

        self.conv_blocks1_0 = nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1)
        self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(num_classes, ngf*2)
        self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks2_0 = nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1)
        self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(num_classes, ngf)
        self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_blocks3_0 = nn.Conv2d(ngf, nc, 3, stride=1, padding=1)
        self.conv_blocks3_1 = nn.Tanh()
        # self.conv_blocks3_2 = nn.BatchNorm2d(nc, affine=False)
        self.conv_blocks3_2 = CategoricalConditionalBatchNorm2d(num_classes, nc)


    def forward(self, z, labels):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0_0(out, labels)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1_0(img)
        img = self.conv_blocks1_1(img, labels)
        img = self.conv_blocks1_2(img)
        b1_2_img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2_0(b1_2_img)
        img = self.conv_blocks2_1(img, labels)
        img = self.conv_blocks2_2(img)
        b2_2_img = self.conv_blocks3_0(img)
        img = self.conv_blocks3_1(b2_2_img)
        final_img = self.conv_blocks3_2(img, labels)
        # final_img = self.conv_blocks2_5(img)
        return final_img

