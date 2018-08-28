############################################################
# Implementation of cycleGAN, custom model 
# Reference PyTorch Example Code, https://github.com/pytorch/examples/tree/master/dcgan
# By Jinwei Yuan 2018
############################################################
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import itertools
from torch.autograd import Variable
from tensorboardX import SummaryWriter 


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--datarootA', required=True, help='path to datasetA')
parser.add_argument('--datarootB', required=True, help='path to datasetB')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    datasetA = dset.ImageFolder(root=opt.datarootA,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    datasetB = dset.ImageFolder(root=opt.datarootB,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))


dataloaderA = torch.utils.data.DataLoader(datasetA, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataloaderB = torch.utils.data.DataLoader(datasetB, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf) # number of generative filters
ndf = int(opt.ndf) # number of discriminator filters
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Dropout2d(0.5),
            # nn.Sigmoid()

            # middle transformation 2 conv layers
            nn.Conv2d(ndf * 8,nz*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nz*2,nz, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nz),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.ReLU(True),
            # nn.BatchNorm1d(nz),


            # input is Z, going into a convolution, ngf  = 128 in the original DCGAN paper
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        
        output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        model = [   nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(ndf*2), 
                    nn.LeakyReLU(0.2, inplace=True) ,

                    nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(ndf*4), 
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(ndf*4, ndf*8, 4, padding=1),
                    nn.InstanceNorm2d(ndf*8), 
                    nn.LeakyReLU(0.2, inplace=True), 
                    nn.Conv2d(ndf*8, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze(1)




netG_A2B = Generator(ngpu).to(device)
netG_B2A = Generator(ngpu).to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
print(netG_A2B)

netD_A = Discriminator(3).to(device)
netD_B = Discriminator(3).to(device)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
print(netD_A)

# Loss function
criterion_GAN = nn.MSELoss()
criterion_Cycle = nn.L1Loss()
criterion_Identity = nn.L1Loss()
# fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# optimizers
optimizerD_A = optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD_B = optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(itertools.chain(netG_A2B.parameters(),netG_B2A.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

writer = SummaryWriter(comment='CycleGAN-Jin')
damn_in = torch.rand(1,3,64,64)
writer.add_graph(netG_A2B,(damn_in.to(device),))

import numpy as np
for epoch in range(opt.niter):
    epoch_error_D_A = 0
    epoch_error_D_B = 0
    epoch_error_G_Total = 0
    epoch_error_Identity_Total = 0
    epoch_error_GAN_Total = 0
    epoch_error_Cycle_Total =0
    b_length = 0

    for i, (data_A,data_B) in enumerate(zip(dataloaderA,dataloaderB)):

        # real_A = Variable(real_A.to(device))
        # real_B = Variable(real_B.to(device))
        # real_A = Variable(real_A.cuda())
        # real_B = Variable(real_B.cuda())
        batchs_lenth = min(len(dataloaderA),len(dataloaderB))
        if i == batchs_lenth -1:
            continue

        real_A = data_A[0].to(device)
        real_B = data_B[0].to(device)
        batch_size = real_A.size(0)
        b_length = batchs_lenth
        # print('[%d/%d]' % (i, batchs_lenth-1))
  
        
        label_real = torch.full((real_A.size(0),), real_label, device=device)
        label_fake = torch.full((real_B.size(0),), fake_label, device=device)
        
        ############################
        # (1) Generators A2B, B2A
        ###########################
        optimizerG.zero_grad()
    
        # Identity Loss
        
        B_origin = netG_A2B(real_B)
        loss_idt_B = criterion_Identity(B_origin,real_B)*10.0*0.5
        A_origin = netG_B2A(real_A)
        loss_idt_A = criterion_Identity(A_origin,real_A)*10.0*0.5

        epoch_error_Identity_Total += (loss_idt_B + loss_idt_A)

        # GAN Loss
        fake_B = netG_A2B(real_A)
        pred_fake_B = netD_B(fake_B)       
        loss_G_A2B = criterion_GAN(pred_fake_B, label_real)

        fake_A = netG_B2A(real_B)
        pred_fake_A = netD_A(fake_A)
        loss_G_B2A = criterion_GAN(pred_fake_A, label_real)

        epoch_error_GAN_Total += (loss_G_A2B + loss_G_B2A)

        # Cycle Loss
        cycled_A = netG_B2A(fake_B)
        cycled_B = netG_A2B(fake_A)
        loss_cycled_A = criterion_Cycle(cycled_A,real_A)*10.0
        loss_cycled_B = criterion_Cycle(cycled_B,real_B)*10.0

        epoch_error_Cycle_Total += (loss_cycled_A + loss_cycled_B)

        # Total G Loss
        loss_G = loss_idt_A + loss_idt_B + loss_G_A2B + loss_G_B2A + loss_cycled_A + loss_cycled_B
        epoch_error_G_Total += loss_G

        # Backward
        loss_G.backward()
        optimizerG.step()

        ############################
        # (2) Discriminators A and B
        ###########################

        # Discriminator A
        optimizerD_A.zero_grad()

        # train with real
        pred_real_A = netD_A(real_A)
        errD_real = criterion_GAN(pred_real_A,label_real)
        #train with fake
        pred_fake_A = netD_A(fake_A.detach())
        errD_fake = criterion_GAN(pred_fake_A,label_fake)

        errD = (errD_fake + errD_real) * 0.5
        epoch_error_D_A += errD
        errD.backward()
        optimizerD_A.step()

        # Discriminator B
        optimizerD_B.zero_grad()

        # train with real
        pred_real_B = netD_B(real_B)
        errD_real = criterion_GAN(pred_real_B,label_real)
        #train with fake
        pred_fake_B = netD_B(fake_B.detach())
        errD_fake = criterion_GAN(pred_fake_B,label_fake)

        errD = (errD_fake + errD_real) * 0.5
        epoch_error_D_B += errD
        errD.backward()
        optimizerD_B.step()

        ############################
        # end (1) and (2)
        ###########################


    epoch_error_D_A /= (b_length - 1)
    epoch_error_D_B /= (b_length - 1)
    epoch_error_G_Total /= (b_length - 1)
    epoch_error_Identity_Total /= (b_length - 1)
    epoch_error_GAN_Total /= (b_length - 1)
    epoch_error_Cycle_Total /= (b_length - 1)
    print('[%d/%d] Loss_G_Total: %.4f Loss_D_A: %.4f Loss_D_B: %.4f'
        % (epoch, opt.niter,
            epoch_error_G_Total.item(), epoch_error_D_A.item(), epoch_error_D_B.item()))

    writer.add_scalar('data/Loss_G_Total',epoch_error_G_Total.item() , epoch)
    writer.add_scalar('data/Loss_D_A',epoch_error_D_A.item() , epoch)
    writer.add_scalar('data/Loss_D_B',epoch_error_D_B.item() , epoch)
    writer.add_scalar('data/Loss_Identity',epoch_error_Identity_Total.item() , epoch)
    writer.add_scalar('data/Loss_GAN',epoch_error_GAN_Total.item() , epoch)
    writer.add_scalar('data/Loss_Cycle',epoch_error_Cycle_Total.item() , epoch)

    vutils.save_image(real_A,
            '%s/real_A_samples_epoch_%03d.png' % (opt.outf,epoch),
            normalize=True)
    fake = netG_A2B(real_A)
    vutils.save_image(fake.detach(),
            '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
            normalize=True)

    inputImg = vutils.make_grid(real_A,normalize=True)
    outputImg = vutils.make_grid(fake.detach(),normalize=True)
    writer.add_image('sampled_generated_images',outputImg,epoch)
    writer.add_image('sampled_input_images',inputImg,epoch)

    # do checkpointing, save the last weights
    torch.save(netG_A2B.state_dict(), '%s/netG_A2B.pth' % (opt.outf))
    torch.save(netG_B2A.state_dict(), '%s/netG_B2A.pth' % (opt.outf))
    torch.save(netD_A.state_dict(), '%s/netD_A.pth' % (opt.outf))
    torch.save(netD_B.state_dict(), '%s/netD_B.pth' % (opt.outf))


writer.close()
