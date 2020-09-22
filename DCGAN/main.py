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
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from torch.utils.tensorboard import SummaryWriter

manualSeed = 999

print("Random seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "./data/fhead_US"
log_path = './logs/tb/'
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64 # Size of feature maps in G
ndf = 64 # Size of feature maps in D
num_epochs = 500

lr = 0.0002

beta1 = 0.5
ngpu = 1

##--------------
## Dataset
##--------------
transform = transforms.Compose([
				transforms.Resize(image_size),
				transforms.CenterCrop(image_size), 
				transforms.ToTensor(), 
				transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)
					)])

dataset = dset.ImageFolder(root=dataroot, transform = transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

device = torch.device("cuda:3" if (torch.cuda.is_available() and ngpu>0) else "cpu")

##--------------
## Visualize images 
##--------------

real_batch = next(iter(dataloader))
vutils.save_image(real_batch[0][:64], './logs/train_sample.png', normalize=True)

##--------------
## Utils
##--------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


##--------------
## Weight init
##--------------

def weights_init(m):
	classname =  m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

##--------------
## Generator code
##--------------

class DCGAN_Gen(nn.Module):
	def __init__(self, ngpu):
		super(DCGAN_Gen, self).__init__()

		self.ngpu = ngpu
		self.main = nn.Sequential(
					nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), 
					nn.BatchNorm2d(ngf*8),
					nn.ReLU(True),

					nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), 
					nn.BatchNorm2d(ngf*4),
					nn.ReLU(True),

					nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), 
					nn.BatchNorm2d(ngf*2),
					nn.ReLU(True),

					nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 0, bias=False), 
					nn.BatchNorm2d(ngf),
					nn.ReLU(True),

					nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), 
					nn.Tanh(),
					)

	def forward(self, x):
		return self.main(x)

netG = DCGAN_Gen(ngpu).to(device)

# Handle multi-gpu 
if (device.type == 'cuda') and (ngpu > 1):
	netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)
print("Number of params in G (million): ", count_parameters(netG)/1000000)

##--------------
## Discriminator code
##--------------

class DCGAN_Dis(nn.Module):
	def __init__(self, ngpu):
		super(DCGAN_Dis, self).__init__()

		self.ngpu = ngpu
		self.main = nn.Sequential(
					nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
					nn.LeakyReLU(0.2, inplace=True), 

					nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ndf*2), 
					nn.LeakyReLU(0.2, inplace=True), 

					nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ndf*4), 
					nn.LeakyReLU(0.2, inplace=True), 

					nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
					nn.BatchNorm2d(ndf*8),
					nn.LeakyReLU(0.2, inplace=True), 

					nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
					nn.Sigmoid()

					)

	def forward(self, x):
		return self.main(x)

netD = DCGAN_Dis(ngpu).to(device)

if (device.type=='cuda') and (ngpu>1):
	netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

print(netD)
print("Number of params in D (million): ", count_parameters(netD)/1000000)

##--------------
## Loss and Optim
##--------------

# Real label: 1 and Fake label: 0

criterion = nn.BCELoss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)

real_label = 1. 
fake_label = 0. 

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas= (beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas= (beta1, 0.999))

##--------------
## Tensorboard Logger
##--------------
writer = SummaryWriter(log_path)
writer.add_graph(netG, torch.randn(batch_size, nz, 1, 1, device=device))
writer.close()
##--------------
## Training Loop
##--------------

img_list = []
G_losses = []
D_losses = []

iters = 0

print("Starting Training Loop...")

for epoch in range(num_epochs):

	for i, data in enumerate(dataloader, 0):
		# Update D: maximize log(D(x)) + log(1-D(G(z)))

		netD.zero_grad()
		real_cpu = data[0].to(device)
		b_size = real_cpu.size(0)
		label = torch.full((b_size, ), real_label, dtype=torch.float, device=device)

		output = netD(real_cpu).view(-1)
		errD_real = criterion(output, label)
		errD_real.backward()
		D_x = output.mean().item()

		#Train with all-fake batch 
		noise = torch.randn(b_size, nz, 1, 1, device=device)
		fake = netG(noise)
		label.fill_(fake_label)

		output = netD(fake.detach()).view(-1)
		errD_fake = criterion(output, label)
		errD_fake.backward()
		D_G_z1 = output.mean().item()

		errD = errD_real + errD_fake
		optimizerD.step()

		# Update G network: maximize log(D(G(z)))
		netG.zero_grad()
		label.fill_(real_label)

		output = netD(fake).view(-1)
		errG = criterion(output, label)
		errG.backward()
		D_G_z2 = output.mean().item()
		optimizerG.step()

		# Output training stats
		if i % 1 == 0:
		    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
		          % (epoch, num_epochs, i, len(dataloader),
		             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
		    writer.add_scalar('Loss/netG', errG.item(), iters)
		    writer.add_scalar('Loss/netD', errD.item(), iters)

		# Save Losses for plotting later
		G_losses.append(errG.item())
		D_losses.append(errD.item())

		# Check how the generator is doing by saving G's output on fixed_noise
		if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
		    with torch.no_grad():
		        fake = netG(fixed_noise).detach().cpu()
		    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
		    img_list.append(img_grid)
		    writer.add_image('Validation', img_grid)

		    torch.save(netG, './logs/model.pth')

		iters += 1			

##--------------
## Visualize G's progression
##--------------
fig = plt.figure(figsize=(8,8))
plt.axis("off")

ims = [np.transpose(i,(1,2,0)) for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

f = "./logs/DCGAN_evolution.gif" 
writergif = animation.PillowWriter(fps=30) 
ani.save(f, writer=writergif)






































