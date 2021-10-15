import datasets.dataset as dataset
from tqdm import tqdm
import torchvision.transforms as transforms
import pandas as pd
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from data_loaders import coco_loaders, coco_mpii_loaders, lets_dance_loaders
import models.models as models

df = pd.read_csv('data/letsdance/train_videos.csv')
transform = {}
device = torch.device('cuda:2')
transform['optical_flow'] = transforms.Compose([transforms.CenterCrop(224)])
mean, std = [0.425, 0.425, 0.425], [0.229, 0.225, 0.224]
transform['frames'] = transforms.Compose([transforms.CenterCrop(224), transforms.Normalize(mean=mean, std=std), transforms.RandomGrayscale(), transforms.ColorJitter()])
data = dataset.DanceVideoDataset(224,df,optical_flow=True,transform=transform)
model = models.RGB_Bodypose_Classifier()
model.to(device)

for i in range(len(data)):
	d = data[i]
	frames = d['frames'].to(device)
	print(frames.shape)
	output = model(frames.unsqueeze(0))
	print(output.shape)
	break



"""
train_test_split = np.random.binomial(size=len(df_all), n=1, p=0.83)
df_train = df_all.iloc[train_ids]
df_val = df_all.iloc[test_ids]
train_data = dataset.DanceVideoDataset(224, df_train, 'train', transform=t)
val_data = dataset.DanceVideoDataset(224, df_val, 'val', transform=t)

train_loader = DataLoader(train_data, batch_size=2, num_workers=5)
model = models.Dance_Classifier2()
device = torch.device('cuda:1')
model.to(device)
for batch in tqdm(train_loader):
	vdo, opt = batch['frames'], batch['optical_flow']
	vdo, opt = vdo.to(device), opt.to(device)
	o = model(vdo, opt)
	print(o.shape)
for t in range(len(train_data)):
	sample = train_data[t]
	try:
		sample = val_data[t]
	except:
		continue


for i in tqdm(range(len(vdo_data))):
	sample = vdo_data[i]



#coco_data = dataset.COCODataset('./data/coco', 'train', 16, 8,  1, 1)
mpii_data = dataset.MPIIDataset('./data/MPII', './data/MPII/annotations', 16, 8,  1, 1)
#lets_dance = dataset.DanceDataset(pd.read_csv('./data/letsdance/val.csv'), 16, 8,  1,1)
datasets = [mpii_data]
for d in (datasets):
	for i in tqdm(range(len(d))):
		sample = d[i]
	print(mpii_data.unav_img)

for i in tqdm(range(len(coco_data))):
	sample = coco_data[i]

#for i in coco_data:
#	sample = coco_data[i]
#lets_dance_data = dataset.DanceDataset(pd.read_csv('./data/letsdance/test.csv'), 386, 386, 1, 1)
#print(len(coco_data), len(lets_dance_data))

dataloader
for i in tqdm(range(len(mpii_dataset))):
	sample = mpii_dataset[i]

print('total number of people is ', tot_people)
print('samples with joint information are ', samples_with_joints)
print('the dataset length is ', len(mpii_dataset))

data = []
t = {}
t['image'] = transforms.Compose([transforms.ToTensor()])

#data.append(dataset.MPIIDataset("./data/MPII", "./data/MPII/annotations.mat", 336, 1, 1, transform=t))

def get_mean_std(loader, device):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0,0,0
    for data in tqdm(loader):
        images = data['image'].clone().detach().to(device)
        channels_sum += torch.mean(images,dim=[0,2,3])
        channels_sqrd_sum += torch.mean(images**2, dim=[0,2,3])
        num_batches += 1
    mean = (channels_sum/num_batches)
    std = (channels_sqrd_sum/num_batches - mean**2)**0.5
    return mean, std


device = torch.device("cuda:1")
val_loader = lets_dance_loaders(386, 1, 2, 2, 1, 1)['val']
max_n_people = 0
for batch in tqdm(val_loader):
	if (batch['n_people'][0] > max_n_people):
		max_n_people = batch['n_people'][0]
print(max_n_people)


dataloaders = {}
for k in list(LD_data.keys()):
	dataloaders[k] = DataLoader(LD_data[k], batch_size=500, num_workers=20, shuffle=False)


for phase in dataloaders:
	loader = dataloaders[phase]
	mean , std = get_mean_std(loader, device)
	print("{} has the mean:".format(phase), mean)
	print("{} has the std: ".format(phase), std)
for k in dataloaders:
	d = dataloaders[k]
	dataloader_len = len(d)
	print(dataloader_len)
	batch_means, batch_stds = [], []
	for i, batch in enumerate(d):
		R_batch, G_batch, B_batch = batch['image'][:, 0, :, :].to(device), batch['image'][:, 1, :, :].to(device), batch['image'][:, 2, :, :].to(device)
		batch_means.append(torch.tensor([torch.mean(R_batch).item(), torch.mean(G_batch).item(), torch.mean(B_batch).item()]))
		batch_stds.append(torch.tensor([torch.std(R_batch).item(), torch.std(G_batch).item(), torch.std(B_batch).item()]))
		if ((i + 1) % 5 == 0):
			print((i+1)/dataloader_len, " is done")
			print(batch_means[i], batch_stds[i])
		#R_batch, B_batch, G_batch = torch.sum(batch['image'][:, 0, :, :].flatten(), batch['image'][:, 1, :, :].flatten(), batch['image'][:, 2, :, :].flatten()
	batch_means, batch_stds = torch.stack(batch_means), torch.stack(batch_stds)
	final_means, final_stds = torch.sum(batch_means, dim=0) / dataloader_len, torch.sum(batch_stds, dim=0) / dataloader_len
	print('The {} partition. '.k, final_means, final_stds)

"""
