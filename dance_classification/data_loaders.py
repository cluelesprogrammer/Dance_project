import datasets.dataset as dataset
from torchvision import transforms, utils
import torch
from torch.utils.data import DataLoader, Subset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def concatenated_dataloaders(load_size, return_size, batch_size, limb_width, sigma, normalize=True):
	torch.manual_seed(0)
	lets_dance_train_mean, lets_dance_train_std = [0.4100, 0.3402, 0.3244], [0.2834, 0.2593, 0.2477]
	lets_dance_val_mean, lets_dance_val_std = [0.4105, 0.3408, 0.3253], [0.2834, 0.2595, 0.2479]
	transform_coco_train, transform_mpii_train, transform_lets_dance_train, transform_coco_val, transform_lets_dance_val = {}, {}, {}, {}, {}
	if (normalize):
		transform_coco_train['image'] = transforms.Compose([transforms.Normalize(mean=[0.4634, 0.4463, 0.4182], std=[0.2777, 0.2724, 0.2863]), transforms.RandomGrayscale()])
		transform_coco_val['image'] = transforms.Compose([transforms.Normalize(mean=[0.4627, 0.4454, 0.4162], std=[0.2789, 0.2737, 0.2862]), transforms.RandomGrayscale()])
		transform_mpii_train['image'] = transforms.Compose([transforms.Normalize(mean=[0.4680, 0.4497, 0.4127], std=[0.2715, 0.2671, 0.2712]), transforms.RandomGrayscale()])
		transform_lets_dance_train['image'] = transforms.Compose([transforms.Normalize(mean=lets_dance_train_mean, std=lets_dance_train_std), transforms.RandomGrayscale()])
		transform_lets_dance_val['image'] = transforms.Compose([transforms.Normalize(mean=lets_dance_val_mean, std=lets_dance_val_std), transforms.RandomGrayscale()])
	coco_train_dataset = dataset.COCODataset('./data/coco', 'train', load_size, return_size, limb_width, sigma, transform=transform_coco_train)
	coco_val_dataset = dataset.COCODataset('./data/coco', 'val', load_size, return_size, limb_width, sigma, transform=transform_coco_val)
	mpii_train_dataset  = dataset.MPIIDataset('./data/MPII', './data/MPII/annotations.mat', load_size, return_size, limb_width, sigma, transform=transform_mpii_train)
	lets_dance_train_dataset = dataset.DanceDataset(pd.read_csv('./data/letsdance/train.csv'), load_size, return_size, limb_width, sigma,transform=transform_lets_dance_train)
	indices = torch.randperm(len(lets_dance_train_dataset))[:50000]
	lets_dance_train_subset = Subset(lets_dance_train_dataset, indices)
	lets_dance_val_dataset = dataset.DanceDataset(pd.read_csv('./data/letsdance/val.csv'), load_size, return_size, limb_width, sigma,transform=transform_lets_dance_val)
	#lets_dance_test_dataset = dataset.DanceDataset(pd.read_csv('./data/letsdance/test.csv'), load_size, return_size, limb_width, sigma,transform=transform_lets_dance_val)
	concat_train_dataset = torch.utils.data.ConcatDataset([coco_train_dataset, mpii_train_dataset, lets_dance_train_subset])
	concat_val_dataset = torch.utils.data.ConcatDataset([coco_val_dataset, lets_dance_val_dataset])
	dataloaders = {}
	dataloaders['train'] = DataLoader(concat_train_dataset, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	dataloaders['coco_val'] = DataLoader(coco_val_dataset, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	dataloaders['lets_dance_val'] = DataLoader(lets_dance_val_dataset, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	#dataloaders['coco_val'] = DataLoader(coco_val_dataset, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	#dataloaders['lets_dance_val'] = DataLoader(lets_dance_val_dataset, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	return dataloaders

def split_videos():
	df = pd.read_csv('data/letsdance/videos.csv')
	inp_ids, labels = list(df.index), np.array([df['dance_type'].iloc[i] for i in range(len(df.index))])
	train_ids, val_ids, train_labels, val_labels = train_test_split(inp_ids, labels, test_size=0.2, stratify=labels, random_state=2)
	train_df, val_df = df.iloc[train_ids], df.iloc[val_ids]
	train_df.to_csv('data/letsdance/train_videos.csv', index=False)
	val_df.to_csv('data/letsdance/val_videos.csv', index=False)

def dance_video_dataloaders(train_df, val_df, return_size, batch_size, n_frames=64, frames_to_skip=0, optical_flow=None,flow_every=8):
	mean, std = [0.4100, 0.3402, 0.3244], [0.2834, 0.2593, 0.2477]
	transform = {}
	transform['frames'] = transforms.Compose([transforms.CenterCrop(return_size), transforms.Normalize(mean=mean, std=std), transforms.RandomGrayscale(), transforms.ColorJitter()])
	if (optical_flow):
		transform['optical_flow'] = transforms.Compose([transforms.CenterCrop(return_size)])
	dataloaders = {}
	if train_df is not None:
		train_data = dataset.DanceVideoDataset(return_size, train_df, n_frames=n_frames, frames_to_skip=frames_to_skip, transform=transform, optical_flow=optical_flow, flow_every=flow_every)
		dataloaders['train'] = DataLoader(train_data, batch_size=batch_size, num_workers=12, pin_memory=True, shuffle=True)
	if val_df is not None:
		val_data = dataset.DanceVideoDataset(return_size, val_df, n_frames=n_frames, transform=transform, optical_flow=optical_flow, flow_every=flow_every)
		dataloaders['val'] = DataLoader(val_data, batch_size=batch_size, num_workers=12, pin_memory=True, shuffle=True)
	return dataloaders



"""
def concatenated_dataloaders(img_size, batch_size, limb_width, sigma, normalize, rotation):


class coco_mpii_lets_dance_dataloaders():
	def __init__(self, img_size, batch_size, limb_width, sigma, normalize, rotation):
		self.coco_mpii_loaders = coco_mpii_loaders(img_size, batch_size, limb_width, sigma, normalize, rotation)
		self.lets_dance_loaders = lets_dance_loaders(img_size, batch_size, limb_width, sigma, normalize, rotation)
		self.train_called_index = torch.zeros(self.val_len())
		self.val_called_index = torch.zero(self.train_len())
	def train_batch(self, idx):
		self.train_called_index[idx] = 1
	def val_batch(self, idx):

	def train_len(self):
		return len(self.coco_mpii_loaders['coco_train']) + len(self.coco_mpii_loaders['mpii_train']) + len(self.lets_dance_loaders['train'])
	def val_len(self):
		return len(self.coco_mpii_loaders['coco_val']) + len(self.lets_dance_loaders['val'])
"""

def coco_mpii_dataloaders(load_size, return_size, batch_size, limb_width, sigma, normalize=True):
	transform_mpii_train, transform_coco_train, transform_coco_val = {}, {}, {}
	if (normalize):
		#transforms.Normalize(mean=[0.2679, 0.2126, 0.2496], std=[0.2720, 0.2717, 0.2675])
		transform_mpii_train['image'] = transforms.Compose([transforms.Normalize(mean=[0.4680, 0.4497, 0.4127], std=[0.2715, 0.2671, 0.2712]), transforms.RandomGrayscale()])
		transform_coco_train['image'] = transforms.Compose([transforms.Normalize(mean=[0.4634, 0.4463, 0.4182], std=[0.2777, 0.2724, 0.2863]), transforms.RandomGrayscale()])
		transform_coco_val['image'] = transforms.Normalize(mean=[0.4627, 0.4454, 0.4162], std=[0.2789, 0.2737, 0.2862])
	datasets = {}
	coco_train_dataset = dataset.COCODataset('./data/coco', 'train', load_size, return_size, limb_width, sigma, transform=transform_coco_train)
	coco_val_dataset = dataset.COCODataset('./data/coco', 'val', load_size, return_size, limb_width, sigma, transform=transform_coco_val)
	mpii_train_dataset = dataset.MPIIDataset('./data/MPII', './data/MPII/annotations.mat', load_size, return_size, limb_width, sigma, transform=transform_mpii_train)
	coco_mpii_train = torch.utils.data.ConcatDataset([coco_train_dataset, mpii_train_dataset])
	dataloaders = {}
	dataloaders['train'] = DataLoader(coco_mpii_train, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	dataloaders['val'] = DataLoader(coco_val_dataset, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	"""
	dataloaders = {}
	for k in datasets:
		dataloaders[k] = DataLoader(datasets[k], batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	"""
	return dataloaders

def mpii_dataloaders(load_size, return_size, batch_size, limb_width, sigma, normalize=True):
	transform_mpii = {}
	if (normalize):
		transform_mpii['image'] = transforms.Normalize(mean=[0.4680, 0.4497, 0.4127], std=[0.2715, 0.2671, 0.2712])
	dataloaders = {}
	for k in datasets:
		dataloaders[k] = DataLoader(datasets[k], batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	return dataloaders

def coco_dataloaders(load_size, return_size, batch_size, limb_width, sigma, normalize=True):
	transform_coco_train, transform_coco_val = {}, {}
	if (normalize == 1):
		transform_coco_train['image'] = transforms.Compose([transforms.Normalize(mean=[0.4634, 0.4463, 0.4182], std=[0.2777, 0.2724, 0.2863]), transforms.RandomGrayscale(),transforms.ColorJitter()])
		transform_coco_val['image'] = transforms.Compose([transforms.Normalize(mean=[0.4627, 0.4454, 0.4162], std=[0.2789, 0.2737, 0.2862]), transforms.RandomGrayscale(), transforms.ColorJitter()])
	datasets = {}
	datasets['train'] = dataset.COCODataset('./data/coco', 'train', load_size, return_size, limb_width, sigma, transform=transform_coco_train)
	datasets['val'] = dataset.COCODataset('./data/coco', 'val', load_size, return_size, limb_width, sigma, transform=transform_coco_val)
	dataloaders = {}
	for k in datasets:
		dataloaders[k] = DataLoader(datasets[k], batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
	return dataloaders


def lets_dance_dataloaders(load_size, return_size, batch_size, limb_width, sigma, normalize=True):
		transform_train, transform_val, transform_test = {}, {}, {}
		train_pd, val_pd, test_pd = pd.read_csv('./data/letsdance/train.csv'), pd.read_csv('./data/letsdance/val.csv'), pd.read_csv('./data/letsdance/test.csv')
		means, stds = {}, {}
		train_mean, train_std = [0.4100, 0.3402, 0.3244], [0.2834, 0.2593, 0.2477]
		val_mean, val_std = [0.4105, 0.3408, 0.3253], [0.2834, 0.2595, 0.2479]
		test_mean, test_std = [0.4113, 0.3412, 0.3255], [0.2835, 0.2595, 0.2479]
		if (normalize == 1):
				transform_train['image'] = transforms.Compose([transforms.Normalize(mean=train_mean, std=train_std), transforms.RandomGrayscale()])
				transform_val['image'] = transforms.Compose([transforms.Normalize(mean=val_mean, std=val_std), transforms.RandomGrayscale()])
				transform_test['image'] = transforms.Compose([transforms.Normalize(mean=test_mean, std=test_std), transforms.RandomGrayscale()])
		train_data, test_data, val_data = dataset.DanceDataset(train_pd, load_size, return_size, limb_width, sigma,transform=transform_train), \
										dataset.DanceDataset(test_pd, load_size, return_size, limb_width, sigma,transform=transform_test), \
										dataset.DanceDataset(val_pd, load_size, return_size, limb_width, sigma,transform=transform_val)
		dataloaders = {}
		dataloaders['train'] = DataLoader(train_data, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
		dataloaders['val'] = DataLoader(val_data, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
		dataloaders['test'] = DataLoader(test_data, batch_size=batch_size, num_workers=11, shuffle=True, pin_memory=True)
		return dataloaders
