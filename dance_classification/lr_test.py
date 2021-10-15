import models.models as models
import datasets.dataset as dataset
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import sys
from data_loaders import dance_video_dataloaders

def train_val(config):
	with wandb.init(project='training_dance_classifier', config=config, name='C3d_skip_0_frame', dir='wandbdir'):
		config = wandb.config
		print('-' *20, 'dataset preparation', '-'*20)
		mean, std = [0.4100, 0.3402, 0.3244], [0.2834, 0.2593, 0.2477]
		transform = {}
		transform['frames'] = transforms.Compose([transforms.CenterCrop(config.img_size), transforms.Normalize(mean=mean, std=std), transforms.RandomGrayscale(), transforms.ColorJitter()])
		#transform['optical_flow'] = transforms.CenterCrop(config.img_size)
		df = pd.read_csv('./data/letsdance/videos.csv')
		inp_ids, labels = list(df.index), np.array([df['dance_type'].iloc[i] for i in range(len(df.index))])
		train_ids, val_ids, train_labels, val_labels = train_test_split(inp_ids, labels, test_size=0.2, stratify=labels, random_state=2)
		train_df, val_df = df.iloc[train_ids], df.iloc[val_ids]
		train_data = dataset.DanceVideoDataset(config.img_size, train_df, frames_to_skip=0, n_frames=16, transform=transform)
		val_data = dataset.DanceVideoDataset(config.img_size, val_df, frames_to_skip=0, n_frames=16, transform=transform)
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, num_workers=12, pin_memory=True, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.batch_size, num_workers=12, pin_memory=True, shuffle=True)
		device = torch.device('cuda:2')
		model = models.C3D_Classifier()
		loss_function = nn.CrossEntropyLoss()
		model.to(device)
		wandb.watch(model)
		dance_types = list(train_data.dance_type_dict.keys())
		optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=2e-2, step_size_up=1000)
		test = False
		best_accuracy=0
		for epoch in range(config.epochs):
			print(f'Starting epoch {epoch+1}')
			model.train()
			wandb.log({'epoch': epoch})
			for data in tqdm(train_loader):
				frames, targets = data['frames'].to(device), data['dance_type'].to(device)
				optimizer.zero_grad()
				outputs = model(frames)
				loss = loss_function(outputs, targets)
				loss.backward()
				optimizer.step()
				scheduler.step()
				wandb.log({'lr': optimizer.param_groups[0]['lr'], 'loss':loss.item()})
				if (test):
					break
			correct, total = 0, 0
			model.eval()
			val_targets, val_predictions = [], []
			correct_style_predictions = dict(zip(range(16), [0]*16))
			for data in tqdm(val_loader):
				frames, targets = data['frames'].to(device), data['dance_type'].to(device)
				with torch.no_grad():
					outputs = model(frames)
				_, predicted = torch.max(outputs.data, 1)
				val_predictions.append(predicted.cpu().numpy())
				style, correct_instances = torch.unique(predicted[predicted==targets], return_counts=True)
				correct_in_batch = dict(zip(style.cpu().numpy(), correct_instances.cpu().numpy()))
				for c in correct_in_batch:
					correct_style_predictions[c] += correct_in_batch[c]
				total += targets.size(0)
				correct += (predicted == targets).sum().item()
				if (test):
					break
			accuracy = 100.0 * correct/total
			wandb.log({'accuracy': accuracy})
			dict_to_save = {}
			if (accuracy > best_accuracy):
				best_accuracy = accuracy
				dict_to_save['model_state_dict'] = model.state_dict()
				dict_to_save['optimizer_state_dict'] = optimizer.state_dict()
				checkpoint_path = 'models/c3d_models/c3d_{}_epoch_{}.pt'.format(wandb.run.name, epoch)
				wandb.log({'best_accuracy_acheived_at_epoch_nr': epoch})
				wandb.log({'best_accuracy_till_now': best_accuracy})
				if (epoch > 10):
					torch.save(dict_to_save, checkpoint_path)

config = {'batch_size': 24, 'epochs': 100, 'img_size': 112, 'optimizer':'sgd', 'scheduler': 'cyclic', 'optimizer':'sgd', 'model':'c3d', 'frames_to_skip':0, 'momentum':0.9, 'lr': 1e-3}
train_val(config)

