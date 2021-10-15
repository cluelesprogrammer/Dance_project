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
	#with wandb.init(project='training_dance_classifiers', config=config, name='C3d_skip_3_frames', dir='wandbdir'):
	with wandb.init(project='training_dance_classifiers', config=config, name='c3d_model_skip_0_frames', dir='wandbdir'):
		config = wandb.config
		run_info = 'c3d_skipping_0_frames_checkpoint_training'
		wandb.run.name = run_info
		print('-' *20, 'dataset preparation', '-'*20)
		train_df, val_df = pd.read_csv('./data/letsdance/train_videos.csv'), pd.read_csv('data/letsdance/val_videos.csv')
		dataloaders = dance_video_dataloaders(train_df, val_df, config.img_size, config.batch_size, n_frames=16, frames_to_skip=0)
		device = torch.device('cuda:2')
		checkpoint = torch.load('models/c3d_models/c3d_skipping_0_frames_final_epoch.pt')
		model = models.C3D_Classifier()
		model.load_state_dict(checkpoint['model_state_dict'])
		loss_function = nn.CrossEntropyLoss()
		model.to(device)
		wandb.watch(model)
		dance_types = list(dataset.DanceVideoDataset.label_to_id.keys())
		#checkpoint = torch.load('models/c3d_models/c3d_C3d_skip_0_frames_epoch_44.pt', map_location=torch.device('cpu'))
		#model.load_state_dict(checkpoint['model_state_dict'])
		optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-3, 5e-3, step_size_up=600, mode='triangular2')
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		test = False
		best_accuracy=48.123
		for epoch in range(199, config.epochs + 100):
			print(f'Starting epoch {epoch+1}')
			model.train()
			wandb.log({'epoch': epoch})
			for data in tqdm(dataloaders['train']):
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
			for data in tqdm(dataloaders['val']):
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
				dict_to_save['scheduler_state_dict'] = scheduler.state_dict()
				dict_to_save['accuracy'] = accuracy
				dict_to_save['epoch'] = epoch
				checkpoint_path = 'models/c3d_models/{}_best_model.pt'.format(run_info, epoch)
				wandb.log({'best_accuracy_acheived_at_epoch_nr': epoch})
				wandb.log({'best_accuracy_till_now': best_accuracy})
				best_checkpoint_path = checkpoint_path
				torch.save(dict_to_save, checkpoint_path)
			if (epoch == config.epochs-1):
				dict_to_save['model_state_dict'] = model.state_dict()
				dict_to_save['optimizer_state_dict'] = optimizer.state_dict()
				dict_to_save['scheduler_state_dict'] = scheduler.state_dict()
				dict_to_save['accuracy'] = accuracy
				dict_to_save['epoch'] = epoch
				checkpoint_path = 'models/c3d_models/{}_final_epoch.pt'.format(run_info)
				torch.save(dict_to_save, checkpoint_path)

config = {'batch_size': 24, 'epochs': 300, 'img_size': 112, 'optimizer':'sgd', 'scheduler': 'cyclic', 'model':'c3d', 'lr':1e-3, 'frames_to_skip':3}
train_val(config)

