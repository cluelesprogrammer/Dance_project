import argparse
import os
import models.models as models
import torch.optim as optim
import torch
import datasets.dataset as dataset
from torchvision import transforms, utils
import random
import numpy as np
from PIL import Image
import time
import pandas as pd
import torch.nn as nn
from datasets.dataset import GroundTruthGenerator as GT
import torch.optim as optim
import copy
import math
from functools import partial
import torch.nn.functional as F
import configparser
import sys
from data_loaders import dance_video_dataloaders
import wandb
from tqdm import tqdm


def train_val(model, dataloaders, optimizer, scheduler, device, epochs_start, epochs_to_train, run_info, best_accuracy=0, optical_flow=None, find_lr_range=False):
	model.to(device)
	loss_function = nn.CrossEntropyLoss()
	wandb.watch(model, log_freq=100)
	dance_types = list(dataset.DanceVideoDataset.label_to_id.keys())
	test = False
	for epoch in range(epochs_start, epochs_start+epochs_to_train):
			print('Starting epoch {}'.format(epoch+1))
			model.train()
			wandb.log({'epoch': epoch})
			for data in tqdm(dataloaders['train']):
				frames, targets = data['frames'].to(device), data['dance_type'].to(device)
				optimizer.zero_grad()
				if (optical_flow):
					flows = data['optical_flow'].to(device)
					outputs = model(frames, flows)
				else:
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
				if (optical_flow):
					flows = data['optical_flow'].to(device)
					with torch.no_grad():
						outputs = model(frames, flows)
				else:
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
			save_dir = 'models/classifier'
			if (accuracy > best_accuracy):
				best_accuracy = accuracy
				dict_to_save['model_state_dict'] = model.state_dict()
				dict_to_save['optimizer_state_dict'] = optimizer.state_dict()
				dict_to_save['scheduler_state_dict'] = scheduler.state_dict()
				dict_to_save['accuracy'] = accuracy
				dict_to_save['epoch'] = epoch
				checkpoint_path = '{}/{}_best_model.pt'.format(save_dir, run_info, epoch)
				wandb.log({'best_accuracy_acheived_at_epoch_nr': epoch})
				wandb.log({'best_accuracy_till_now': best_accuracy})
				best_checkpoint_path = checkpoint_path
				if (find_lr_range):
					print(checkpoint_path)
					torch.save(dict_to_save, checkpoint_path)
			if (epoch == epochs_to_train - 1):
				dict_to_save['model_state_dict'] = model.state_dict()
				dict_to_save['optimizer_state_dict'] = optimizer.state_dict()
				dict_to_save['scheduler_state_dict'] = scheduler.state_dict()
				dict_to_save['accuracy'] = accuracy
				dict_to_save['epoch'] = epoch
				checkpoint_path = '{}/{}_final_epoch.pt'.format(save_dir,run_info)
				if (find_lr_range):
					print(checkpoint_path)
					torch.save(dict_to_save, checkpoint_path)

def parse():
	parser = argparse.ArgumentParser(description='This is a training parser')
	parser.add_argument('batch_size', type=int, help='batch size')
	parser.add_argument("gpu_number", type=int, help="which GPU to use")
	parser.add_argument("epochs", type=int, help="epochs to train for")
	parser.add_argument("-c", "--checkpoint_path", type=str, help="checkpoint to load model and other dict from")
	parser.add_argument('-e_resume', '--e_resume', type=int, help='where the model was trained till')
	parser.add_argument('-sch_steps', '--sch_steps', type=int, help='the number of steps that scheduler has taken')
	parser.add_argument('-name', '--run_name', type=str, help='run name for saving model and wandb logging')
	args = parser.parse_args()
	return args

def get_model(model_info, flow_every):
	model_type = model_info['model_type']
	train_rgb, train_flow, train_pose = model_info.getboolean('train_rgb'), model_info.getboolean('train_flow'), model_info.getboolean('train_pose')
	if (model_type == 'c3d'):
		model = models.C3D_Classifier(train_whole=model_info.getboolean('train_whole'))
	elif (model_type == 'rgb_pose'):
		model = models.RGB_Bodypose_Classifier(train_features=(train_rgb, train_pose))
	elif (model_type == 'rgb_flow'):
		model = models.RGB_Flow_Classifier(train_features=(train_rgb, train_flow), flow_every=flow_every)
	elif (model_type == 'flow_pose'):
		model = models.Flow_Bodypose_Classifier(train_flow=train_flow, flow_every=flow_every)
	elif (model_type == 'flow'):
		model = models.Flow_Classifier(train_features=train_flow, flow_every=flow_every)
	elif (model_type == 'pose'):
		model = models.Bodypose_Classifier()
	else:
		print('not recognizable_model')
	return model

def get_opt_sch(model, training_details, sch_steps=None):
	opt_type = training_details['optimizer']
	min_lr, max_lr, momentum, cycle_len = float(training_details['min_lr']), float(training_details['max_lr']), float(training_details['momentum']), int(training_details['cycle_len'])

	if (opt_type == 'adamw'):
		optimizer = optim.AdamW(model.parameters(), lr=max_lr)
	elif (opt_type == 'sgd'):
		optimizer = optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)
	else:
		optimizer = None
		print('only adamw and sgd optimizers supported')

	sch_type = training_details['scheduler']
	if (opt_type == 'adamw'):
		scheduler = None
	else:
		if (sch_type == 'cyclic'):
			scheduler = optim.lr_scheduler.CyclicLR(optimizer, min_lr, max_lr, step_size_up=cycle_len, mode='triangular2')
		elif (sch_type == 'warmrestarts'):
			scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cycle_len, T_mult=3)
		else:
			scheduler = None
			print('only cyclic and warmrestarts supported')
	if (sch_steps):
		for i in range(sch_steps):
			scheduler.step()
	return optimizer, scheduler

def checkpoint_settings(checkpoint, model, optimizer, scheduler):
	model = model.load_state_dict(checkpoint['model_state_dict'])
	#optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	#if (scheduler):
	#	scheduler = scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	#return model
	return model, optimizer, scheduler

if __name__ == "__main__":
	#parsing and reading configuration from file, and making wandb configuration dictionary to log
	torch.manual_seed(0)
	args = parse()
	config = configparser.ConfigParser()
	config.read('classifier_config.ini')
	dict_list = [dict(config[k]) for k in ['DATASETINFO', 'MODELINFO', 'TRAININGDETAILS']]
	training_details, dataset_info = config['TRAININGDETAILS'], config['DATASETINFO']

	wandb_config = {}
	wandb_config = dict_list[0]
	wandb_config.update(dict_list[1])
	wandb_config.update(dict_list[2])
	wandb_config['batch_size'] = args.batch_size

	model = get_model(config['MODELINFO'], dataset_info.getint('flow_every'))
	device = torch.device('cuda:{}'.format(args.gpu_number))

	train_df, val_df = pd.read_csv('data/letsdance/train_videos.csv'), pd.read_csv('data/letsdance/val_videos.csv')
	dataloaders = dance_video_dataloaders(train_df, val_df, config.getint('DATASETINFO','frame_size'), args.batch_size, n_frames=config.getint('DATASETINFO', 'n_frames'), frames_to_skip=config.getint('DATASETINFO', 'frames_to_skip'), optical_flow=config.getboolean('DATASETINFO', 'optical_flow'), flow_every=config.getint('DATASETINFO', 'flow_every'))

	optimizer, scheduler = get_opt_sch(model, training_details, sch_steps=args.sch_steps)

	epochs_start = 0
	best_accuracy=0.0
	if (args.checkpoint_path):
		if (os.path.isfile(args.checkpoint_path)):
			print('checkpoint_found')
			checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
			model, optimizer, scheduler = checkpoint_settings(checkpoint, model, optimizer, scheduler)
			epochs_start = args.e_resume
			try:
				best_accuracy = checkpoint['accuracy']
			except:
				best_accuracy = 0.0
				print('best accuracy not logged in checkpoint')
		else:
			print('invalid checkpoint path. enter command again')
	if (args.run_name):
		model_info = args.run_name
	else:
		model_info = '{}_{}_{}_{}'.format(wandb_config['model_type'], wandb_config['optimizer'] ,wandb_config['scheduler'], wandb_config['flow_every'])
	wandb.init(project = 'lr_test', name=model_info, config=wandb_config)
	train_val(model, dataloaders, optimizer, scheduler, device, epochs_start, args.epochs, model_info, best_accuracy=best_accuracy, optical_flow=config.getboolean('DATASETINFO', 'optical_flow'))

