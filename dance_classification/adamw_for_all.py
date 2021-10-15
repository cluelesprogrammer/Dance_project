import torch
import torch
import numpy as np
import time
import argparse
import wandb
import yaml
from tqdm import tqdm
from loss import paf_loss, conf_loss
from data_loader import lets_dance_loaders, coco_mpii_loaders, concatenated_dataloaders
from models_loader import get_model
#from opt_sch_loaders import get_sgd_opt, get_adam_opt, get_one_cycle_sch, get_cyclic_sch, get_warm_restarts_sch
from datasets.dataset import GroundTruthGenerator as G
import torch.optim as optim
import math
from train import train_epoch, val_epoch

def adamw_find(config):
	with wandb.init(config=config):
		config = wandb.config
		device = torch.device('cuda:1')
		dataloader = coco_mpii_loaders(384, config['batch_size'], 3, 2, 1, 1)
		paf_activation, conf_activation = None, None
		#train_features = bool(config['train_features'])
		if (config['paf_activation'] == 'tanh'):
			paf_activation = 'tanh'
		if (config['conf_activation'] == 'sigmoid'):
			conf_activation = 'sigmoid'
		model = get_model('new', True, final_activations=(paf_activation, conf_activation)).to(device)
		#features_param = model.features_extractor.parameters()
		optimizers = [optim.AdamW(model.parameters())]
		schedulers=[None]
		print('after activation')
		min_val_loss, min_paf_val_loss, min_conf_val_loss = math.inf, math.inf, math.inf
		wandb.watch(model)
		for epoch in range(config['tot_epochs']):
			train_epoch(model, optimizers, schedulers, dataloader['train'], device)
			epoch_val_loss, epoch_paf_val_loss, epoch_conf_val_loss = val_epoch(model, dataloader['val'], device, epoch)
			if ((epoch_val_loss < min_val_loss) and (epoch_paf_val_loss < min_paf_val_loss) and (epoch_conf_val_loss < min_conf_val_loss)):
				min_val_loss, min_paf_val_loss, min_conf_val_loss = epoch_val_loss, epoch_paf_val_loss, epoch_conf_val_loss
			else:
				break
				print('did not cause lower val_loss and paf_loss and conf_val loss')
			dict_to_save = {}
			dict_to_save['model_state_dict'] = model.state_dict()
			for i, opt in enumerate(optimizers):
				dict_to_save['opt_{}_state_dict'.format(i)] = opt.state_dict()
			for i, sch in enumerate(schedulers):
				if (sch):
					dict_to_save['sch_{}_state_dict'.format(i)] = sch.state_dict()
			saved_file_path = './models/admaw_for_all_epoch_{}.pt'.format(epoch)
			torch.save(dict_to_save, saved_file_path)
			print(saved_file_path)
		wandb.log({'min_val_loss': min_val_loss})
		return min_val_loss

config = {'batch_size': 16, 'paf_activation': 'tanh', 'conf_activation': 'sigmoid', 'tot_epochs': 7, 'train_features': 'adamw'}
adamw_find(config)
