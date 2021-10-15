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
from opt_sch_loaders import get_cyclic_sch, get_warm_restarts_sch, MultipleOptimizer, MultipleScheduler
from datasets.dataset import GroundTruthGenerator as G
import torch.optim as optim
import math
from train import train_epoch, val_epoch

device = torch.device('cuda:2')

min_runs_val_loss, min_runs_paf_val_loss, min_runs_conf_val_loss = math.inf, math.inf, math.inf


def adamw_find(config=None):
	with wandb.init(config=config, project='THE_experiments', entity='sbaral'):
		config = wandb.config
		#dataloader = coco_mpii_loaders(448, 384, config['batch_size'], 3, 2, 1)
		dataloader = concatenated_dataloaders(448, 384, config['batch_size'], 3, 2, True)
		epoch_nr = 0
		#checkpoint = torch.load('./models/adamw/coco_mpii_letsdance_None_None_ftrs_lr_5e-08_epoch_{}.pt'.format(epoch_nr))
		#train_features = bool(config['train_features'])
		model = get_model('new', config['train_features'], paf_stages=config['paf_stages'], conf_stages=config['conf_stages'], final_activations=(config['paf_activation'], config['conf_activation'])).to(device)
		#model.load_state_dict(checkpoint['model_state_dict'])
		#checkpoint = torch.load('./models/adamw/adamw_sgd_None_None_ftrs_lr_1e-07_epoch_3.pt')
		#model.load_state_dict(checkpoint['model_state_dict'])
		if (config['train_features']):
			features_param = model.features_extractor.parameters()
		paf_params, conf_params = [], []
		all_layers = list(model.children())[0]
		features_layer = all_layers[0]
		model_layers = all_layers[1:]
		"""
		for m in model.paf_models:
			paf_params.append(list(m.parameters()))
		for m in model.conf_models:
			conf_params.append(list(m.parameters()))
		"""
		if (config['train_features']):
			opt_ftrs = optim.SGD(features_layer.parameters(), lr=config['features_lr'], momentum=0.95)
			sch_ftrs = optim.lr_scheduler.StepLR(opt_ftrs, 2)
			#sch_ftrs = optim.StepLR(
			#sch_ftrs = get_cyclic_sch(opt_ftrs, config['features_lr']/10, config['features_lr'], int(len(dataloader['train'])/10), len(dataloader['train']))
		opt_model = optim.SGD(model_layers.parameters(), lr=1e-3, momentum=0.9)
		sch_model = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_model, int(len(dataloader['train']) * 0.1), 2)
		#opt_model.load_state_dict(checkpoint['opt_state_dict_model'])
		#opt_ftrs.load_state_dict(checkpoint['opt_state_dict_ftrs'])
		#sch_ftrs.load_state_dict(checkpoint['sch_state_dict_ftrs'])
		#opt_paf, opt_conf = optim.AdamW(paf_params), optim.AdamW(conf_params)
		optimizers = MultipleOptimizer(opt_ftrs, opt_model)
		schedulers = MultipleScheduler(sch_ftrs, sch_model)
		#optimizers, schedulers = MultipleOptimizer(optim.AdamW(model.parameters())), MultipleScheduler(None)
		#print('after activation')
		#weighted_min_val_loss, weighted_min_paf_val_loss, weighted_min_conf_val_loss = math.inf, math.inf, math.inf
		min_val_loss, min_paf_val_loss, min_conf_val_loss = math.inf, math.inf, math.inf
		best_epoch = None
		wandb.watch(model)
		for epoch in range(epoch_nr, config['tot_epochs']):
			train_epoch(model, optimizers, schedulers, dataloader['train'], device, epoch)
			val_loss, paf_val_loss, conf_val_loss = val_epoch(model, dataloader['val'], device, epoch, 'coco')
			#letsdance_val_loss, letsdance_paf_val_loss, letsdance_conf_val_loss = val_epoch(model, dataloader['lets_dance_val'], device, epoch, 'letsdance')
			#weighted_val_loss = 0.5 * coco_val_loss + 0.5 * letsdance_val_loss
			#weighted_paf_val_loss = 0.5 * coco_paf_val_loss + 0.5 * letsdance_paf_val_loss
			#weighted_conf_val_loss = 0.5 * coco_conf_val_loss + 0.5 * letsdance_conf_val_loss
			if (val_loss > min_val_loss):
				print('greater epoch val loss than previous')
			else:
				#weighted_min_val_loss = weighted_val_loss
				min_val_loss = val_loss
				best_epoch = epoch
				#if ((coco_paf_val > weighted_min_paf_val_loss) and (weighted_conf_val_loss > weighted_min_conf_val_loss)):
				if ((paf_val_loss > min_paf_val_loss) and (conf_val_loss > min_conf_val_loss)):
					print('did not cause either paf or conf val loss to decrease')
				if (paf_val_loss < min_paf_val_loss):
					min_paf_val_loss = paf_val_loss
				if (conf_val_loss < min_conf_val_loss):
					min_conf_val_loss = conf_val_loss
			dict_to_save = {}
			dict_to_save['model_state_dict'] = model.state_dict()
			dict_to_save['opt_state_dict_ftrs'] = optimizers.get(0).state_dict()
			dict_to_save['opt_state_dict_model'] = optimizers.get(1).state_dict()
			dict_to_save['sch_state_dict_ftrs'] = schedulers.get(0).state_dict()
			dict_to_save['best_epoch'] = best_epoch
			"""
			for i, opt in enumerate(optimizers):
				dict_to_save['opt_{}_state_dict'.format(i)] = opt.state_dict()
			for i, sch in enumerate(schedulers):
				if (sch):
					dict_to_save['sch_{}_state_dict'.format(i)] = sch.state_dict()
			"""
			saved_file_path = './models/sgdwarmrestarts/letsdancecocompii_{}_{}_ftrs_lr_{}_epoch_{}.pt'.format(config['paf_activation'], config['conf_activation'], config['features_lr'], epoch)
			print(saved_file_path)
			torch.save(dict_to_save, saved_file_path)
		wandb.log({'min_val_loss': min_val_loss})
		return min_val_loss

#"" c = yaml.full_load(open('adam_ftrs_lr.yaml')) sweep_id = wandb.sweep(c, project='the_adamw_experiments') wandb.agent(sweep_id, adamw_find)
config = {'batch_size': 64, 'paf_stages':5, 'conf_stages':2, 'conf_activation':None, 'paf_activation': None, 'tot_epochs':20, 'optimizer': 'adamwforall', 'train_features': True, 'features_lr': 5e-8, 'dataset': 'letsdanceocompii'}
adamw_find(config)
