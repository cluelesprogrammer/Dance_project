import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import wandb
import yaml
from tqdm import tqdm
from loss import mse_paf_loss, mse_conf_loss, paf_loss, conf_loss
from data_loaders import lets_dance_dataloaders, coco_mpii_dataloaders, concatenated_dataloaders
from models_loader import get_model
from datasets.dataset import GroundTruthGenerator as G
import torch.optim as optim
import math

def train_epoch(model, optimizers, schedulers, dataloader, device, epoch, cosine_annealing=False, test=False):
	model.train()
	n_batches_loss, n_batches_paf_loss, n_batches_conf_loss, log_freq = 0.0, 0.0, 0.0, 1
	i = epoch * len(dataloader) + 1
	#clip_value = 0.01
	for batch in tqdm(dataloader):
		optimizers.zero_grad()
		images, paf_truths, conf_truths, paf_mask, conf_mask = batch['image'].float().to(device), batch['pafs'].float().to(device), batch['confs'].float().to(device), batch['paf_mask'][0].to(device), batch['conf_mask'][0].to(device)
		paf_preds, conf_preds = model(images)
		paf_l, conf_l = paf_loss(paf_preds, paf_truths, paf_mask=paf_mask), conf_loss(conf_preds, conf_truths, conf_mask=conf_mask)
		loss = torch.sum(paf_l) + torch.sum(conf_l)
		loss.backward()
		optimizers.step()
		dict_to_log = {}
		if (i % log_freq == 0):
			for j, lr in enumerate(optimizers.get_lr()):
				dict_to_log['lr_{}'.format(j)] = lr[0]
			dict_to_log['training_batch_nr'] = i
			dict_to_log['training_loss'] = loss.detach().clone().item()
			dict_to_log['training_log_loss'] = math.log(loss.detach().clone().item())
			dict_to_log['training_paf_loss'] = torch.sum(paf_l.detach().clone()).item()
			dict_to_log['training_paf_log_loss'] = math.log(torch.sum(paf_l.detach().clone()).item())
			dict_to_log['training_conf_loss'] = torch.sum(conf_l.detach().clone()).item()
			dict_to_log['training_conf_log_loss'] = math.log(torch.sum(conf_l.detach().clone()).item())
			wandb.log(dict_to_log)
		i += 1
		if (test):
			break

def val_epoch(mod, dataloader, device, epoch, dataset, test=False, type='mid_training_validation'):
	mod.eval()
	mod.to(device)
	running_loss = 0.0
	paf_losses, conf_losses = [], []
	running_paf_loss = 0.0
	#if (epoch >= 0):
	#	i = len(dataloader) * epoch
	running_conf_loss = 0.0
	batch_size = None
	for batch in tqdm(dataloader):
		images, paf_truths, conf_truths, paf_mask, conf_mask = batch['image'].to(device), batch['pafs'].to(device), batch['confs'].to(device), batch['paf_mask'][0].to(device), batch['conf_mask'][0].to(device)
		batch_size = images.shape[0]
		with torch.no_grad():
			paf_preds, conf_preds = mod(images)
		paf_l, conf_l = paf_loss(paf_preds, paf_truths, paf_mask), conf_loss(conf_preds, conf_truths, conf_mask)
		paf_losses.append(paf_l)
		conf_losses.append(conf_l)
		loss = torch.sum(paf_l) + torch.sum(conf_l)
		dict_to_log = {}
		#dict_to_log['{}_batch_nr'.format(dataset)] = i
		dict_to_log['val_loss'.format(dataset)] = loss.item()
		dict_to_log['val_log_loss'.format(dataset)] = math.log(loss.item())
		dict_to_log['val_paf_loss'.format(dataset)] = torch.sum(paf_l).item()
		dict_to_log['val_paf_log_loss'.format(dataset)] = math.log(torch.sum(paf_l).item())
		dict_to_log['val_conf_loss'.format(dataset)] = torch.sum(conf_l).item()
		dict_to_log['val_conf_log_loss'.format(dataset)] = math.log(torch.sum(conf_l).item())
		wandb.log(dict_to_log)
		running_loss += loss.item()
		running_paf_loss += torch.sum(paf_l).item()
		running_conf_loss += torch.sum(conf_l).item()
		#i += 1
		if (test):
			break
	stage_mean_paf_loss, stage_mean_conf_loss = torch.mean(torch.stack(paf_losses), dim=0), torch.mean(torch.stack(conf_losses), dim=0)
	#stage_mean_paf_loss, stage_mean_conf_loss = torch.sum(torch.stack(paf_losses), dim=0), torch.sum(torch.stack(conf_losses), dim=0)
	total_batches = len(dataloader)
	data_paf_loss = [[layer, stage_paf_loss] for (layer, stage_paf_loss) in enumerate(stage_mean_paf_loss)]
	paf_table = wandb.Table(data=data_paf_loss, columns = ["layer", "layer_paf_loss"])
	paf_title = "stagewise_avg_paf_loss"
	conf_title = "stagewise_avg_conf_loss"
	if (type == 'mid_training_validation'):
		paf_title += '_{}'.format(epoch)
		conf_title += '_{}'.format(epoch)
	wandb.log({paf_title: wandb.plot.line(paf_table, "layer", "layer_paf_loss", title=paf_title)})
	data_conf_loss = [[layer, stage_conf_loss] for (layer, stage_conf_loss) in enumerate(stage_mean_conf_loss)]
	conf_table = wandb.Table(data=data_conf_loss, columns = ["layer", "layer_conf_loss"])
	wandb.log({conf_title: wandb.plot.line(conf_table, "layer", "layer_conf_loss", title=conf_title)})
	avg_val_loss, avg_val_paf_loss, avg_val_conf_loss = running_loss/total_batches, running_paf_loss/total_batches, running_conf_loss/total_batches
	avg_paf_stage_loss, avg_conf_stage_loss = torch.mean(stage_mean_paf_loss), torch.mean(stage_mean_conf_loss)
	wandb.log({'epoch':epoch, 'avg_val_loss':avg_val_loss, 'avg_val_paf_loss':avg_val_paf_loss, 'avg_conf_val_loss': avg_val_conf_loss})
	wandb.log({'avg_paf_stage_loss': avg_paf_stage_loss, 'avg_stage_conf_loss': avg_conf_stage_loss})
	wandb.log({'per_stage_val_loss': avg_val_loss / mod.total_stages})
	return avg_val_loss, avg_val_paf_loss, avg_val_conf_loss
	"""
	data_conf_loss = [[layer, layer_conf_loss] for (layer, layer_paf_loss) in enumerate(layerwise_mean_paf_loss)]
	for i, l in enumerate(layerwise_mean_paf_loss):
	wandb.log({'paf_layer':i, 'paf_layerwise_per_batch_val_loss': l})
	for i, l in enumerate(layerwise_mean_conf_loss):
		wandb.log({'conf_layer':i,'conf_layerwise_per_batch_val_loss': l})
	"""


def train_n_epochs(model, dataloaders, tot_epochs, device, optimizers, schedulers, model_info, test=False, checkpoint=False, epoch_start=0):
	min_val_loss, min_paf_val_loss, min_conf_val_loss = math.inf, math.inf, math.inf
	if (checkpoint):
		print('assuming that the model with lowest validation loss is selected')
		min_val_loss, min_paf_val_loss, min_conf_val_loss = val_epoch(model, dataloaders['val'], device, 0, test=test)
	#swa_model = optim.swa_utils.AveragedModel(model)
	#swa_scheduler = optim.swa_utils.SWALR(optimizers[0], swa_lr = 0.05)
	model_save_freq = 3
	for epoch in range(epoch_start, epoch_start + tot_epochs):
		train_epoch(model, optimizers, schedulers, dataloaders['train'], device, epoch, test=test)
		schedulers.step(0)
		#if (epoch > 0):
			#swa_scheduler.step()
		coco_val_loss, coco_paf_val_loss, coco_conf_val_loss = val_epoch(model, dataloaders['val'], device, epoch, 'combined', test=test)
		#lets_dance_val_loss, lets_dance_paf_val_loss, lets_dance_conf_val_loss = val_epoch(model, dataloaders['lets_dance_val'], device, epoch, 'combined_dataset', test=test)
		"""
		if (epoch_val_loss < min_val_loss):
			min_val_loss = epoch_val_loss
		else:
			print('val loss did not decrease after previous epoch. Time to terminate and tweak the hyperparameters')
			break
		if (epoch_paf_val_loss < min_paf_val_loss):
			min_paf_val_loss = epoch_paf_val_loss
		if (epoch_conf_val_loss < min_conf_val_loss):
                	min_conf_val_loss = epoch_conf_val_loss
		"""
		dict_to_save = {}
		dict_to_save['model_state_dict'] = model.state_dict()
		state_dicts_opt = optimizers.get_state_dicts()
		state_dicts_sch = schedulers.get_state_dicts()
		for i, s in enumerate(state_dicts_opt):
			dict_to_save['opt_{}_state_dict'.format(i)] = s
		for i, s in enumerate(state_dicts_sch):
			dict_to_save['sch_{}_state_dict'.format(i)] = s
		saved_file_path = './models/newer_openpose_model/{}_{}_{}_epoch_{}.pt'.format(model_info, model.paf_stages, model.conf_stages, epoch)
		if (epoch % model_save_freq == 0):
			torch.save(dict_to_save, saved_file_path)
			print(saved_file_path)
	#optim.swa_utils.update_bn(dataloaders['train'], swa_model)
	#swa_dict = {}
	#swa_dict['model_dict'] = swa_model.state_dict()
	#file_path = './models/swa_older_model_after_1_epoch.pt'
	#torch.save(swa_dict, file_path)

