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
from opt_sch_loaders import get_sgd_opt, get_adam_opt, get_one_cycle_sch, get_cyclic_sch, get_warm_restarts_sch, get_step_sch
from datasets.dataset import GroundTruthGenerator as G
import torch.optim as optim

def train_epoch(model, opt, sch, dataloader, device):
	model.train()
	n_batches_loss, n_batches_paf_loss, n_batches_conf_loss, log_freq = 0.0, 0.0, 0.0, 100
	i = 0
	for batch in tqdm(dataloader):
		images, paf_truths, conf_truths = batch['image'].float().to(device), batch['pafs'].float().to(device), batch['confs'].float().to(device)
		paf_preds, conf_preds = model(images)
		opt.zero_grad()
		paf_l, conf_l = paf_loss(paf_preds, paf_truths), conf_loss(conf_preds, conf_truths)
		loss = torch.sum(paf_l) + torch.sum(conf_l)
		n_batches_loss += loss.clone().detach().item()
		n_batches_paf_loss += torch.sum(paf_l.clone().detach()).item()
		n_batches_conf_loss += torch.sum(conf_l.clone().detach()).item()
		if ((i+1)% log_freq == 0):
			wandb.log({'adam_no_sch_lr': opt.param_groups[0]['lr']})
			wandb.log({'rolling_avg_loss_last_{}_batches '.format(log_freq) : loss.item()})
			wandb.log({'rolling_avg_paf_loss_last_{}_batches'.format(log_freq): torch.sum(paf_l).item()})
			wandb.log({'rolling_avg_conf_loss_last_{}_batches'.format(log_freq): torch.sum(conf_l).item()})
			n_batches_loss, n_batches_paf_loss, n_batches_conf_loss = 0.0, 0.0, 0.0
		loss.backward()
		opt.step()
		#sch.step()
		i += 1

def val_epoch(mod, dataloader, device):
	mod.eval()
	running_loss = 0.0
	paf_losses, conf_losses = [], []
	running_paf_loss = 0.0
	running_conf_loss = 0.0
	for batch in tqdm(dataloader):
		images, paf_truths, conf_truths = batch['image'].to(device), batch['pafs'].to(device), batch['confs'].to(device)
		with torch.no_grad():
			paf_preds, conf_preds = model(images)
		paf_l, conf_l = paf_loss(paf_preds, paf_truths), conf_loss(conf_preds, conf_truths)
		paf_losses.append(paf_l)
		conf_losses.append(conf_l)
		loss = torch.sum(paf_l) + torch.sum(conf_l)
		running_loss += loss.item()
		running_paf_loss += torch.sum(paf_l).item()
		running_conf_loss += torch.sum(conf_l).item()
	layerwise_mean_paf_loss, layerwise_mean_conf_loss = torch.mean(torch.stack(paf_losses), dim=0), torch.mean(torch.stack(conf_losses), dim=0)
	for l in layerwise_mean_paf_loss:
		wandb.log({'paf_layerwise_per_batch_loss': l})
	for l in layerwise_mean_conf_loss:
		wandb.log({'conf_layerwise_per_batch_loss': l})
	wandb.log({'running_loss': running_loss})
	wandb.log({'running_loss': running_paf_loss})
	wandb.log({'running_loss': running_conf_loss})

#wandb.init()
device = torch.device('cuda:0')
model = get_model('new', True, 1).float().to(device)
optimizers = ['adam', 'sgd']
dataloaders = coco_mpii_loaders(386, 24, 3, 2, 1, 1)
for o in optimizers:
	wandb.init(project='adam_sgd_compare_coco_mpii', config={'batch_size':24, 'optimizer':o}, entity='sbaral')
	if (o == 'adam'):
		optimizer = optim.AdamW(model.paramters())
		scheduler = optim.CyclicLR(optimizer, 
	else:
		optimizer = 
"""

opt = torch.optim.AdamW(model.parameters())
#print(sum([(len(dataloaders[d]) * 24) for d in dataloaders])
#train_dataloader = all_train_dataloaders(386, 24, 3, 2, 1, 1)

#val_dataloader = all_val_dataloaders(386, 24, 3, 2, 1, 1)
#len_train_loaders = len(dataloaders['coco_train']) + len(dataloaders['mpii_train'])
#sch = optim.lr_scheduler.CyclicLR(opt, 0.00001, 0.0001, step_size_up=int(len(dataloaders['train'])*0.5), cycle_momentum=False)
model.to(device)
wandb.init(project='Comparing optimizers', entity='sbaral')
#wandb.watch(model)
for e in range(3):
	train_epoch(model, opt, None, dataloaders['train'], device)
	val_epoch(model, dataloaders['val'], device)
	torch.save(model, 'adam_no_sch_new_{}_{}_epoch_{}.pt'.format(model.paf_stages, model.conf_stages, e))
"""
