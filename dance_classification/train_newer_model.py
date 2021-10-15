import torch
import numpy as np
import time
import argparse
import wandb
import yaml
from tqdm import tqdm
from loss import paf_loss, conf_loss
from data_loaders import lets_dance_dataloaders, coco_mpii_dataloaders, concatenated_dataloaders, coco_dataloaders
from models_loader import get_model
from opt_sch_loaders import get_adam_opt, get_one_cycle_sch, get_cyclic_sch, get_warm_restarts_sch, get_step_sch, get_single_sgd_opt
from datasets.dataset import GroundTruthGenerator as G
import torch.optim as optim
from train import train_epoch, val_epoch, train_n_epochs
import math
import os
from opt_sch_loaders import MultipleOptimizers, MultipleSchedulers
import torch.nn as nn
#wandb.init()

device = torch.device('cuda:0')
config = {}
config['features_lr'], config['features_mms'] = 5e-5, 0.9
#config['paf_momentum'], config['conf_momentum'] = 0.9, 0.9
config['load_size'], config['return_size'] = 448, 384
config['paf_model_lr'],config['conf_model_lr'] = 1e-4, 1e-4

config['total_epochs'], config['batch_size'], config['limb_width'], config['sigma'] = 15, 8, 2, 2
config['optimizers'] = 'sgd, adamw, adamw'
config['schedulers'] = 'step, none, none'
model = get_model('old', True, False).to(device)

"""
all_stages = model.paf_models + model.conf_models
for m in (all_stages):
	for p in m.parameters():
		p.requires_grad = True
"""
paf_params, conf_params = [], []

for m in model.paf_models:
	paf_params += list(m.parameters())
for m in model.conf_models:
	conf_params += list(m.parameters())
params = conf_params + paf_params
optimizer = optim.SGD(params, lr=0.1, momentum=0.9)
inp, pafs, confs = torch.randn(3,3,224,224).to(device), torch.randn(3, 38, 28, 28).to(device), torch.randn(3, 19, 28, 28).to(device)
wandb.init(project='test', name='check', dir='wandbdir')
loss_fn = nn.MSELoss()
wandb.watch(model, log_freq=1)
for i in range(200):
	optimizer.zero_grad()
	pafs_pred, confs_pred = model(inp)
	loss = loss_fn(pafs_pred, pafs) + loss_fn(confs_pred, confs)
	wandb.log({'loss':loss.item()})
	loss.backward()
	optimizer.step()


"""
dataloaders = coco_mpii_loaders(config['load_size'], config['return_size'], config['batch_size'], config['limb_width'], config['sigma'], True)
ftrs_opt, paf_opt, conf_opt = optim.SGD(model.features_extractor.parameters(), lr=config['features_lr'], momentum=config['features_mms']), optim.AdamW(paf_params, lr=config['paf_model_lr']), optim.AdamW(conf_params, lr=config['conf_model_lr'])
optimizers = MultipleOptimizers(ftrs_opt, paf_opt, conf_opt)
ftrs_sch = optim.lr_scheduler.StepLR(ftrs_opt, step_size=4, gamma=0.1)
schedulers = MultipleSchedulers(ftrs_sch)
#sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, int(len(dataloaders['train']) * 0.2), T_mult=2)
os.environ['WANDB_SILENT'] = 'true'
wandb.init(project='training_from_scratch_older_model', entity='sbaral', config=config, dir='wandbdir')
wandb.run.name = 'training_from_scratch_older_model'
wandb.watch(model)
file_path = 'older_model_train_from_scratch'
train_n_epochs(model, dataloaders, config['total_epochs'], device, optimizers, schedulers, file_path, checkpoint=False, epoch_start=0, test=True)
"""
