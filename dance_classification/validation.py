import torch
import numpy as np
import time
import argparse
import wandb
import yaml
from tqdm import tqdm
from loss import paf_loss, conf_loss
from data_loader import lets_dance_loaders, coco_mpii_loaders, coco_loaders
from models_loader import get_model
from opt_sch_loaders import get_single_sgd_opt, get_adam_opt, get_one_cycle_sch, get_cyclic_sch, get_warm_restarts_sch, get_step_sch, MultipleOptimizers, MultipleSchedulers 
import torch.optim as optim
from train import train_epoch, val_epoch, train_n_epochs
import math
#from torch.multiprocessing import Pool, Process, set_start_method


model = get_model('old', True, True) # paf_stages=4, conf_stages=2)
#model = get_model('new', True,  paf_stages=5, conf_stages=2)
device = torch.device('cuda:2')
config = {'resized_size': 448, 'crop_size': 384, 'limb_width':2, 'sigma':2, 'batch_size': 16}
#train_val(config)
dataloaders = {}
dataloaders['coco'] = coco_mpii_loaders(config['resized_size'], config['crop_size'], config['batch_size'], config['limb_width'], config['sigma'], True)
dataloaders['letsdance'] = lets_dance_loaders(config['resized_size'], config['crop_size'], config['batch_size'], config['limb_width'], config['sigma'], True)
checkpoint_paths = ['models/pretrained_openpose/last_conv_layer_training_epoch_9.pt']
#checkpoint = torch.load('models/adamw/adamw_sgd_None_None_ftrs_lr_None_epoch_6.pt')
"""
for c in checkpoint_paths:
	checkpoint = torch.load(c, map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
"""

for k in dataloaders:
	run = wandb.init(reinit=True, name='last_layer_training_10'+k , config=config, project='FINAL VAL', entity='sbaral')
	val_loss, paf_loss, conf_loss = val_epoch(model, dataloaders[k]['val'], device, 0, k)
	run.finish()
