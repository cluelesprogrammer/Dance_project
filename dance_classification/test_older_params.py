import torch
import numpy as np
import time
import argparse
import wandb
import yaml
from tqdm import tqdm
from loss import paf_loss, conf_loss
from data_loader import lets_dance_loaders, coco_mpii_loaders
from models_loader import get_model
from opt_sch_loaders import get_sgd_opt, get_adam_opt, get_one_cycle_sch, get_cyclic_sch, get_warm_restarts_sch, get_step_sch
from datasets.dataset import GroundTruthGenerator as G
import os
from train import train_epoch, val_epoch
import math
import config

def train_val(config=None):
	with wandb.init(config=config, project='finetune_lr_search', entity='sbaral'):
		dataloader = lets_dance_loaders(



