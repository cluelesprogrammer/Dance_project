import torch
import numpy as np
import time
import argparse
import wandb
import yaml
from tqdm import tqdm
from loss import paf_loss, conf_loss
from data_loader import lets_dance_loaders, coco_mpii_loaders, concatenated_dataloaders, coco_loaders
from models_loader import get_model
from opt_sch_loaders import get_cyclic_sch, get_warm_restarts_sch, MultipleOptimizer, MultipleScheduler
from datasets.dataset import GroundTruthGenerator as G
import torch.optim as optim
import math
from train import train_epoch, val_epoch

parser = argparse.ArgumentParser()
parser.add_argument('model_type', type=str, help='model to choose')
parser.add_argument('checkpoint_path', type=str, help='checkpoint to load')
parser.add_argument('dataset', type=str, help='dataset to validate on')
parser.add_argument('gpu_device', type=int, help='gpu device to train on')
parser.add_argument('--paf_stages', type=int, help='number of paf stages')
parser.add_argument('--conf_stages', type=int, help='number of conf stages')
args = parser.parse_args()
config = {'batch_size':64, 'load_size': 448, 'return_size': 384, 'limb_width':2, 'sigma':2}

if (args.model_type=='old'):
	model = get_model('old', False, True)
else:
	model = get_model('new', False, par_stages=args.paf_stages, conf_stages=args.conf_stages)

if (args.dataset == 'coco'):
	dataloader = coco_loaders(448,384,config['batch_size'], config['limb_width'], config['sigma'], True)
elif (args.dataset == 'letsdance'):
	dataloader = lets_dance_loaders(448, 384, config.batch_size, config['limb_width'], config['sigma'], True)
else:
	print('enter a valid dataset')

checkpoint = args.checkpoint_path

if (checkpoint != 'pretrained'):
	checkpoint_path = torch.load(checkpoint)
	checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device('cuda:{}'.format(args.gpu_device))
wandb.init(project='validation of different checkpoints', config=config,name='{}_{}'.format(checkpoint, args.dataset))
val_loss, paf_loss, conf_loss = val_epoch(model, dataloader['val'], device, 0, args.dataset, type='only validation')

"""
parser.add_argument('

#parser.add_argument("-pre", "--pretrained", type=int, help="0 for not trained and 1 for trained version", choices=[0,1])
parser.add_argument("-sch", "--scheduler", type=str, help="choose scheduler:onecycle,cyclic, warmres", choices=['onecycle', 'cyclic', 'warmres'])
parser.add_argument("-lr1", "--learning_rate1", type=int, help="choose learning rate max learning rate")
parser.add_argument("-lr2", "--learning_rate2", type=int, help="choose learning rate min learning rate")
parser.add_argument("-lr1", "--learning_rate3", type=int, help="choose learning rate  learning rate")
parser.add_argument("-tvgg", "--train_vgg", type=int, help="0 to not train vgg extractor, 1 to train it", choices=[0,1])
parse.add_argument("-f", "--frame_size", type=int, help="size of the frames for network", choices = [224, 336, 448, 672])
parser.add_argument("-paf", "--paf_stages", type=int, help="number of paf states if newer model", choices = list(range(1, 7)))
parser.add_argument("-conf", "--conf_stages", type=int, help="number of conf stages if newer model", choices = list(range(1, 6)))
parser.add_argument("-e", "--epoch", type=int, help="epoch to train from scratch")                                                                                                                                                           parser.add_argument("batch_size", type=int, help="batch size")                                                                                                                                                                               parser.add_argument("train_frac", type=float, help="percentage of the dataset to be trained on")                                                                                                                                             #parser.add_argument("num_workers", type=int, help="dataloader num workers")
parser.add_argument("gpu_number", type=int, help="which GPU to use")
args = parser.parse_args()                                                                                                                                                                                                                   return args                                                                                                                                                                                                                                               
"""
