import datasets.dataset as dataset
import models.models as models
import torch
from loss import paf_loss, conf_loss
from data_loader import lets_dance_loaders, coco_mpii_loaders
from tqdm import tqdm
import wandb
import numpy as np
import yaml

def val(config=None):
	with wandb.init(config=config):
		device = torch.device('cuda:3')
		model = models.bodypose_model().eval().to(device)
		config = wandb.config
		val_loader = lets_dance_loaders(config.img_size, 24, config.limb_width, config.sigma, 1, config.rotation)['val']
		running_loss = 0.0
		layerwise_paf_loss, layerwise_conf_loss = 0.0, 0.0
		i = 0
		t = 'lets_dance_val'
		for batch in tqdm(val_loader):
			images, paf_truth, conf_truth = batch['image'].to(device), batch['pafs'].to(device), batch['confs'].to(device)
			with torch.no_grad():
				paf_pred, conf_pred = model(images)
			batch_paf_loss, batch_conf_loss = paf_loss(paf_pred, paf_truth), conf_loss(conf_pred, conf_truth)
			loss = torch.sum(batch_paf_loss) + torch.sum(batch_conf_loss)
			layerwise_paf_loss += batch_paf_loss
			layerwise_conf_loss += batch_conf_loss
			running_loss += loss.detach().clone().item()
			if (i % 5 == 0):
				wandb.log({'{}_running_loss'.format(t): running_loss})
			i += 1
		dict_to_log = {}
		for i in range(len(layerwise_paf_loss)):
			i_paf = layerwise_paf_loss[i]
			i_conf = layerwise_conf_loss[i]
			wandb.log({'{}_paf'.format(t): i_paf/len(val_loader)})
			wandb.log({'{}_conf'.format(t): i_conf/len(val_loader)})
			wandb.log({'i_stage_paf_and_conf_loss': (i_paf + i_conf)/len(val_loader)})
		dict_to_log['{}_total_val_loss'.format(t)] = running_loss
		dict_to_log['per_sample_val_loss'] = running_loss / (len(val_loader))
		wandb.log(dict_to_log)

config = yaml.full_load(open('val_config.yaml'))
sweep_id = wandb.sweep(config, project="val_loss_lets_dance")
wandb.agent(sweep_id, val)
"""
type = ['coco_val', 'lets_dance_val']
config = {'batch_size':24, 'limb_width': 2, 'sigma': 2, 'rotation':0, 'img_size': 224}
#config = {'batch_size':12, 'limb_width': 2, 'sigma': 2, 'rotation':0, 'image_size': 336}
#config = {'batch_size':12, 'limb_width': 2, 'sigma': 2, 'rotation':0, 'image_size': 386}
#config = {'batch_size':12, 'limb_width': 2, 'sigma': 2, 'rotation':0, 'image_size': 448}

wandb.init(config=config)

for i, val_loader in enumerate(val_loaders):
	val(model, val_loader, config['img_size'], device, type[i])


"""
