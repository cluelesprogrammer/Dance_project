import datasets.dataset as dataset
from torchvision import transforms, utils
import models.models as models
import torch
from torch.utils.data import DataLoader
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
import torch.nn.functional as F
from torch.utils.data import random_split

def paf_layer_loss(pp, pt, crt):
	paf_ls = torch.stack([crt(pp[i], pt) for i in range(pp.shape[0])])
	return paf_ls

def conf_layer_loss(cp, ct, crt):
	conf_ls = torch.stack([crt(cp[i], ct) for i in range(cp.shape[0])])
	return conf_ls


normalize = transforms.Normalize(mean=[0.4103, 0.3404, 0.3246],std=[0.0877, 0.0720, 0.0492])

transform = {}
transform['image'] = transforms.Compose([transforms.ToTensor(), normalize])

batch_size = 16
N_workers = 6
device = torch.device('cuda:1')
#n_paf_layers, n_conf_layers = 6, 3
train_pd, val_pd, test_pd = pd.read_csv("data/train.csv"), pd.read_csv("data/val.csv"), pd.read_csv("data/test.csv")
load_size, return_size = 336, 336

train_data = dataset.DanceDataset(train_pd, load_size, return_size, transform=transform, map_index=2)
val_data = dataset.DanceDataset(val_pd, load_size, return_size, transform=transform, map_index=2)
test_data = dataset.DanceDataset(test_pd, load_size, return_size, transform=transform, map_index=2)
dataloaders = {}
dataloaders['train'] = DataLoader(train_data, batch_size=batch_size,num_workers=N_workers, shuffle=True)
dataloaders['val'] = DataLoader(val_data, batch_size=batch_size,num_workers=N_workers, shuffle=True)
dataloaders['test'] = DataLoader(test_data, batch_size=batch_size,num_workers=N_workers, shuffle=True)


criterion = nn.MSELoss()
trained_model, train_features = True, False
model = models.bodypose_model(trained_model, train_features).to(device)
lr_models, mm_models = 0.0001, 0.85
#lr_features, mm_features = 0.00001, 0.9
all_layers = list(model.children())
features_layer, all_other_layers = all_layers[0], all_layers[1:]

model_params = []
for m in all_layers:
	model_params += list(m.parameters())

optimizer_models = optim.SGD(model_params, lr=lr_models, momentum=mm_models)
#optimizer_features = optim.SGD(features_layer.parameters(), lr=lr_features, momentum=mm_features)
lowest_val_loss = math.inf

step_up,step_down = int(len(train_data) / (10 * batch_size * 3)) , int(len(train_data) / (10 * batch_size * 1.5))
scheduler_models = torch.optim.lr_scheduler.CyclicLR(optimizer_models, base_lr=lr_models/10, max_lr=lr_models, step_size_up=step_up, step_size_down=step_down, base_momentum=0.85, max_momentum=0.95)
best_model_weights = model.state_dict()
total_epochs = 10

for e in range(10):
	all_layers_paf_loss = []
	all_layers_conf_loss = []
	for phase in (['train', 'val']):
		start = time.time()
		model.load_state_dict(best_model_weights)
		running_loss = 0.0
		avg_val = 0.0
		time_elapsed = 0.0
		thirty_batch_sum = 0.0
		if phase == 'train':
			model.train()
			dataloader = dataloaders['train']
		else:
			model.eval()
			dataloader = dataloaders['val']
		for i, batch in enumerate(dataloader):
			since = time.time()
			masks = (batch['mask'].bool()).to(device)
			inps = batch['image'][masks].to(device)
			paf_truths = batch['pafs'][masks].to(device)
			conf_truths = batch['confs'][masks].to(device)
			optimizer_models.zero_grad()
			#optimizer_features.zero_grad()
			with torch.set_grad_enabled(phase == 'train'): #phase == 'train'):t_grad(d == 'train'):
				paf_pred, conf_pred = model(inps.float())
				i_paf_layers_loss = paf_layer_loss(paf_pred, paf_truths, criterion)
				i_conf_layers_loss = conf_layer_loss(conf_pred, conf_truths, criterion)
				loss = torch.sum(i_paf_layers_loss) + torch.sum(i_conf_layers_loss)
				all_layers_paf_loss.append(i_paf_layers_loss.clone().detach())
				all_layers_conf_loss.append(i_conf_layers_loss.clone().detach())
			running_loss += loss.item()
			thirty_batch_sum += loss.item()
			if (phase == 'train'):
				loss.backward()
				optimizer_models.step()
				scheduler_models.step()
				#optimizer_features.step()
			if ((i+1) % 30 == 0):
				#print(scheduler_models.get_lr())
				paf_loss_i_batch, conf_loss_i_batch = torch.sum(torch.stack(all_layers_paf_loss[-30:]), dim=0), torch.sum(torch.stack(all_layers_conf_loss[-30:]), dim=0)
				time_elapsed += time.time() - start
				start = time.time()
				fin_proc = i / len(dataloader)
				print("last 30 {} batch LOSS: ".format(phase), thirty_batch_sum / 30)
				print("average paf layers loss of {}-th batch: ".format(i+1), paf_loss_i_batch/ 30)
				print("conf_layers loss after {}-th batch: ".format(i+1), conf_loss_i_batch/30)
				print('learning rat: {}, momentum:{}'.format(optimizer_models.param_groups[0]['lr'], optimizer_models.param_groups[0]['momentum']))
				print("{} batches are done. the estimated remaining time is: ".format(i + 1), (1 - fin_proc) * time_elapsed / (fin_proc))
				print("\n")
				thirty_batch_sum = 0.0
				#print("per pixel_loss: ", loss.item() / (batch_size * (return_size/8) * (return_size/8)))
			time_elapsed += time.time() - since
		loss_per_pixel = running_loss / (len(dataloader) * batch_size * (return_size/8) * (return_size/8) * (12))
		avg_loss = running_loss / (len(dataloader) * batch_size)
		print("per avg loss of {} phase of epoch {} is: ".format(phase, e) , loss_per_pixel)
		if (phase == 'val' and avg_loss < lowest_val_loss):
			print('val per image loss: ', avg_loss)
			print('avg pixel loss: ', loss_per_pixel)
			print("epoch {} resulted in lowest validation loss".format(e))
			print("\n")
			lowest_val_loss = avg_loss
			best_model_weights = model.state_dict()
			lowest_pixel_loss = loss_per_pixel
		elif (phase == 'val' and avg_loss > lowest_val_loss):
			print("val loss did not decrease. Time to stop training")
		else:
			print("training phase is done")
	#scheduler_models = torch.optim.lr_scheduler.OneCycleLR(optimizer_models, lr_models, base_momentum=0.8, max_momentum=0.9, steps_per_epoch=len(dataloaders['train']), epochs=1)
	torch.save({'epoch': e, 'total_epochs':total_epochs, 'batch_size': batch_size, #'dataset_split':dataset_split, #'lr_features':lr_features, 'mm_features':mm_features,
			 'lr_models': lr_models, 'mm_models': mm_models,  'model_dict': model.state_dict(),'optimizer_models_state_dict': optimizer_models.state_dict(),
			'scheduler_models': 'one_cycle', 'scheduler_models_state_dict':scheduler_models.state_dict(), #optimizer_features_state_dict':optimizer_features.state_dict()'scheduler_features': 'steplr',
			'paf_layers_loss': torch.stack(all_layers_paf_loss), 'conf_layers_loss': torch.stack(all_layers_conf_loss),  'lowest_pixel_loss': lowest_pixel_loss, 'val_loss_per_pixel': loss_per_pixel, 'val_loss':avg_loss, 'lowest_val_loss':lowest_val_loss,'avg_pixel_loss': loss_per_pixel}, "models/final_models/older/older_model_without_features_training_epoch_{}.pt".format(e))

			#'scheduler_features_state_dict' : scheduler_features.state_dict(), 
	#scheduler_features.step()
	print("ok everything worked")
	#scheduler_models = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_models, len(dataloaders['train']))






