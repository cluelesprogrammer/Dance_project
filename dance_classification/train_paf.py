import datasets.dataset as dataset
from torchvision import transforms, utils
import models.models as models
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from PIL import Image
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imageio
import subprocess
import time
import pandas as pd
import torch.nn as nn
import datasets.groundtruths as G
import torch.optim as optim
import copy
from pycocotools.coco import COCO
import pylab


def paf_collage(batch):
	batch_tensor = torch.zeros(batch_tensor_size)
	for b in range(batch_tensor.shape[0]):
		batch_tensor[b] = batch[b]['image']
	GT = G.GroundTruthGenerator(batch[0]['image'].shape[1:3])
	sep_joints = [item['joints_list'] for item in batch]
	maps = torch.stack([GT.generate_pafs(s) for s in sep_joints])
	paf_masks = torch.stack([GT.get_paf_masks(s) for s in sep_joints]) #x values and y values of paf values first and then confidence maps: batch_size * 3 * maps
	return [batch_tensor, maps, masks]

def nms(t, nms_param = 3):
	m = nn.MaxPool2d(kernel_size=nms_param, padding=int(nms_param/2), stride=1)
	pooled_t = m(t)
	mask = torch.eq(t, pooled_t) * 1
	return pooled_t * mask

def upsample(t, factor = 8):
	m = nn.Upsample(scale_factor=8, mode='nearest')
	up_t = torch.stack([m(small_t) for small_t in t])
	return up_t

def get_loss(truths, pred):
	#pred = upsample(pred)
	diff = (pred - truths) #extended_mask
	return torch.sum(torch.sqrt(diff * diff))
"""
def train_model(model, criterion, optimizer, scheduler, stage, num_epochs=1):
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	batch_time_elapsed = 0.0
	for epoch in range(num_epochs):
		start = time.time()
		print('Epoch'.format(epoch, num_epochs - 1))
		for phase in ['train', 'val']:
			if (phase == 'train'):
				model.train()
			else:
				model.eval()
			running_loss = 0.0

		for i, inputs in enumerate(dataloaders[phase]):
			images = inputs[0].to(device)
			maps = inputs[1].to(device)
			masks = inputs[2].to(device)
			extended_mask = torch.empty(maps.shape).to(device)
			for index in np.ndindex(masks.shape):
				conf_map = torch.empty(maps.shape[-2:]).fill_(masks[index]).to(device)
				extended_mask[index] = conf_map

			#zero the parameter gradients
			optimizer.zero_grad()
			#track history if only in train
			with torch.set_grad_enabled(phase == 'train'):
				outputs = model(images)
				pred = upsample(outputs)
				loss = 100 * sum(criterion(pred * extended_mask, maps)
				if (phase == 'train'):
					loss.backward()
					optimizer.step()
			if (i % 3 == 0):
				print(loss.item())
			running_loss += loss.item() * images.size(0)
			batch_time_elapsed += time.time() - start
			print(batch_time_elapsed)
			if (batch_time_elapsed > 30.0):
				timestr = time.strftime("%Y%m%d-%H%M%S")
				path = timestr + ".pth"
				torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),'best_model_weights': best_model_wts, 'optimizer_state_dict':optimizer.state_dict(),
					'loss': loss, 'scheduler_state_dict': scheduler.state_dict()}, path)
				batch_time_elapsed = 0.0
		if (phase == 'train'):
			scheduler.step()
		epoch_loss = running_loss / len(dataloaders[phase]) * batch_size
		print('{} Loss: {:.4f}'.format(phase, epoch_loss))
		if phase == 'val' and epoch_acc > best_acc:
			best_acc = epoch_acc
			best_model_wts = copy.deepcopy(model.state_dict())
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model
"""
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.ToTensor(), normalize])
device = "cuda" if torch.cuda.is_available() else "cpu"


"""

paf_model = models.new_bodypose_model(8, 0)
for param in paf_model.models[0].parameters():
	param.requires_grad = False
batch = torch.rand(8,3,224,224)
dataDir = 'data/coco'
dataType = 'val2017'
annFile='{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps = COCO(annFile)
print(coco_kps)
#model = train_model(paf_model, criterion, optimizer, exp_lr_scheduler, 6, num_epochs=1)




#device = 'cpu'
frames_csv = pd.read_csv("data/frames/data.csv")
bernoulli_idx = np.random.binomial(size=len(frames_csv), n=1, p=0.75)
train_idx, test_val_idx = np.where(bernoulli_idx==1)[0], np.where(bernoulli_idx==0)[0]

#test_val_idx = np.random.shuffle(test_val_idx)
val_idx, test_idx = test_val_idx[0:int(len(test_val_idx) * 0.25)], test_val_idx[0:int(len(test_val_idx) * 0.75)]
train_csv, val_csv, test_csv = frames_csv.iloc[train_idx, :], frames_csv.iloc[val_idx, :], frames_csv.iloc[test_idx, :]
#all_csv = pd.concat(train_csv, val_csv, test_csv)
train_data, val_data, test_data = dataset.DanceDataset(train_csv, transform=transform), dataset.DanceDataset(val_csv, transform=transform), dataset.DanceDataset(test_csv, transform=transform)

batch_size = 16
dataloaders = {}
dataloaders['train_paf'], dataloaders['val'], dataloaders['test'] = DataLoader(train_data, batch_size, shuffle=True, collate_fn=paf_collate, num_workers=2), DataLoader(val_data, batch_size, shuffle=True, collate_fn=paf_collate, num_workers=2), DataLoader(test_data, batch_size, shuffle=True, collate_fn=paf_collate, num_workers=2)
train_loader = dataloaders['train_paf']
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = models.new_bodypose_model()
for param in model.vgg19_10.parameters():
	param.requires_grad = False

for i, batch in enumerate(train_loader):
	since = time.time()
	print(batch[1].shape)
	print(time.time() - since)


"""

