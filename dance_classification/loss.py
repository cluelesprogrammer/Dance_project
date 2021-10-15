import torch
import torch.nn as nn

def paf_loss(pp, pt, paf_mask, grad_stages='all'):
	true_indices = [i for i in range(len(paf_mask)) if (paf_mask[i] == True)]
	pp, pt = pp[:, :, true_indices, :, :], pt[:, true_indices, :, :]
	#print(type(pt))
	if (grad_stages=='all'):
		paf_ls = torch.stack([torch.sum((pp[i] -  pt) ** 2) for i in range(pp.shape[0])])
		return paf_ls/pt.numel()
	all_paf_stages = range(pp.shape[0])
	no_grad_stages = set(all_paf_stages) - set(grad_stages)
	all_loss = []
	for i in no_grad_stages:
		with torch.no_grad():
			all_loss.append(torch.stack([torch.sum(torch.sqrt((pp[i] -  pt) ** 2)) for i in range(pp.shape[0])]))
	for i in grad_stages:
		all_loss.append(torch.sum((pp[i] - pt) ** 2))
	return torch.stack(all_loss)/pt.numel()

def mse_paf_loss(pp, pt, paf_mask=None):
	criterion = nn.MSELoss()
	if (paf_mask != None):
		true_indices = [i for i in range(len(paf_mask)) if (paf_mask[i] == True)]
		pp, pt = pp[:, :, true_indices, :, :], pt[:, true_indices, :, :]
	stage_loss = torch.stack([criterion(pp[i], pt) for i in range(pp.shape[0])])
	return stage_loss

def mse_conf_loss(cp, ct, conf_mask=None):
	criterion = nn.MSELoss()
	if (conf_mask != None):
		true_indices = [i for i in range(len(conf_mask)) if (conf_mask[i] == True)]
		cp, ct = cp[:, :, true_indices, :, :], ct[:, true_indices, :, :]
	stage_loss = torch.stack([criterion(cp[i], ct) for i in range(cp.shape[0])])
	return stage_loss

def conf_loss(cp, ct, conf_mask,grad_stages='all'):
	true_indices = [i for i in range(len(conf_mask)) if (conf_mask[i] == True)]
	cp, ct = cp[:, :, true_indices, :, :], ct[:, true_indices, :, :]
	if (grad_stages =='all'):
		conf_ls = torch.stack([torch.sum((cp[i] -  ct) ** 2) for i in range(cp.shape[0])])
		return conf_ls/ct.numel()
	all_conf_stages = range(cp.shape[0])
	no_grad_stages = set(all_conf_stages) - set(grad_stages)
	all_loss = []
	for i in no_grad_stages:
		with torch.no_grad():
			layer_loss = torch.sum((cp[i] - ct) ** 2)
		all_loss.append(layer_loss)
	for i in grad_stages:
		all_loss.append(torch.sum((cp[i] - ct) ** 2))
	return torch.stack(all_loss)/ct.numel()

