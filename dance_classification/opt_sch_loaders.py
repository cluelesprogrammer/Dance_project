import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

class MultipleOptimizers(object):
	def __init__(self,*op):
		self.optimizers = op
	def len(self):
		return len(self.optimizers)
	def zero_grad(self):
		for op in self.optimizers:
			if (op):
				op.zero_grad()
	def step(self):
		for op in self.optimizers:
			if (op):
				op.step()
	def get(self, i):
		if (i <= (len(self.optimizers) -1)):
			return self.optimizers[i]
		else:
			return IndexError
	def get_lr(self):
		all_lrs = []
		for opt in self.optimizers:
			lr = []
			if (opt):
				for param_group in opt.param_groups:
					lr.append(param_group['lr'])
				all_lrs.append(list(set(lr)))
		return all_lrs
	def get_state_dicts(self):
		state_dicts = []
		for opt in self.optimizers:
			if (opt):
				state_dicts.append(opt.state_dict())
			else:
				state_dicts.append(None)
		return state_dicts

class MultipleSchedulers(object):
	def __init__(self,*sch):
		self.schedulers = sch
	def step(self, i=None, step_value=None):
		if (i == None):
			for s in self.schedulers:
				if (s != None):
					s.step()
			return
		if(self.schedulers[i]):
			if (step_value):
				self.schedulers[i].step(step_value)
			else:
				self.schedulers[i].step()
		"""
		if (step_value==None):
			print('this is happening lcd')
			self.schedulers[i].step()
		else:
		for sch in self.schedulers:
			if (sch):
				sch.step()
		"""

	def get(self):
		if (i <= (len(self.schedulers) -1)):
			return self.schedulers[i]
		else:
			return IndexError
	def len(self):
		return len(self.schedulers)
	def get_state_dicts(self):
		state_dicts = []
		for sch in self.schedulers:
			if (sch):
				state_dicts.append(sch.state_dict())
			else:
				state_dicts.append(None)
		return state_dicts


def get_single_sgd_opt(parameters, lrs, mms): #, features_lr=None, features_mm=None):
	"""
	if (model.model_type == 'old'):
		all_layers = list(model.children())
	else
		all_layers = list(model.children())[0]
	features_model, rest_models = all_layers[0], all_layers[1:]
	model_params = []
	for m in rest_models:
		model_params += list(m.parameters()
	"""
	return optim.SGD([
		{'params': parameters[0]},
		{'params': parameters[1], 'lr': lrs[1], 'momentum': mms[1]},
		{'params': parameters[2], 'lr': lrs[2], 'momentum': mms[2]}
		], lr=lrs[0], momentum=mms[0])

"""
def get_adamw_opt(parameters, lrs, mms):
"""

def get_adam_opt(paramters, lr):
	return optim.Adam(paramters, lr=lr, betas=(0.9, 0.999))
	"""
        optimizer_models = optim.SGD(model_params, lr=model_lr, momentum=model_mm)
	optimizer_features = None
        if (train_features):
		if (features_lr == None and features_mm == None):
			print("please specify learning rate and momentum for feautres if they are set to train. Without them it will lead to an error")
                optimizer_features = optim.SGD(features_model.parameters(), lr=features_lr, momentum=features_mm)
	return optimizer_models, optimizer_features
	"""

def get_one_cycle_sch(opt, max_lr, steps_per_epoch, epochs, base_mm=0.85, max_mm=0.95):
	if (opt):
		return lr_scheduler.OneCycleLR(opt, max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, base_momentum=base_mm, max_momentum=max_mm)
	else:
		return None

def get_cyclic_sch(opt, base_lr, max_lr, cycles_per_epoch, loader_len, base_mm=0.85, max_mm=0.95):
	if (opt):
		step_size_up = int(loader_len / (cycles_per_epoch * 2))
		return lr_scheduler.CyclicLR(opt, base_lr, max_lr, step_size_up=step_size_up, base_momentum=base_mm, max_momentum=max_mm)
	else:
		return None

def get_warm_restarts_sch(opt, batches_per_restart, restart_scale=1):
	if (opt):
		return lr_scheduler.CosineAnnealingWarmRestarts(opt, batches_per_restart, restart_scale)
	else:
		return None

def get_step_sch(opt, step_size, gamma=0.5):
	if (opt):
		return lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
	else:
		return None

