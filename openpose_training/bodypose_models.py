class features_extractor(nn.Module):
	#vgg19 first ten layers and channel reduction layer
	def __init__(self, rg, pre_trained = True):
		super(features_extractor, self).__init__()
		self.all_features = models.vgg19(pretrained = pre_trained).features[0:23]
		self.all_features.add_module('conv4_3_CPM', nn.Conv2d(in_channels=512, out_channels=256,
								kernel_size=3, stride=1,
								padding=1))
		self.all_features.add_module('relu_4_3_CPM', nn.ReLU(inplace=True))
		self.all_features.add_module('conv4_4_CPM', nn.Conv2d(in_channels=256, out_channels=128,
								kernel_size=3, stride=1,
								padding=1))
		self.all_features.add_module('relu_4_4_CPM', nn.ReLU(inplace=True))
		self.seq_list = [nn.Sequential(ele) for ele in self.all_features]
		self.vgg_layer = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
						 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
						 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
						 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3_CPM', 'relu_4_3_CPM', 'conv4_4_CPM']
			#'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
						# 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
		for parameter in self.parameters():
				parameter.requires_grad = rg
	def forward(self, x):
		conv1_1 = self.seq_list[0](x)
		relu1_1 = self.seq_list[1](conv1_1)
		conv1_2 = self.seq_list[2](relu1_1)
		relu1_2 = self.seq_list[3](conv1_2)
		pool1 = self.seq_list[4](relu1_2)
		conv2_1 = self.seq_list[5](pool1)
		relu2_1 = self.seq_list[6](conv2_1)
		conv2_2 = self.seq_list[7](relu2_1)
		relu2_2 = self.seq_list[8](conv2_2)
		pool2 = self.seq_list[9](relu2_2)
		conv3_1 = self.seq_list[10](pool2)
		relu3_1 = self.seq_list[11](conv3_1)
		conv3_2 = self.seq_list[12](relu3_1)
		relu3_2 = self.seq_list[13](conv3_2)
		conv3_3 = self.seq_list[14](relu3_2)
		relu3_3 = self.seq_list[15](conv3_3)
		conv3_4 = self.seq_list[16](relu3_3)
		relu3_4 = self.seq_list[17](conv3_4)
		pool3 = self.seq_list[18](relu3_4)
		conv4_1 = self.seq_list[19](pool3)
		relu4_1 = self.seq_list[20](conv4_1)
		conv4_2 = self.seq_list[21](relu4_1)
		relu4_2 = self.seq_list[22](conv4_2)
		conv4_3_CPM = self.seq_list[23](relu4_2)
		relu4_3_CPM = self.seq_list[24](conv4_3_CPM)
		conv4_4_CPM = self.seq_list[25](relu4_3_CPM)
		return conv4_4_CPM



class simple_bodypose_model(nn.Module):
	def __init__(self):
		super(simple_bodypose_model).__init__()
		self.features_extractor = feautres_extractor


def make_layers(block, no_relu_layers):
	layers = []
	skip_conv_layers = ['MConv4_stage', 'MConv7_stage', 'MConv10_stage', 'MConv13_stage', 'MConv16_stage']
	for layer_name, v in block.items():
		if 'pool' in layer_name:
			layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
									padding=v[2])
			layers.append((layer_name, layer))
		else:
			conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
							   kernel_size=v[2], stride=v[3],
							   padding=v[4])
			layers.append((layer_name, conv2d))
			if layer_name not in no_relu_layers:
				layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))
	return nn.Sequential(OrderedDict(layers))

def get_stage_model(base_modules, middle_modules, final_modules, s, batch_norm, final_activation):
	mod = nn.Sequential()
	mod.add_module("base_module_stage", Conv_Block(base_modules, batch_norm))
	for m in range(1, middle_modules[1] + 1):
		mod.add_module("middle_module", Conv_Block(middle_modules[0], batch_norm))
	if (batch_norm):
		final_module1 = nn.Sequential(nn.Conv2d(final_modules[0][0], 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
	else:
		final_module1 = nn.Sequential(nn.Conv2d(final_modules[0][0], 128, 3, 1, 1), nn.ReLU(inplace=True))
	if (final_activation == 'sigmoid'):
		final_module2 = nn.Sequential(nn.Conv2d(final_modules[0][1], final_modules[1], 1, 1, 0), nn.Sigmoid())
	elif (final_activation == 'tanh'):
		final_module2 = nn.Sequential(nn.Conv2d(final_modules[0][1], final_modules[1], 1, 1, 0), nn.Tanh())
	elif (final_activation == 'relu'):
		final_module2 = nn.Sequential(nn.Conv2d(final_modules[0][1], final_modules[1], 1, 1, 0), nn.ReLU())
	else:
		final_module2 = nn.Sequential(nn.Conv2d(final_modules[0][1], final_modules[1], 1, 1, 0))

	mod.add_module("final_module1", final_module1)
	#mod.add_module("dropout_module_stage{}".format(s), final_module2)
	mod.add_module("final_module2", final_module2)
	return mod

def get_seq_block(in_channels, out_channels, kernel, stride, padding, batch_norm):
	if batch_norm:
		return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), \
				nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
	else:
		return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), \
				nn.ReLU(inplace=True))



class Conv_Block(nn.Module):
	def __init__(self, in_channels, batch_norm, out_channels=128):
		super(Conv_Block, self).__init__()
		self.C1 = get_seq_block(in_channels[0], out_channels, 3, 1, 1, batch_norm)
		self.C2 = get_seq_block(in_channels[1], out_channels, 3, 1, 1, batch_norm)
		self.C3 = get_seq_block(in_channels[2], out_channels, 3, 1, 1, batch_norm)
	def forward(self, x):
		c1_out = self.C1(x)
		c2_out = self.C2(c1_out)
		c3_out = self.C3(c2_out)
		out = torch.cat((c1_out, c2_out, c3_out), dim=1)
		return out

class new_bodypose_model(nn.Module):
	def __init__(self, paf_stages, conf_stages, train_features, batch_norm, final_activations):
		super(new_bodypose_model, self).__init__()
		self.train_features = train_features
		self.model_type = 'new'
		# these layers have no relu layer
		self.paf_stages, self.conf_stages, self.total_stages = paf_stages, conf_stages, paf_stages + conf_stages
		n_confs, n_pafs  = 19, 38
		#vgg19_model = vgg19(pretrained=False).features
		#self.features_extractor  = vgg19_model[:23] #first ten layers of vgg19 pretrained model)
		self.models = nn.Sequential()
		self.features_extractor = features_extractor(train_features)
		self.models.add_module('features_extractor', self.features_extractor)
		no_relu_layers = ['PConv5_stage1'] + ['MConv17_stage%i'%i for i in range(2, self.total_stages + 1)]
		blocks = {}

		# Stage 1
		block1 = OrderedDict([
						('PConv1_stage1', [128, 128, 3, 1, 1]), #P stands for Paf
						('PConv2_stage1', [128, 128, 3, 1, 1]),
						('PConv3_stage1', [128, 128, 3, 1, 1]),
						('PConv4_stage1', [128, 512, 1, 1, 0]),
						('PConv5_stage1', [512, 38, 1, 1, 0])
						])
		self.paf_models, self.conf_models = [], []
		paf1 = make_layers(block1, no_relu_layers)
		self.models.add_module('paf1', paf1)
		self.paf_models.append(paf1)
		#self.models.append(make_layers(block0, no_relu_layers))
		#self.models.append(make_layers(block1, no_relu_layers))

		# reamaining stages
		n_features_paf = 128 + n_pafs
		n_features_paf_conf = 128 + n_pafs + n_confs

		for stage in range(2, self.total_stages + 1):
			if (stage <= self.paf_stages):
				module_name = 'paf{}'.format(stage)
				#dropout = dropouts[0]
				paf_stage = get_stage_model([n_features_paf, 128, 128], ([384, 128, 128], 4), ([384, 128], n_pafs), stage, batch_norm, final_activations[0])
				self.models.add_module(module_name, paf_stage)
				self.paf_models.append(paf_stage)
			else:
				module_name = 'conf{}'.format(stage % self.paf_stages)
				#dropout = dropouts[1]
				if (stage == self.paf_stages + 1):
					conf_stage = get_stage_model([n_features_paf, 128, 128], ([384, 128, 128], 4), ([384, 128], n_confs), stage, batch_norm, final_activations[1])
				else:
					conf_stage = get_stage_model([n_features_paf_conf, 128, 128], ([384, 128, 128], 4), ([384, 128], n_confs), stage, batch_norm, final_activations[1])
				self.models.add_module(module_name, conf_stage)
				self.conf_models.append(conf_stage)

			#the arguments of function get_stage are the number of inputs to every CNN layer
			#self.models.append(get_stage_model(base_modules=[inp, 128, 128], middle_modules=([384, 128, 128], 4), final_modules=([384, 128], no_heat_maps), s=stage))
		print('finished initializing newer model')
	def get_phase(self):
		return self.training_stage
	def toggle_phase(self, x):
		if (self.training_stage == 'paf'):
			self.training_stage = 'conf_map'
		else:
			self.training_stage = 'paf'
	def forward(self,x):
		modules = list(self.models.modules())[0]
		if (self.train_features):
			features = modules[0](x)
		else:
			with torch.no_grad():
				features = modules[0](x)
		outputs, inp  = [], features
		paf_out = [modules[1](features)] # first stage paf output
		inp = torch.cat((features,paf_out[0]), dim = 1)
		for i in range(2, self.paf_stages + 1):
			refined_paf = modules[i](inp)
			paf_out.append(refined_paf)
			inp = torch.cat((features,refined_paf), dim = 1) #second stage onwards concatanation of features and first model output
		#paf_outputs = torch.stack(paf_outputs)
		paf_outputs = torch.stack(paf_out)

		if (self.conf_stages == 0):
			return paf_outputs
		else:
			conf_outputs = []
			for i in range(self.paf_stages + 1, self.total_stages + 1):
				refined_conf = modules[i](inp)
				#out = torch.cat([out, refined_conf], dim=1)
				conf_outputs.append(refined_conf) #first confidence map based on paf output
				inp = torch.cat((features, refined_paf, refined_conf), dim=1)
			conf_outputs = torch.stack(conf_outputs)
			#conf_outputs = torch.stack(conf_outputs)
			#out = torch.cat([paf_outputs, conf_outputs], dim=2)
			return paf_outputs, conf_outputs #(paf_outputs, conf_outputs)

class new_bodypose_model(nn.Module):
	def __init__(self, paf_stages, conf_stages, train_features, batch_norm, final_activations):
		super(new_bodypose_model, self).__init__()
		self.train_features = train_features
		self.model_type = 'new'
		# these layers have no relu layer
		self.paf_stages, self.conf_stages, self.total_stages = paf_stages, conf_stages, paf_stages + conf_stages
		n_confs, n_pafs  = 19, 38
		#vgg19_model = vgg19(pretrained=False).features
		#self.features_extractor  = vgg19_model[:23] #first ten layers of vgg19 pretrained model)
		self.models = nn.Sequential()
		self.features_extractor = features_extractor(train_features)
		self.models.add_module('features_extractor', self.features_extractor)
		no_relu_layers = ['PConv5_stage1'] + ['MConv17_stage%i'%i for i in range(2, self.total_stages + 1)]
		blocks = {}

		# Stage 1
		block1 = OrderedDict([
						('PConv1_stage1', [128, 128, 3, 1, 1]), #P stands for Paf
						('PConv2_stage1', [128, 128, 3, 1, 1]),
						('PConv3_stage1', [128, 128, 3, 1, 1]),
						('PConv4_stage1', [128, 512, 1, 1, 0]),
						('PConv5_stage1', [512, 38, 1, 1, 0])
						])
		self.paf_models, self.conf_models = [], []
		paf1 = make_layers(block1, no_relu_layers)
		self.models.add_module('paf1', paf1)
		self.paf_models.append(paf1)
		#self.models.append(make_layers(block0, no_relu_layers))
		#self.models.append(make_layers(block1, no_relu_layers))

		# reamaining stages
		n_features_paf = 128 + n_pafs
		n_features_paf_conf = 128 + n_pafs + n_confs

		for stage in range(2, self.total_stages + 1):
			if (stage <= self.paf_stages):
				module_name = 'paf{}'.format(stage)
				#dropout = dropouts[0]
				paf_stage = get_stage_model([n_features_paf, 128, 128], ([384, 128, 128], 4), ([384, 128], n_pafs), stage, batch_norm, final_activations[0])
				self.models.add_module(module_name, paf_stage)
				self.paf_models.append(paf_stage)
			else:
				module_name = 'conf{}'.format(stage % self.paf_stages)
				#dropout = dropouts[1]
				if (stage == self.paf_stages + 1):
					conf_stage = get_stage_model([n_features_paf, 128, 128], ([384, 128, 128], 4), ([384, 128], n_confs), stage, batch_norm, final_activations[1])
				else:
					conf_stage = get_stage_model([n_features_paf_conf, 128, 128], ([384, 128, 128], 4), ([384, 128], n_confs), stage, batch_norm, final_activations[1])
				self.models.add_module(module_name, conf_stage)
				self.conf_models.append(conf_stage)

			#the arguments of function get_stage are the number of inputs to every CNN layer
			#self.models.append(get_stage_model(base_modules=[inp, 128, 128], middle_modules=([384, 128, 128], 4), final_modules=([384, 128], no_heat_maps), s=stage))
		print('finished initializing newer model')
	def get_phase(self):
		return self.training_stage
	def toggle_phase(self, x):
		if (self.training_stage == 'paf'):
			self.training_stage = 'conf_map'
		else:
			self.training_stage = 'paf'
	def forward(self,x):
		modules = list(self.models.modules())[0]
		if (self.train_features):
			features = modules[0](x)
		else:
			with torch.no_grad():
				features = modules[0](x)
		outputs, inp  = [], features
		paf_out = [modules[1](features)] # first stage paf output
		inp = torch.cat((features,paf_out[0]), dim = 1)
		for i in range(2, self.paf_stages + 1):
			refined_paf = modules[i](inp)
			paf_out.append(refined_paf)
			inp = torch.cat((features,refined_paf), dim = 1) #second stage onwards concatanation of features and first model output
		#paf_outputs = torch.stack(paf_outputs)
		paf_outputs = torch.stack(paf_out)

		if (self.conf_stages == 0):
			return paf_outputs
		else:
			conf_outputs = []
			for i in range(self.paf_stages + 1, self.total_stages + 1):
				refined_conf = modules[i](inp)
				#out = torch.cat([out, refined_conf], dim=1)
				conf_outputs.append(refined_conf) #first confidence map based on paf output
				inp = torch.cat((features, refined_paf, refined_conf), dim=1)
			conf_outputs = torch.stack(conf_outputs)
			#conf_outputs = torch.stack(conf_outputs)
			#out = torch.cat([paf_outputs, conf_outputs], dim=2)
			return paf_outputs, conf_outputs #(paf_outputs, conf_outputs)


class bodypose_model(nn.Module):
	def __init__(self, trained_model=True, train_features=False, n_stages=6):
		super(bodypose_model, self).__init__()
		self.train_features = train_features
		self.paf_stages, self.conf_stages = 6, 6
		self.total_stages = self.paf_stages + self.conf_stages
		self.model_type = 'old'
		# these layers have no relu layer
		no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1',\
						  'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2',\
						  'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1',\
						  'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
		blocks = {}
		block0 = OrderedDict([
					  ('conv1_1', [3, 64, 3, 1, 1]),
					  ('conv1_2', [64, 64, 3, 1, 1]),
					  ('pool1_stage1', [2, 2, 0]),
					  ('conv2_1', [64, 128, 3, 1, 1]),
					  ('conv2_2', [128, 128, 3, 1, 1]),
					  ('pool2_stage1', [2, 2, 0]),
					  ('conv3_1', [128, 256, 3, 1, 1]),
					  ('conv3_2', [256, 256, 3, 1, 1]),
					  ('conv3_3', [256, 256, 3, 1, 1]),
					  ('conv3_4', [256, 256, 3, 1, 1]),
					  ('pool3_stage1', [2, 2, 0]),
					  ('conv4_1', [256, 512, 3, 1, 1]),
					  ('conv4_2', [512, 512, 3, 1, 1]),
					  ('conv4_3_CPM', [512, 256, 3, 1, 1]),
					  ('conv4_4_CPM', [256, 128, 3, 1, 1])
				  ])


		# Stage 1
		block1_1 = OrderedDict([
						('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
						('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
						('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
						('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
						('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
					])

		block1_2 = OrderedDict([
						('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
						('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
						('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
						('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
						('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
					])
		blocks['block1_1'] = block1_1
		blocks['block1_2'] = block1_2

		self.model0 = make_layers(block0, no_relu_layers)

		# Stages 2 - 6
		for i in range(2, 7):
			blocks['block%d_1' % i] = OrderedDict([
					('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
					('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
					('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
					('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
					('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
					('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
					('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
				])

			blocks['block%d_2' % i] = OrderedDict([
					('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
					('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
					('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
					('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
					('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
					('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
					('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
				])

		for k in blocks.keys():
			blocks[k] = make_layers(blocks[k], no_relu_layers)

		self.model1_1 = blocks['block1_1']
		self.model2_1 = blocks['block2_1']
		self.model3_1 = blocks['block3_1']
		self.model4_1 = blocks['block4_1']
		self.model5_1 = blocks['block5_1']
		self.model6_1 = blocks['block6_1']

		self.model1_2 = blocks['block1_2']
		self.model2_2 = blocks['block2_2']
		self.model3_2 = blocks['block3_2']
		self.model4_2 = blocks['block4_2']
		self.model5_2 = blocks['block5_2']
		self.model6_2 = blocks['block6_2']
		if (trained_model):
			weights_path = os.path.join(os.getcwd(), "models/body_pose_model.pth")
			self.load_state_dict(self.transfer(weights_path))
			print("trained_model")
		self.features_extractor = [self.model0]
		self.paf_models = [self.model1_1, self.model2_1, self.model3_1, self.model4_1, self.model5_1, self.model6_1]
		self.conf_models = [self.model1_2, self.model2_2, self.model3_2, self.model4_2, self.model5_2, self.model6_2]
	def max_pafs(self,x):
		pafs, _ = self.forward(x)
		even_index, odd_index = list(range(0, 38, 2)), list(range(1, 38, 2))
		pafs_xy = [pafs[:, even_index, :,:], pafs[:,odd_index,:,:]]
		maxed_xy = []
		for p in pafs_xy:
			max_abs, max_p = torch.max(torch.abs(p), dim=1)[0], torch.max(p, dim=1)[0]
			negative_max = max_abs != max_p
			max_p[negative_max] = max_abs[negative_max] * -1
			maxed_xy.append(max_p)
		maxed_paf = torch.cat([maxed_xy[0].unsqueeze(1), maxed_xy[1].unsqueeze(1)],1)
		return maxed_paf

	def produce_maxed_heatmaps(self, x):
		even_index = list(range(0, 38, 2))
		odd_index = list(range(1, 38, 2))
		pafs, confs = self.forward(x)
		pafs, confs = pafs[-1], confs[-1] #considering only last stage
		maxed_confs = torch.max(confs[:, 0:18, :, :], dim=1)[0]
		mag_pafs = torch.sqrt(pafs[:,even_index,:,:]*pafs[:,even_index,:,:] + pafs[:,odd_index,:,:]*pafs[:,odd_index,:,:])
		maxed_pafs = torch.max(pafs,dim=1)[0]
		single_pafs_confs = torch.stack([maxed_pafs, maxed_confs], dim=1)
		return single_pafs_confs
	def forward(self, x):
		out1 = self.model0(x)
		out1_1 = self.model1_1(out1)
		out1_2 = self.model1_2(out1)
		out2 = torch.cat([out1_1, out1_2, out1], 1)

		out2_1 = self.model2_1(out2)
		out2_2 = self.model2_2(out2)
		out3 = torch.cat([out2_1, out2_2, out1], 1)

		out3_1 = self.model3_1(out3)
		out3_2 = self.model3_2(out3)
		out4 = torch.cat([out3_1, out3_2, out1], 1)

		out4_1 = self.model4_1(out4)
		out4_2 = self.model4_2(out4)
		out5 = torch.cat([out4_1, out4_2, out1], 1)

		out5_1 = self.model5_1(out5)
		out5_2 = self.model5_2(out5)
		out6 = torch.cat([out5_1, out5_2, out1], 1)

		out6_1 = self.model6_1(out6)
		out6_2 = self.model6_2(out6)
		return torch.stack([out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]), torch.stack([out1_2, out2_2, out3_2, out4_2, out5_2, out6_2])
	def transfer(self, model_path):
		transfered_model_weights = {}
		model_weights = torch.load(model_path)
		for weights_name in self.state_dict().keys():
			transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
		return transfered_model_weights


