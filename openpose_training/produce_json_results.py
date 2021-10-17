from pycocotools.coco import COCO
import pandas as pd
import os
import time
from scipy.ndimage.filters import gaussian_filter
import bodypose_models
from pycocotools.cocoeval import COCOeval
import skimage.io as io
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import argparse
from matplotlib.path import Path
from skimage.transform import rescale

def nms(confidence_map):
	thres = 0.1
	confidence_map[confidence_map < 0.1] = 0
	max_pool = nn.MaxPool2d(kernel_size=5, stride=1,padding=2)
	pooled_confidence_map = max_pool(confidence_map)
	pooled_confidence_map[torch.where(pooled_confidence_map != confidence_map)] = 0
	#pooled_confidence_map = upsample(pooled_confidence_map)
	max_conf_values, max_conf_indices = torch.max(pooled_confidence_map, dim=1)
	supressed_conf_map = torch.zeros(pooled_confidence_map.shape).to(device)
	for ind in np.ndindex(pooled_confidence_map.shape[2:]):
		supressed_conf_map[0,max_conf_indices[0, ind[0], ind[1]], ind[0], ind[1]] = max_conf_values[0, ind[0], ind[1]]
	return supressed_conf_map


def parse():
	parser = argparse.ArgumentParser(description='model and checkpoint to test')
	parser.add_argument('model_type', type=str, help='model type')
	parser.add_argument('-g', '--gpu_device', type=int, help='enter the location to load tensors and models on')
	parser.add_argument('-checkpoint', '--checkpoint location', type=str, help='checkpoint to load')
	parser.add_argument('-paf and conf stages', '--paf_conf_stages', type=str, help='number of paf and conf stages in that order')
	arguments = parser.parse_args()
	return arguments

def find_parts(heatmaps, pafs):
	candidates, peaks_counter = [], 0
	for (i, map) in enumerate(heatmaps):
		candidates_y, candidates_x = np.where(map > 0)
		if (len(candidates_y) == 0):
			candidates.append([])
			continue
		score = map[candidates_y, candidates_x]
		peaks_id = list(range(peaks_counter, peaks_counter + len(candidates_y)))
		peaks_counter += len(candidates_y)
		candidates.append(np.array(list(zip([candidates_y, candidates_x, score, peaks_id]))))
	joint_limb_correspondence = np.array([[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,2],[2,16],[3,4],[16,2],[1,5],[5,6], [6,7],[5,17],[0,1],[0,14], [0,15],[14,16], [15,17]])
	limb_width = 3
	for (i, joints_id) in enumerate (joint_limb_correspondence):
		if (candidates[joints_id[0]] == [] or candidates[joints_id[1]] == []):
			continue
		else:
			candidates_i, candidates_j = candidates[joints_id[0]], candidates[joints_id[1]]
			i_ys, i_xs, i_scores, i_peaks_ids = candidates_i[0] * 8, candidates_i[1] * 8, candidates_i[2], candidates_i[3] #part affinity fields are eight times as big
			j_ys, j_xs, j_scores, j_peaks_ids = candidates_j[0] * 8, candidates_j[1] * 8, candidates_j[2], candidates_j[3]
			for (i_yx, i_peak_id) in np.array(list(zip(i_ys, i_xs, i_peak_ids))):
				for j_yx in np.array(list(zip(j_ys, j_xs))):
					diff, mag = (i_yx - j_yx).astype(float), np.linalg.norm(i_yx - j_yx)
					unit_vec = np.divide(diff, mag, out=np.zeros_like(diff), where=mag!=0)
					unit_vec_p = np.array([-unit_vec[1], unit_vec[0]])
					vec_offset = unit_vec_p * (limb_width/2)
					b_box = np.array([(i_yx - vec_offset), (i_yx + vec_offset), (j_yx + vec_offset), (j_yx - vec_offset)]).reshape(-1, 2)
					path = Path(b_box)
					sa = i_yx[0] == j_yx[0] or i_yx[1] == j_yx[1]
					if (i_yx[0] == j_yx[0]):
						Y, X = np.mgrid[i_yx[0] - limb_width/2: i_yx[0] + limb_width/2, min(i_yx[1], j_yx[1]): max(i_yx[1], j_yx[1])]
					elif (i_yx[1] == j_yx[1]):
						Y, X = np.mgrid[min(i_yx[0], j_yx[0]): max(i_yx[0], j_yx[0]), i_yx[1] - limb_width/2: i_yx[1] + limb_width/2]
					else:
						Y, X = np.mgrid[min(i_yx[0], j_yx[0]): max(i_yx[0], j_yx[0]), min(i_yx[1], j_yx[1]): max(i_yx[1], j_yx[1])]
					possible_points = np.vstack((Y.ravel(), X.ravel())).T
					paf_points = possible_points[path.contains_points(possible_points)]
					paf_points_Y, paf_points_X = paf_points[:,0], paf_points[:,1]
					try:
						paf_vector = np.array([sum(pafs[2 * i][paf_points_Y, paf_points_X]), sum(pafs[2 * i + 1][paf_points_Y, paf_points_X])])
					except:
						paf_points_Y, paf_points_X = paf_points_Y.astype(int), paf_points_X.astype(int)
						paf_vector = np.array([sum(pafs[2 * i][paf_points_Y, paf_points_X]), sum(pafs[2 * i + 1][paf_points_Y, paf_points_X])])
					print(paf_vector/mag)
					"""
					if (sa):
						print('same level joints paf points')
					else:
						print('diff level joints paf points')
					try:
					except:
						print(i_yx, j_yx)
					"""





	"""
		limb2_co = np.vstack([joints[self.limb_order[l][1]] for l in range(len(self.limb_order))]).reshape(-1,2)
		#finding magnitude of each limb vector for unit vector calculation
		#finding perpendicular unit vector of a limb to get a box for the limb
		mag = self.get_magnitude(unit_vecs_p)
		unit_vecs_p = self.no_error_div(unit_vecs_p, mag)
		#finding how much the co-ordinates of joints must be offset to get the bounding box co-ordinates for each limb
		#shape of array containing all limbs separated by joint type
		sep_shape = (len(self.limb_order), joints.shape[1])
		#getting bounding boxes
		#the joints are set to (0,0) if they are not present to acheive uniformity in data, so masking is used to not make bounding box with them
		limbs_inds = np.where(limbs_mask == 1)
		unit_vecs = unit_vecs.reshape((sep_shape[0], sep_shape[1], 2))
		limbs_mask_ext = np.tile(limbs_mask, (1, 2)).reshape(unit_vecs.shape)
		unit_vecs *= limbs_mask_ext

		#declaring outside of loop to same function being carried out everytime
		paf_truths = []
		paf_vec_x, paf_vec_y = np.zeros(self.img_size), np.zeros(self.img_size)
		since = time.time()
		for i in range(sep_shape[0]):
			boxes = bounding_boxes_co[i]
			for i, m in enumerate(masks):
				mask = m.reshape(self.img_size)
				#print(mask)
				l = len(paf_vec_x[mask])
				paf_vec_x[mask] += np.repeat(unit_vec[i][0], l)
				paf_vec_y[mask] += np.repeat(unit_vec[i][1], l)
			mag = np.sqrt(paf_vec_x * paf_vec_x + paf_vec_y * paf_vec_y)
			paf_vec_x = self.no_error_div(paf_vec_x, mag)
			paf_vec_y = self.no_error_div(paf_vec_y, mag)
			paf_truths.append(paf_vec_x)
			paf_truths.append(paf_vec_y)
			paf_vec_x, paf_vec_y = np.zeros(self.img_size), np.zeros(self.img_size)
			candidates_i_y, candidates_i_x, score_i = candidates[joints_id[0]]
	"""
args = parse()
if (args.model_type == 'bodypose_old'):
	model = bodypose_models.bodypose_model().float()
else:
	print('not a valid model')

if (args.gpu_device):
	device = torch.device('cuda:{}'.format(args.gpu_device))
else:
	device = torch.device('cpu')

model = model.to(device)
#train_annot_path = 'data/coco/annotations/person_keypoints_train2017.json'
parent_dir = os.path.dirname(os.getcwd())
coco = COCO(os.path.join(parent_dir, 'data/coco/annotations/person_keypoints_val2017.json'))

for i in tqdm(coco.getImgIds()):
	img_metadata = coco.loadImgs(i)[0]
	file_loc = os.path.join(parent_dir, "data/coco/images/val2017", img_metadata["file_name"])
	img_array = io.imread(file_loc)
	sorted_dims = sorted(set(img_array.shape))
	if (sorted_dims[0] == 3):
		smallest_dim = sorted_dims[1]
		rescale_ratio = 224 / smallest_dim
		new_shape = (int(img_array.shape[0] * rescale_ratio), int(img_array.shape[1] * rescale_ratio), 3)
	else:
		smallest_dim = sorted_dims[0]
		rescale_ratio = 224 / smallest_dim
		new_shape = (int(img_array.shape[0] * rescale_ratio), int(img_array.shape[1] * rescale_ratio))
	img_array = resize(image=img_array, output_shape=new_shape)
	if (len(img_array.shape) < 3):
		img_array = np.stack([img_array, img_array, img_array], axis=2)
	mean, std = np.mean(img_array), np.std(img_array)
	normalized_img = (img_array - mean) / std
	normalized_img = img_array
	tensor_img = torch.from_numpy(normalized_img).permute(2,0,1)
	tensor_img = tensor_img.unsqueeze(0)
	tensor_img = tensor_img.to(device)
	with torch.no_grad():
		pafs, confs = model(tensor_img.float())
	upsample = torch.nn.Upsample(scale_factor=8, mode='bicubic')
	#pafs = upsample(pafs)
	final_stage_conf_map = confs[-1]
	#final_stage_conf_map = upsample(final_stage_conf_map)
	#print(torch.sum((final_stage_conf_map > 0).int()), 'non zero values in confidence map with background')
	nms_confs = nms(final_stage_conf_map)
	#print(torch.sum((nms_confs > 0).int()), ' non zero values in confidence map with background after non max supression')
	heatmaps = nms_confs[0, 0:18, :, :]
	#print(torch.sum((heatmaps > 0).int()), ' non zero values in confidence map without background after non max supression')
	parts = find_parts(heatmaps.squeeze(0).cpu().numpy(), upsample(pafs[0]).squeeze(0).cpu().numpy())
	break
	#img_array = io.imread(img_metadata['coco_url'])





