import os
import os.path
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import json
import cv2
import re
import time
from PIL import Image
import random
import imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import imageio
from collections import namedtuple
import re
import math
from numpy.linalg import norm
import torch.nn as nn
import time
from matplotlib.path import Path
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
import scipy.io as sio

np.random.seed(0)

class ImageHeatmapTransforms():
	def __init__(self, load_size, return_size):
		self.load_size, self.return_size = load_size, return_size
	def apply_random_transforms(self, img, pafs, confs):
		img, pafs, confs = self.crop(img, pafs, confs)
		if (np.random.uniform() < 1):
			#p = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0]
			p = [0] * 7
			transform_p = np.random.uniform(size=len(p))
			if transform_p[0] < p[0]:
				img = TF.adjust_brightness(img, np.random.uniform(low=0.5, high=1.5))
			if (transform_p[1] < p[1]):
				img = TF.adjust_contrast(img, np.random.uniform(low=0.5, high=1.5))
			if (transform_p[2] < p[2]):
				img = TF.adjust_saturation(img, np.random.uniform(low=0.0, high=2.0))
			if (transform_p[3] < p[3]):
				img = TF.adjust_hue(img, np.random.uniform(low=-0.1, high=0.1))
			if (transform_p[4] < p[4]):
				img = TF.gaussian_blur(img, int(np.random.choice([3, 5])))
			if (transform_p[5] < p[5]):
				img, pafs, confs = self.rotate(img, pafs, confs)
			if (transform_p[6] < p[6]):
				img, pafs, confs = TF.hflip(img), TF.hflip(pafs), TF.hflip(confs)
				right_conf_indices, left_conf_indices = [2, 3, 4, 8, 9, 10, 14, 16], [5, 6, 7, 11, 12, 13, 15, 17]
				confs[left_conf_indices], confs[right_conf_indices] = confs[right_conf_indices], confs[left_conf_indices]
				right_paf_indices, left_paf_indices = [0,1,2,3,4,5,12,13,14,15,16,17,18,19,30,31,32,33], [6,7,8,9,10,11,20,21,22,23,24,25,26,27,34,35,36,37]
				pafs[left_paf_indices], pafs[right_paf_indices] = pafs[right_paf_indices], pafs[left_paf_indices]
		return img, pafs, confs
	def rotate(self, img, pafs, confs):
		random_angle = int(np.random.choice(np.arange(-61,60)))
		img, pafs, confs = TF.rotate(img, random_angle), TF.rotate(pafs, random_angle), TF.rotate(confs, random_angle)
		return img, pafs, confs
	def crop(self, img, pafs, confs):
		top_img, left_img = np.random.choice(self.load_size - self.return_size, 2)
		top_heatmap, left_heatmap = int(top_img/8), int(left_img/8)
		img_cropped = TF.crop(img, top_img, left_img, self.return_size, self.return_size)
		pafs_cropped =  TF.crop(pafs, top_heatmap, left_heatmap, int(self.return_size/8), int(self.return_size/8))
		confs_cropped =  TF.crop(confs, top_heatmap, left_heatmap, int(self.return_size/8), int(self.return_size/8))
		return img_cropped, pafs_cropped, confs_cropped

def resize(xy, org_shape, new_shape):
	w, h = org_shape
	new_w, new_h = new_shape
	new_x = xy[0] * (new_w/ w)
	new_y = xy[1] * (new_h/ h)
	return [new_y, new_x]

class MPIIDataset(Dataset):
	def __init__(self, data_dir, annotations_path, load_size, return_size, limb_width, sigma, transform=None):
		self.return_size = return_size
		self.data_dir = data_dir
		self.transform = transform
		self.ImageHeatmapTransforms = ImageHeatmapTransforms(load_size, return_size)
		self.GT = GroundTruthGenerator((int(load_size/8), int(load_size/8)), limb_width, sigma)
		matfile = sio.loadmat(annotations_path, struct_as_record=False)
		self.load_size = (load_size, load_size)
		self.heatmap_size = (int(load_size/8), int(load_size/8))
		self.annolist = matfile['RELEASE'][0,0].__dict__['annolist']
		mpii_keypoints = {'right_ankle':0, 'right_knee':1, 'right_hip':2,'left_hip':3,'left_knee':4, 'left_ankle': 5, 'pelvis':6, 'thorax':7, 'upper_neck':8,  'head top':9,
				'right_wrist':10, 'right_elbow':11, 'right_shoulder':12, 'left_shoulder':13, 'left_elbow':14, 'left_wrist':15}

		openpose_keypoints =  {"nose":0, "neck":1, "right_shoulder":2, "right_elbow":3, "right_wrist":4, "left_shoulder":5, "left_elbow":6, "left_wrist":7, "right_hip":8,
							"right_knee":9, "right_ankle":10, "left_hip":11, "left_knee":12,"left_ankle":13, "right_eye":14, "left_eye":15, "right_ear":16,
							"left_ear":17}
		self.mpii_openpose_mapping = {0: 10, 1: 9, 2: 8, 3: 11, 4: 12, 5: 13, 6:-1, 7:-1, 8:-1, 9:-1, 10: 4, 11: 3, 12: 2, 13: 5, 14: 6, 15: 7}
		self.working_shape = 0
		joints_present = list(self.mpii_openpose_mapping.values())
		for i in range(joints_present.count(-1)):
			joints_present.remove(-1)
		list.sort(joints_present)
		self.joints_mask = []
		j = 0
		for i in range(19):
			try:
				m = joints_present[j]
			except:
				continue
			if (i != m):
				self.joints_mask.append(0)
			else:
				self.joints_mask.append(1)
				j += 1
		self.joints_mask += [0, 0, 0, 0, 1]
		self.joints_mask = torch.tensor(self.joints_mask).bool()
		limb_order = np.array([[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,2],[2,16],[3,4],[16,2],[1,5],[5,6], [6,7],[5,17],[0,1],[0,14], [0,15],[14,16], [15,17]])
		self.limbs_mask = torch.tensor([bool(self.joints_mask[limb_order[i][0]]) and bool(self.joints_mask[limb_order[i][1]]) for i in range(len(limb_order))]).repeat_interleave(2)
	def __len__(self):
		return self.annolist.shape[1]
	def __getitem__(self, idx):
		object_i = self.annolist[0, idx]
		fds = object_i.__dict__['image'][0,0]
		image_name = fds.__dict__['name'][0]
		image_path = os.path.join(self.data_dir, 'images', image_name)
		try:
			org_img = Image.open(image_path)
			org_size = org_img.size
			image = org_img.resize(self.load_size)
		except FileNotFoundError:
			#print('file not found')
			heatmap_size = int(self.return_size/8)
			image, pafs, confs = torch.zeros(3, self.return_size, self.return_size), torch.zeros(38, heatmap_size, heatmap_size), torch.zeros(19, heatmap_size, heatmap_size)
			return {'image':image, 'paf_mask': self.limbs_mask, 'conf_mask': self.joints_mask, 'pafs':pafs, 'confs': confs}
		annorect = object_i.__dict__['annorect']
		all_joints = []
		for i in (range(annorect.shape[1])):
			person_i_ann = annorect[0, i]
			if 'annopoints' in person_i_ann._fieldnames:
				annopoints_i = person_i_ann.__dict__['annopoints']
				for j in range(annopoints_i.shape[1]):
					person_i_points = np.array([])
					points = annopoints_i[0,j].__dict__['point']
					for p in range(points.shape[1]):
						x, y, i = points[0, p].__dict__['x'], points[0, p].__dict__['y'], points[0, p].__dict__['id']
						x,y = resize([x.item(), y.item()], org_size,  self.heatmap_size)
						person_i_points = np.append(person_i_points, [x, y, self.mpii_openpose_mapping[i.item()]], axis = 0)
						#person_i_points.append([x, y, i.item()])
						#all_joints = np.append(all_joints, [x, y, i.item()], axis = 0)
					person_i_points = person_i_points.reshape(-1, 3)
					present_joints = (person_i_points.reshape(-1, 3))[:, 2]
					absent_joints = set(range(18)) - set(present_joints)
					for a in absent_joints:
						person_i_points = np.append(person_i_points, np.array([[0,0,a]]), axis=0)
					person_i_points = person_i_points[np.argsort(person_i_points[:, 2])]
					rows_non_data = np.where(person_i_points==-1)[0]
					person_i_points = np.delete(person_i_points, rows_non_data, axis=0)
					all_joints.append(person_i_points)

		if (len(all_joints) == 0):
			image = transforms.ToTensor()(image)
			if self.transform:
				image = self.transform['image'](image)
			confs = torch.zeros(18, int(self.load_size[0]/8), int(self.load_size[1]/8))
			back_conf = torch.ones((1, int(self.load_size[0]/8), int(self.load_size[1]/8)))
			confs = torch.cat([confs, back_conf], dim=0)
			pafs = torch.zeros(38, int(self.load_size[0]/8), int(self.load_size[1]/8))
			image, pafs, confs = self.ImageHeatmapTransforms.apply_random_transforms(image, pafs, confs)
			return {'image':image, 'conf_mask': self.joints_mask, 'paf_mask': self.limbs_mask, 'pafs':pafs, 'confs': confs}
		try:
			all_joints = np.stack(all_joints)
			#print('before stacking all joints', all_joints)
			all_joints = np.stack([all_joints[:, i, :] for i in range(18)])[:, :, 0:2]
		except:
			hs = (int(self.return_size/8), int(self.return_size/8))
			image, pafs, confs = torch.zeros(3, self.return_size, self.return_size), torch.zeros(38, hs[0], hs[1]), torch.zeros(19, hs[0], hs[1])
			return {'image': image, 'paf_mask': self.limbs_mask, 'conf_mask': self.joints_mask, 'pafs': pafs, 'confs':confs}
		confs = self.GT.generate_confidence_maps(all_joints)
		pafs = torch.from_numpy(self.GT.generate_pafs(all_joints))
		n_people = all_joints.shape[1]
		"""
		if (self.rot):
			random_angle = int(np.random.choice(np.arange(-45,46)))
			image, pafs, confs = TF.rotate(image, random_angle), TF.rotate(pafs, random_angle), TF.rotate(confs, random_angle)
			pafs = rotate_paf_heatmap(pafs, random_angle)
		"""
		image = transforms.ToTensor()(image)
		image, pafs, confs = self.ImageHeatmapTransforms.apply_random_transforms(image, pafs, confs)

		if (self.transform):
			image = self.transform['image'](image)
		if (image.shape[0] == 1):
			img = image[0]/3
			image = torch.stack([img, img, img])
		return {'image': image, 'confs': confs, 'pafs':pafs, 'conf_mask': self.joints_mask, 'paf_mask': self.limbs_mask} #,'confs':confs, 'pafs':pafs}

class DanceDataset(Dataset):
	def __init__(self, images_csv, load_size, return_size, limb_width, sigma,transform=None):
		self.images_csv = images_csv
		self.return_size = return_size
		self.transform = transform
		self.ImageHeatmapTransforms = ImageHeatmapTransforms(load_size, return_size)
		dance_type_list = list(set(self.images_csv['dance_type']))
		dance_type_list.sort(key=str.lower)
		self.dance_type_dict = {d:i for i, d in enumerate(dance_type_list)}
		self.img_size = np.array([load_size, load_size])
		#self.heatmap_size = self.img_size
		self.heatmap_size = (int(self.img_size[0]/8), int(self.img_size[1]/8)) #openpose downsamples the image size by 8
		#device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.GT = GroundTruthGenerator(self.heatmap_size, limb_width, sigma)
		self.no_kps = 0
		
		self.keypoints_dir = os.path.join(os.getcwd(), 'data/letsdance/keypoints')
	def __len__(self):
		return (len(self.images_csv.index))
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		dance_type = self.images_csv.iloc[idx,1]
		img_path = os.path.join(os.getcwd(), 'data/letsdance', self.images_csv.iloc[idx,0][5:])
		frame_url = img_path[img_path.rfind("/") + 1:-4]
		keypoints_path = os.path.join(self.keypoints_dir, dance_type, frame_url + ".json")
		with open(os.path.join(keypoints_path), 'r') as read_file:
			key_data = np.array(json.load(read_file))
		org_img = Image.open(img_path)
		image = org_img.resize(tuple(self.img_size))
		COCO_keypoints =  {"nose":0, "neck":1, "right_shoulder":2, "right_elbow":3, "right_wrist":4,
                                "left_shoulder":5, "left_elbow":6, "left_wrist":7, "right_hip":8,
                                "right_knee":9, "right_ankle":10, "left_hip":11, "left_knee":12,
                                "left_ankle":13, "right_eye":14, "left_eye":15, "right_ear":16,
                                "left_ear":17, "background":18}
		annotation_dict = {}
		for i in range(len(key_data)):
			person_i = key_data[i].reshape(1,-1)
			body_parts_info = person_i[0, 1:person_i.shape[1]]
			body_parts_keys = [body_parts_info[i][0] for i in range(person_i.shape[1] - 1)]
			body_parts_xy = [body_parts_info[i][1] for i in range(person_i.shape[1] -  1)]
			projected_xy = [resize(k, org_img.size, self.heatmap_size) for k in body_parts_xy]
			annotation_dict[i] = dict(zip(body_parts_keys, projected_xy))
		joints_list = []
		for i in annotation_dict:
			single_dict = {COCO_keypoints[k]:annotation_dict[i][k] for k in annotation_dict[i]}
			single_dict[1] = [(single_dict[2][0] + single_dict[5][0])/2, (single_dict[2][1] + single_dict[5][1])/2]
			sorted_dict =  {}
			for key in (sorted(single_dict.keys())):
				sorted_dict[key] = single_dict[key]
			joints_list.append(list(sorted_dict.values()))
		joints_list = np.array(joints_list)
		#co-ordinates of joints per joint type. eg. if a data has 8 people, every joint type will have eight co-ordinates(unless missing data)
		if (joints_list != []):
			joints_per_type = np.array([joints_list[:,j,:] for j in range(joints_list.shape[1])])
			n_people = joints_per_type.shape[1]
			pafs = torch.from_numpy(self.GT.generate_pafs(joints_per_type)).float()
			confs = self.GT.generate_confidence_maps(joints_per_type)
		else:
			n_people = 0
			pafs = torch.zeros(38, self.heatmap_size[0], self.heatmap_size[1])
			confs = torch.zeros(18, self.heatmap_size[0], self.heatmap_size[1])
			back_conf = torch.ones(1, self.heatmap_size[0], self.heatmap_size[1])
			confs = torch.cat([confs, back_conf], dim=0)
		image, pafs, confs = self.ImageHeatmapTransforms.apply_random_transforms(image, pafs, confs)
		image = transforms.ToTensor()(image)
		if (self.transform):
			image = self.transform['image'](image)
		if (image.shape[0] == 1):
			image = torch.stack([image[0]/3, image[0]/3, image[0]/3])
		sample = {'image':image, 'pafs':pafs, 'confs':confs, 'paf_mask': torch.tensor([True] * 38), 'conf_mask': torch.tensor([True] * 19)} #'dance_type':int(self.dance_type_dict[dance_type}
		return sample


#when rotating part affinity fields, the vector values also need to be changed
def rotate_paf_heatmap(pafs, angle):
	radians = (angle / 180) * math.pi
	x = torch.stack([pafs[2 * i] for i in range(19)])
	y = torch.stack([pafs[2 * i + 1] for i in range(19)])
	xx = x * math.cos(radians) + y * math.sin(radians)
	yy = -x * math.sin(radians) + y * math.cos(radians)
	rotated_paf = []
	for i in range(19):
		rotated_paf.append(xx[i])
		rotated_paf.append(yy[i])
	return torch.stack(rotated_paf)

def flatten_list(lst):
	l = []
	for e in lst:
		for el in e:
			l.append(el)
	return l


class COCODataset(Dataset):
	def __init__(self, datadir, partition, load_size, return_size, limb_width, sigma, transform=None):
		self.return_size = return_size
		dataType = '{}2017'.format(partition)
		annFile = '{}/annotations/person_keypoints_{}.json'.format(datadir, dataType)
		self.datadir = './data/coco/images/{}'.format(dataType)
		self.coco_file = COCO(annFile)
		self.load_size = (load_size, load_size)
		self.return_size = (return_size, return_size)
		self.ImageHeatmapTransforms = ImageHeatmapTransforms(load_size, return_size)
		img_ids = list(self.coco_file.imgs.keys())
		self.img_ann_map = {img_ids[i]: self.coco_file.getAnnIds(imgIds=img_ids[i]) for i in range(len(img_ids))}
		j = 0
		"""
		for i in (list(self.img_ann_map.keys())):
			if (self.img_ann_map[i] == []):
				j += 1
				del(self.img_ann_map[i])
		"""
		self.ann_ids = flatten_list(list(self.img_ann_map.values()))
		self.img_ids = list(self.img_ann_map.keys())
		self.transform = transform
		self.heatmap_size = (int(self.load_size[0]/8), int(self.load_size[1]/8))
		self.final_heatmap_size = (int(self.return_size[0]/8), int(self.return_size[1]/8))
		openpose_keypoints =  {"nose":0, "neck":1, "right_shoulder":2, "right_elbow":3, "right_wrist":4,
								"left_shoulder":5, "left_elbow":6, "left_wrist":7, "right_hip":8,
								"right_knee":9, "right_ankle":10, "left_hip":11, "left_knee":12,
								"left_ankle":13, "right_eye":14, "left_eye":15, "right_ear":16,
								"left_ear":17}
		COCO_keypoints = ["nose",   "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
						 "right_shoulder", "neck", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
						  "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
		coco_kps_dict = dict(zip(COCO_keypoints, range(len(COCO_keypoints))))
		self.coco_to_openpose_index = []
		for o in (list(openpose_keypoints.keys())):
			self.coco_to_openpose_index.append(coco_kps_dict[o])
		self.GT = GroundTruthGenerator(self.heatmap_size, limb_width, sigma)
		self.joints_mask = torch.tensor([True] * 19)
		self.limbs_mask = torch.tensor([True] * 38)
	def __len__(self):
		return len(self.img_ids)
	def no_annotations(self):
		return len(self.ann_ids)
	def __getitem__(self, id):
		img_id = self.img_ids[id]
		img = self.coco_file.loadImgs(img_id)[0]
		img_path = os.path.join(self.datadir, img['file_name'])
		image = Image.open(img_path).convert('RGB')
		org_img_size = image.size
		image = image.resize(self.load_size)
		ann_ids = self.img_ann_map[img_id]
		#return a conf maps with 0 for every joint and 1 for background confidnce map, similarly return 0 for every paf limb
		if (ann_ids == []):
			confs = torch.zeros(18, int(self.load_size[0]/8), int(self.load_size[1]/8))
			back_conf = torch.ones((1, int(self.load_size[0]/8), int(self.load_size[1]/8)))
			confs = torch.cat([confs, back_conf], dim=0)
			pafs = torch.zeros(38, int(self.load_size[0]/8), int(self.load_size[1]/8))
			image, pafs, confs = self.ImageHeatmapTransforms.apply_random_transforms(image, pafs, confs)
			image = transforms.ToTensor()(image)
			if self.transform:
				image = self.transform['image'](image)
			#image = image / torch.max(image) - 0.5
			#print('no annotation data')
			return {'image': image, 'conf_mask': self.joints_mask, 'paf_mask': self.limbs_mask, 'pafs':pafs, 'confs': confs}
		anns = self.coco_file.loadAnns(ann_ids)
		keypoints_list = []
		num_people = 0
		for a in anns:
			if 'keypoints' in a.keys():
				num_people += 1
				keypoints_list.append(a['keypoints'])
		keypoints = np.array(keypoints_list)
		#keypoints_vis = (keypoints[:, 2] > 1).reshape(-1)
		#body_parts_xy = keypoints[:, 0:2].reshape(-1, 2)
		#resize the keypoints to the heatmap that is original load image size / 8
		one_arr_kps = keypoints.reshape(-1,3)
		#resizing the keypoints to match the heatmap's dimension
		keypoints = np.stack([(resize(k[0:2], org_img_size, self.heatmap_size) + [k[2]]) for k in keypoints.reshape(-1, 3)]).reshape(num_people, -1, 3)
		#image = image.resize(self.final_img_size)
		all_joints = []
		for i in range(keypoints.shape[0]):
			k = keypoints[i]
			neck_kp = np.zeros((1, 2))
			if (k[5][2] and k[6][2]):
				neck_kp = ((k[5][0:2] + k[6][0:2]) / 2).reshape(1, 2)
				#reshaping the joints co-ordinates array to shape that ground truth generator accepts
			joints_per_type = k[:, 0:2].reshape(17, 2)
			joints_per_type = np.concatenate((k[0:7,0:2], neck_kp, k[7:,0:2]), axis = 0)
			#permuting the joints xy to match the openpose joints order
			joints_per_type = joints_per_type[self.coco_to_openpose_index]
			all_joints.append(joints_per_type)
		all_joints = np.stack(all_joints)
		separated_joints = np.zeros((all_joints.shape[1], all_joints.shape[0], 2))
		for i in range(all_joints.shape[1]):
			separated_joints[i] = np.stack([all_joints[n,i] for n in range(num_people)])
		n_people = separated_joints.shape[1]
		pafs = torch.from_numpy(self.GT.generate_pafs(separated_joints)).float()
		confs = self.GT.generate_confidence_maps(separated_joints)
		image, pafs, confs = self.ImageHeatmapTransforms.apply_random_transforms(image, pafs, confs)
		image = transforms.ToTensor()(image)
		#image = image / torch.max(image) - 0.5
		n_people = separated_joints.shape[1]
		if (image.shape[0] == 1):
			img = image[0]/3
			image = torch.stack([img, img, img])
		if (self.transform):
			image = self.transform['image'](image)
		sample = {'image': image, 'pafs': pafs, 'confs':confs, 'paf_mask': torch.tensor([True] * 38), 'conf_mask': torch.tensor([True]*19)}
		return sample #, 'n_people': n_people}


class GroundTruthGenerator():
	def __init__(self, img_size, limbwidth, sigma):
		self.img_size = list(img_size)
		self.limb_order = np.array([[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[1,2],[2,3],[3,4],[16,2],[1,5],[5,6],[6,7],[5,17],[0,1],[0,14],[0,15],[14,16],[15,17]])
		self.single_map_indices = torch.from_numpy(np.array(list(np.ndindex(tuple(self.img_size)))).reshape(self.img_size[0], self.img_size[1], 2))
		self.empty_tensor = torch.zeros((self.img_size[0], self.img_size[1], 2))
		self.limb_width = limbwidth
		self.sigma = sigma

	def generate_maps_masks(self, separated_joints, sigma=3, nms_param=5,limb_width = 3):
		if (separated_joints == []):
			return torch.tensor([0]), torch.empty(18 + 2 * len(self.limb_order),self.img_size[0], self.img_size[1])
		else:
			conf_maps, paf_maps = self.generate_confidence_maps(separated_joints,sigma, nms_param) , self.generate_pafs(separated_joints, limb_width)
			all_maps = torch.cat((conf_maps, torch.from_numpy(paf_maps).float()), dim=0)
			return  torch.tensor([1]), all_maps

	def get_background_conf(self, confs, pow):
		sum_conf_maps = torch.sum(confs, dim=0)
		max_conf_value = torch.max(sum_conf_maps)
		normalized_sum = sum_conf_maps / max_conf_value
		background_conf = 1 - (normalized_sum ** (pow))
		background_conf = background_conf.unsqueeze(0)
		return background_conf

	def generate_confidence_maps(self, joints_list):
		sj = joints_list.flatten()
		all_diff = []
		for i in range(int(len(sj)/2)):
			joint_tensor, single_map_indices = self.empty_tensor, self.single_map_indices
			joint_tensor[:,:,0], joint_tensor[:,:,1] = sj[2*i], sj[2*i+1]
			#print(sj[2 * i], sj[2 * i +1])
			diff = single_map_indices - joint_tensor
			if (torch.all(torch.all(diff == single_map_indices))):
				#print("the if statement works")
				diff = diff.fill_(10)
			all_diff.append(diff)
		all_diff = torch.stack(all_diff).reshape(joints_list.shape[0], joints_list.shape[1], self.img_size[0], self.img_size[1], 2)
		conf_maps = torch.exp(-(torch.sqrt(torch.sum(all_diff * all_diff, dim=4)) ** 2/self.sigma **2))
		max_tensor = torch.max(conf_maps, dim=1)[0].reshape(joints_list.shape[0], self.img_size[0], self.img_size[1])
		background_conf = self.get_background_conf(max_tensor, 0.1)
		return torch.cat([max_tensor, background_conf], dim=0)

	def get_limb_masks(self, joints_list):
		joints_mask = self.get_joints_mask(joints_list)
		limbs_mask = []
		for l in self.limb_order:
			j1, j2 = joints_mask[l[0]], joints_mask[l[1]]
			single_limb_mask = torch.stack([j_mask[0] and j_mask[1] for j_mask in zip(j1, j2)])
			limbs_mask.append(single_limb_mask)
		limbs_mask = torch.stack(limbs_mask)
		return limbs_mask

	def get_joints_mask(self, joints):
		joint_mask = torch.empty((joints.shape[0], joints.shape[1], 1))
		for index in list(np.ndindex(joint_mask.shape[0:2])):
			joint_mask[index] = joints[index][0] + joints[index][1]
		return (joint_mask != 0)
	def get_paf(self, bd,unit_vec):
		return paf_vec_x, paf_vec_y
	def m_dot(self, vec1, vec2):
		return np.einsum('ij,ij->i', vec1, vec2)
	def get_magnitude(self, vec):
		return np.vstack([np.linalg.norm(v) for v in vec])
	def no_error_div(self, vec1, vec2):
		return np.divide(vec1, vec2, out=np.zeros_like(vec1), where=vec2!=0)
	def generate_pafs(self,joints, p=1):
		limbs_mask = self.get_limb_masks(joints)
		#saving positions of first and second joints of a limb in a separate array for easy computing
		#print(limbs_mask)
		limb1_co = np.vstack([joints[self.limb_order[l][0]] for l in range(len(self.limb_order))]).reshape(-1, 2)
		limb2_co = np.vstack([joints[self.limb_order[l][1]] for l in range(len(self.limb_order))]).reshape(-1,2)
		#finding magnitude of each limb vector for unit vector calculation
		diff = limb1_co - limb2_co
		mag = np.vstack([np.linalg.norm(d) for d in diff])
		unit_vecs = np.divide(diff, mag, out=np.zeros_like(diff), where=mag!=0)
		#finding perpendicular unit vector of a limb to get a box for the limb
		unit_vecs_p =  np.ones_like(unit_vecs) - np.einsum('ij,ij->i', unit_vecs, np.ones_like(unit_vecs)).reshape(-1,1) * unit_vecs
		mag = self.get_magnitude(unit_vecs_p)
		unit_vecs_p = self.no_error_div(unit_vecs_p, mag)
		#finding how much the co-ordinates of joints must be offset to get the bounding box co-ordinates for each limb
		vec_offset = unit_vecs_p * (self.limb_width/2)
		#shape of array containing all limbs separated by joint type
		sep_shape = (len(self.limb_order), joints.shape[1])
		#getting bounding boxes
		bounding_boxes_co = np.concatenate((limb1_co - vec_offset, limb1_co + vec_offset, limb2_co + vec_offset, limb2_co - vec_offset), axis = 1).reshape(len(self.limb_order), joints.shape[1], 8)
		#the joints are set to (0,0) if they are not present to acheive uniformity in data, so masking is used to not make bounding box with them
		limbs_inds = np.where(limbs_mask == 1)
		unit_vecs = unit_vecs.reshape((sep_shape[0], sep_shape[1], 2))
		limbs_mask_ext = np.tile(limbs_mask, (1, 2)).reshape(unit_vecs.shape)
		unit_vecs *= limbs_mask_ext

		#declaring outside of loop to same function being carried out everytime
		x, y = np.mgrid[:self.img_size[0], :self.img_size[1]]
		points = np.vstack((x.ravel(), y.ravel())).T
		paf_truths = []
		paf_vec_x, paf_vec_y = np.zeros(self.img_size), np.zeros(self.img_size)
		since = time.time()
		for i in range(sep_shape[0]):
			boxes = bounding_boxes_co[i]
			b_boxes = [np.array([(b[0], b[1]), (b[2], b[3]), (b[4],b[5]), (b[6],b[7])]) for b in boxes]
			unit_vec = unit_vecs[i]
			paths = [Path(b) for b in b_boxes]
			masks = np.array([path.contains_points(points) for path in paths])
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
		paf_truths = np.stack(paf_truths)
		return paf_truths
