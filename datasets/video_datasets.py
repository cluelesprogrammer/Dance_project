import os
import os.path
import torch
from skimage import io, transform
import numpy as np
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

def get_optical_flow(files_list, resize_shape, flow_every):
	frame1 = cv2.imread(files_list[0])
	#frame1.resize((resize_shape[1], resize_shape[0]))
	prvs = cv2.cvtColor(np.float32(frame1), cv2.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	flows = []
	for i in range(2, len(files_list)):
		if ((i-1)%flow_every==0):
			frame1 = cv2.imread(files_list[i-1])
			prvs = cv2.cvtColor(np.float32(frame1), cv2.COLOR_BGR2GRAY)
			frame2 = cv2.imread(files_list[i])
			next = cv2.cvtColor(np.float32(frame2), cv2.COLOR_BGR2GRAY)
			prvs, next = cv2.resize(prvs, resize_shape), cv2.resize(next, resize_shape)
			flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			flows.append(torch.from_numpy(flow))
	return flows

class DanceVideoDataset(Dataset):
	label_to_id = {'rumba':0, 'break':1, 'flamenco':2, 'foxtrot':3, 'samba':4, 'cha':5, 'quickstep':6, 'latin':7, 'jive':8, 'pasodoble':9, 'swing':10, 'ballet':11, 'square':12, 'tango':13, 'tap':14, 'waltz':15}
	def __init__(self, return_size, df, n_frames=64, frames_to_skip=0, transform=None, optical_flow=None, flow_every=8):
		self.video_df = df
		self.flow_every = flow_every
		self.n_frames = n_frames
		self.return_size = return_size
		self.transform = transform
		self.frames_dir = frames_dir
		self.optical_flow = optical_flow
		self.frames_to_skip = frames_to_skip
		self.dance_type_dict = {'rumba':0, 'break':1, 'flamenco':2, 'foxtrot':3, 'samba':4, 'cha':5, 'quickstep':6, 'latin':7, 'jive':8, 'pasodoble':9, 'swing':10, 'ballet':11, 'square':12, 'tango':13, 'tap':14, 'waltz':15}
	def __len__(self):
		return len(self.video_df)
	def __getitem__(self, id):
		vdo_id, dance_type, frames = self.video_df.iloc[id, 0], self.video_df.iloc[id, 1], self.video_df.iloc[id, 2]
		frames = re.findall('\d{4}', frames)
		temporal_footprint = (self.frames_to_skip+1) * self.n_frames
		start_frame_i = np.random.choice(len(frames) - temporal_footprint , 1)[0]
		if (start_frame_i + temporal_footprint > len(frames)):
			start_frame_i = 0
		else:
			start_frame_i = len(frames) // 2 - temporal_footprint // 2  #take the center 16 frames
			if (start_frame_i + temporal_footprint > len(frames)):
				start_frame_i = 0
		frames_ids = [frames[start_frame_i + i * (1 + self.frames_to_skip)] for i in range(self.n_frames)]
		"""
		frames_ids = [str(start_frame_i + i*(self.frames_to_skip+1)) for i in range(self.n_frames)]
		for i in range(len(frames_ids)):
			if (len(frames_ids[i]) == 3):
				frames_ids[i] = '0' + frames_ids[i]
			elif (len(frames_ids[i]) == 2):
				frames_ids[i] = '00' + frames_ids[i]
			elif (len(frames_ids[i]) == 1):
				frames_ids[i] = '00' + frames_ids[i]
			else:
				continue
		"""
		#chosen_frames = [frames[i] for i in frames_ids]
		#processing the first frame to get the size ratio
		frame_paths = [os.path.join(self.frames_dir, dance_type, vdo_id+'_'+f+'.jpg') for f in frames_ids]
		pil_frame = Image.open(frame_paths[0])
		pil_size = np.array(pil_frame.size)
		min_dim = np.argmin(pil_size)
		resize_ratio = self.return_size / pil_size[min_dim]
		resize_shape = tuple((pil_size * resize_ratio).astype(int))
		tensor_frames, image_paths = [], []
		cv2_frames = []
		skip_frames = 1
		for frame_path in frame_paths:
			image_paths.append(frame_path)
			pil_frame = Image.open(frame_path)
			#cv2_frame = cv2.imread(frame_path)
			pil_frame = pil_frame.resize(resize_shape)
			#cv2_frames.append(cv2.resize(cv2_frame, resize_shape))
			#tensor_frame = transforms.ToTensor()(pil_frame)
			tensor_frame = transforms.ToTensor()(pil_frame)
			tensor_frame = self.transform['frames'](tensor_frame)
			tensor_frames.append(tensor_frame)
		tensor_video = torch.stack(tensor_frames)
		tensor_video = tensor_video.permute(1, 0, 2, 3)
		dict_to_return = {}
		dict_to_return['frames'] = tensor_video
		dict_to_return['dance_type'] = self.dance_type_dict[dance_type]

		if (self.optical_flow):
			optical_flow = torch.stack(get_optical_flow(image_paths, tuple(resize_shape), self.flow_every))
			optical_flow = optical_flow.permute(3, 0, 1, 2)
			try:
				optical_flow = self.transform['optical_flow'](optical_flow)
			except:
				print('center or equivalent crop required for optical flow otherwise the images do not match')

			dict_to_return['optical_flow'] = optical_flow
		return dict_to_return
