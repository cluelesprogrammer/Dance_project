import models.models as models
import datasets.dataset as dataset
import os
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sys
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_loaders import dance_video_dataloaders


def visualize_split_composition(df, s):
	split_composition = dict(df['dance_type'].value_counts())
	composition = [[it[0],it[1]] for it in split_composition.items()]
	composition_table = wandb.Table(data=composition, columns=['dance_style', 'number_of_occurences'])
	dict_to_log = {'{}_composition'.format(s): wandb.plot.bar(composition_table, 'dance_style', 'number_of_occurences', title='{}_composition'.format(s))}
	wandb.log(dict_to_log)


def validate(config):
	with wandb.init(project='classifier_validation', config=config, name='whateveritwillchange', dir='wandbdir'):
		os.environ['WANDB_SILENT'] = 'true'
		config = wandb.config
		device = torch.device('cuda:3')
		network = models.RGB_Flow_Classifier()
		wandb.run.name = 'rgb_flow_classifier'
		network.to(device)
		checkpoint = torch.load('models/classifier/i3d_model_chckpoint_2_epoch_51.pt', map_location=torch.device('cpu'))
		network.load_state_dict(checkpoint['model_state_dict'])
		softmax = nn.Softmax(dim=1)
		train_df, val_df = pd.read_csv('data/letsdance/train_videos.csv'), pd.read_csv('data/letsdance/val_videos.csv')
		#visualize_split_composition(train_df, 'train')
		#visualize_split_composition(val_df, 'val')
		val_composition = dict(val_df['dance_type'].value_counts())
		dataloaders = dance_video_dataloaders(None, val_df, config.img_size, config.batch_size, optical_flow=config.optical_flow, n_frames=config.n_frames, frames_to_skip=config.frames_to_skip)
		dataset_style_counts = dict(val_df['dance_type'].value_counts())
		labels = list(dataset.DanceVideoDataset.label_to_id.keys())
		correct_style_predictions = dict(zip(labels, [0]*16))
		id_to_label = {value:key for (key, value) in dataset.DanceVideoDataset.label_to_id.items()}
		# Run the training loop for defined number of epochs
		best_accuracy = 0.0
		correct, total = 0, 0
		network.eval()
		val_targets, val_predictions = [], []
		for data in tqdm(dataloaders['val']):
			frames, targets, flows = data['frames'].to(device), data['dance_type'].to(device), data['optical_flow'].to(device)
			val_targets.append(targets.cpu().numpy())
			with torch.no_grad():
				outputs = network(frames, flows)
			#outputs = softmax(outputs)
			_, predicted = torch.max(outputs.data, 1)
			print(targets, predicted)
			val_predictions.append(predicted.cpu().numpy())
			style, correct_instances = torch.unique(predicted[predicted==targets], return_counts=True)
			correct_in_batch = dict(zip(style.cpu().numpy(), correct_instances.cpu().numpy()))
			for c in correct_in_batch:
				correct_style_predictions[id_to_label[c]] += correct_in_batch[c]
			total += targets.size(0)
			correct += (predicted == targets).sum().item()
		"""
		accuracy_table = wandb.Table(data=data_accuracy_by_style, columns=['dance_style', 'accuracy'])
		wandb.log({'accuracy_bar_chart': wandb.plot.bar(accuracy_table, 'dance_style', 'accuracy', title='accuracy_of_the_model_according_to_dance_style')})
		accuracy = 100.0 * correct/total
		wandb.log({'accuracy': accuracy})
		"""
		val_targets, val_predictions = np.concatenate(val_targets, axis=0), np.concatenate(val_predictions, axis=0)
		wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, y_true=val_targets, preds=val_predictions, class_names=labels)})
		conf_matrix = confusion_matrix(val_targets, val_predictions)
		print(conf_matrix)
		dance_style_accuracy = {}
		for (i,l) in enumerate(labels):
			dance_style_accuracy[l] = 100 * correct_style_predictions[l] / dataset_style_counts[l]
		data_accuracy_by_style = [[l, accur] for (l, accur) in dance_style_accuracy.items()]
		accuracy_table = wandb.Table(data=data_accuracy_by_style, columns=['dance_style', 'accuracy'])
		wandb.log({'accuracy_bar_chart': wandb.plot.bar(accuracy_table, 'dance_style', 'accuracy', title='accuracy_of_the_model_according_to_dance_style')})
		accuracy = 100.0 * correct/total
		wandb.log({'accuracy': accuracy})

config = {'batch_size': 6, 'img_size': 224, 'wandb_run': 'i3d_bodypose_model', 'optical_flow': True, 'n_frames':64, 'frames_to_skip':0}
validate(config)

