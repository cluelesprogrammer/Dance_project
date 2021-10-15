import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix
import datasets.dataset as dataset
import os
from collections import Counter

os.environ['WANDB_SILENT'] = 'true'
wandb.init(project='test_charts', name='test_val_split', dir='wandbdir')
df = pd.read_csv('data/letsdance/videos.csv')
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=2)
X_ids, data_labels = list(df.index), np.array([df['dance_type'].iloc[i] for i in range(len(df.index))])
skf.get_n_splits(np.array(X_ids), np.array(data_labels))
i = 0
for train_idx, val_idx in skf.split(X_ids, data_labels):
	train_labels, val_labels = data_labels[train_idx], data_labels[val_idx]
	val_labels, val_label_counts= list(Counter(val_labels).keys()), list(Counter(val_labels).values())
	train_labels, train_label_counts= list(Counter(train_labels).keys()), list(Counter(train_labels).values())
	val_split_table = [[val_labels[i], val_label_counts[i]] for i in range(len(val_labels))]
	val_wandb_table = wandb.Table(data=val_split_table, columns=['dance_style', 'counts'])
	wandb.log({'val_class_split_{}'.format(i): wandb.plot.bar(val_wandb_table, 'dance_style', 'counts', title='val_class_split_{}'.format(i))})
	train_split_table = [[train_labels[i], train_label_counts[i]] for i in range(len(train_labels))]
	train_wandb_table = wandb.Table(data=train_split_table, columns=['dance_style', 'counts'])
	wandb.log({'train_class_split_{}'.format(i): wandb.plot.bar(train_wandb_table, 'dance_style', 'counts', title='train_class_split_{}'.format(i))})
	i += 1
"""

skf = StratifiedKFold(n_splits=6)
for train_index, test_index in skf.split(dummy_X, dummy_Y):
	print(dummy_X[train_index])

print(all_videos_df)

kf = kfold(n_splits=5)
X = np.arange(len(all_frames.index))
y = np.arange(len(all_frames.index))

#test_ids = [t[1] for t in split_indices]
folds = []
for k in range(5):
	kf = kfold(n_splits=5)
	split_indices = kf.split(X, y)
	for train_id, test_id in split_indices:
		folds.append(train_id)
		print(k)
		break
print(folds[0].all() == folds[1].all() == folds[2].all() == folds[3].all() == folds[4].all())

	for i in range(len(test_ids)):
		id = set(test_ids[i])
		if (i == (len(test_ids) -1)):
			break
		for j in range(i+1, len(test_ids)):
			id_next = test_ids[j]
			intersection = id.intersection(id_next)
			print(len(intersection))
"""
