import pandas as pd
import numpy as np
import csv

def parse_img_path(s):
	return s[find(s, '/')+1:-4]

def find(s, ch, i):
	return [i for i, itr in enumerate(s) if itr == ch][i]

def parse_frame(f):
	vdo_id = f[find(f, '/',-1)+1:find(f, '_',-1)]
	vdo_type = f[find(f, '/',1)+1:find(f, '/', 2)]
	frame_id = f[find(f,'_',-1)+1:-4]
	return vdo_id, vdo_type, frame_id

lets_dance_data = pd.read_csv('./data/letsdance/sorted_frames.csv')
all_frames = list(lets_dance_data['file_path'].values)

with open('./data/letsdance/video_frames.csv', 'w', newline='') as csvfile:
	i, frame_ids, tot_frames = 0, [], len(all_frames)
	fieldnames = ['video_id', 'dance_type', 'frames']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	dict_to_write = {}
	while(True):
		if (i == tot_frames-1):
			writer.writerow(dict_to_write)
			break
		f_curr, f_next = all_frames[i], all_frames[i+1]
		vdo_id, dance_type, frame_id = parse_frame(f_curr)
		vdo_id_next, dance_type_next, frame_id_next = parse_frame(f_next)
		if (frame_id not in frame_ids):
			frame_ids.append(frame_id)
		if (vdo_id_next != vdo_id):
			dict_to_write = {'video_id': vdo_id, 'dance_type': dance_type, 'frames': frame_ids}
			writer.writerow(dict_to_write)
			frame_ids = []
		i = i+1

