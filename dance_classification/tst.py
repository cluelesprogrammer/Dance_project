from data_loader import dance_video_dataloaders

d = dance_video_dataloaders(112,24)
for k in d:
	print(len(d[k]))
