--------Openpose config file details--------

dataset=letsdancecoco To make two dataloaders lets dance and coco for older model validation on both letsdane and coco
dataset=letsdance To train and validate the model just on letsdance model
dataset=combined To train and validate the model on a dataset made from combining lets dance and coco dataset

model_type=old|new
for older and newer versions of Openpsoe respectively
trained=True or False for the older model
paf_stages=[1..] number of paf stages of newer openpose model
conf_stages=[1..] number of conf stages of newer openpose model

train_part=last_layer to just train the last layer of older model(does not work with newer model as it need full body training)
train_part=whole for whole model training

ftrs_opt=sgd only sgd is supported
features scheduler can be be only cyclic or None
paf_opt=sgd|adamw
conf_opt=sgd|adamw


--------Dance Classifier config details-------
frame_size=112|224 other frame sizes could also be tried however these are appropriate for c3d and i3d models respectively
flow_every=[1..] to select flow with or without skipping frames
frames_to_skip=[0..] to increase the temporal length of the samples
n_frames=16|64 16 for c3d while 32 for i3d

model_type=c3d|rgb_pose|rgb_flow|rgb|pose|flow all except c3d are i3d models, with x_y representing two streams, pose and flow models do not use rgb
train_flow=True|False train the whole flow stream when flow is being used
train_rgb=True|False train the rgb stream 









