Code for training Openpose model(sequential CNN based model produces joints' and limbs' heatmaps) on dance dataset. The code trains the Openpose model that has been trained on COCO pose estimation model with the Let's dance dataset that features dancers in numerous dance performances scenarios. 


-------------------------------Openpose-------------------------------

Training Openpose:

- python openpose_train.py <arguments> to train openpose model

- python openpose_train.py --help for options and their definitions

- go to config/openpose.ini file to adjust the hyperparameters according to the experiment you want to run. Read Readme.md in config folder to see which values for different hyperparameters are supported

-Features, part affinity fields and confidence maps can be trained separately with optimizers of their own. Adamw is supported for part affinity field and confidence maps. Similarly, only final layers can also be trained on the older model


------------------------Dance Classification----------------------------

Training Classifier models

- python classifier_train.py <arguments>

- python classifier_train.py --help options and their definitions

- go to config/classifier.ini to select model to train, and numerous hyperparameter. Read Readme.md in config folder for hyperparameters definition and values supported

When both the streams of two streamed i3d models are set not to be trained, only the final layers(conv3d and final linear layer are trained) See the images i3d_pose.png and i3d_flow.png for this training schema

-----------------------Dataset-------------------------------------

For lets dance

- Download and upzip the lets dance dataset on data/ directory, put the frames in frames/ directory and keypoints in keypoints/ directory. frames and keypoints are both sorted according to dance types

- Train/val/test split in csv files of keypoints are put in pose_training_split/ directory. Alternatively, csv file resembling the train val test split can also be created

- Train_videos.csv, Val_videos.csv are also present. Another split to resemble the structre of this csv file can also be created

The structure is:
-data
 -letsdance
  -frames
  -keypoints
  -pose_training_split
   -train.csv
   -val.csv
   -test.csv
  -videos_split
   -train_videos.csv
   -val_videos.csv



