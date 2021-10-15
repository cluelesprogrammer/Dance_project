from pycocotools.coco import COCO
import json
import time
import random

cocoGT = COCO('data/coco/annotations/person_keypoints_val2017.json')
json_data = []
imgids = cocoGT.getImgIds()

for i in cocoGT.getImgIds():
	for j in range(7):
		k = [1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1,1,2,1]
		dummyfile = {'image_id': 0, 'category_id': 1, 'keypoints': k, 'score': 1.0}
		dummyfile['image_id'] = i
		json_data.append(dummyfile)


with open('dummy_detections.json', 'w') as outfile:
	json.dump(json_data, outfile)
