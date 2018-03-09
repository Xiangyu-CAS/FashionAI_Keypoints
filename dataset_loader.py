import torch
import torch.utils.data as data
import numpy as np
import os
import math
from PIL import Image
import cv2
import csv

class dataset_loader(data.Dataset):

	def __init__(self, img_dir, ann_path, stride, transforms=False):

		self.sigma = 15 #9 #15
		self.stride = stride
		self.img_dir = img_dir
		self.trasforms = transforms
		self.anns = []
		self.info = []
		with open(ann_path,'rb') as f:
			reader = csv.reader(f)
			for row in reader:
				self.anns.append(row)
		self.info.append(self.anns[0])
		self.anns=self.anns[1:]


	def __getitem__(self, index):
		# ---------------- read info -----------------------
		ann = self.anns[index]
		img_path = os.path.join(self.img_dir, ann[0])
		img = cv2.imread(img_path) # BGR
		catergory = ann[1]
		kpt = _get_keypoints(ann)
		# ----------------- transform ----------------------
		# croppad
		center = [img.shape[0]/2,img.shape[1]/2]
		img, kpt = _croppad(img, kpt, center, 384, 384)
		heatmaps = _generate_heatmap(img, kpt,self.stride, self.sigma)

		img = np.array(img, dtype=np.float32)
		img -= 128.0
		img /= 255.0

		img = torch.from_numpy(img.transpose((2, 0, 1)))
		heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)))

		# img = self.trasforms(img)
		# heatmaps = self.trasforms(heatmaps)

		return img, heatmaps

	def __len__(self):
		return len(self.anns)

def _croppad(img, kpt, center, w, h):
	num = len(kpt)
	height, width, _ = img.shape
	new_img = np.empty((h, w, 3), dtype=np.float32)
	new_img.fill(128)

	# calculate offset
	offset_up = -1*(h/2 - center[0])
	offset_left = -1*(w/2 - center[1])

	for i in range(num):
		kpt[i][0] -= offset_left
		kpt[i][1] -= offset_up

	st_x = 0
	ed_x = w
	st_y = 0
	ed_y = h
	or_st_x = offset_left
	or_ed_x = offset_left + w
	or_st_y = offset_up
	or_ed_y = offset_up + h

	if offset_left < 0:
		st_x = -offset_left
		or_st_x = 0
	if offset_left + w > width:
		ed_x = width - offset_left
		or_ed_x = width
	if offset_up < 0:
		st_y = -offset_up
		or_st_y = 0
	if offset_up + h > height:
		ed_y = height - offset_up
		or_ed_y = height
	new_img[st_y: ed_y, st_x: ed_x, :] = img[or_st_y: or_ed_y, or_st_x: or_ed_x, :].copy()

	return np.ascontiguousarray(new_img), kpt


def _get_keypoints(ann):
	kpt = np.zeros((24, 3))
	for i in range(2, len(ann)):
		str = ann[i]
		[x_str, y_str, vis_str] = str.split('_')
		kpt[i - 2, 0], kpt[i - 2, 1], kpt[i - 2, 2] = int(x_str), int(y_str), int(vis_str)
	return kpt

def _generate_heatmap(img, kpt, stride, sigma):
	height, width, _ = img.shape
	heatmap = np.zeros((height / stride, width / stride, len(kpt) + 1), dtype=np.float32) # (24 points + background)
	height, width, num_point = heatmap.shape
	start = stride / 2.0 - 0.5

	num = len(kpt)
	for i in range(num):
		if kpt[i][2] == -1:  # not labeled
			continue
		x = kpt[i][0]
		y = kpt[i][1]
		for h in range(height):
			for w in range(width):
				xx = start + w * stride
				yy = start + h * stride
				dis = ((xx - x) * (xx - x) + (yy - y) * (yy - y)) / 2.0 / sigma / sigma
				if dis > 4.6052:
					continue
				heatmap[h][w][i] += math.exp(-dis)
				if heatmap[h][w][i] > 1:
					heatmap[h][w][i] = 1

	heatmap[:, :, -1] = 1.0 - np.max(heatmap[:, :, :-1], axis=2)  # for background
	return heatmap

'''
0: labeled but not visble
1: labeled and visble
-1: not labeled

'image_id',
 'image_category',
0'neckline_left',
1'neckline_right',
2 'center_front',
3'shoulder_left',
4 'shoulder_right',
5 'armpit_left',
6 'armpit_right',
7 'waistline_left',
8 'waistline_right',
9 'cuff_left_in',
10 'cuff_left_out',
11 'cuff_right_in',
12 'cuff_right_out',
13 'top_hem_left',
14 'top_hem_right',
15 'waistband_left',
16 'waistband_right',
17 'hemline_left',
18 'hemline_right',
19 'crotch',
20 'bottom_left_in',
21 'bottom_left_out',
22 'bottom_right_in',
23 'bottom_right_out
'''