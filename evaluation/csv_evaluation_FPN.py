import csv
import os
import sys
import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
import math, time
import torch
import csv
import util
sys.path.append('../')
def apply_model(oriImg, model, multiplier):
	stride = 8
	height, width, _ = oriImg.shape
	normed_img = np.array(oriImg, dtype=np.float32)
	heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 25), dtype=np.float32)
	for m in range(len(multiplier)):
		scale = multiplier[m]
		imageToTest = cv2.resize(normed_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
		# imgToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, 128)
		imgToTest_padded, pad = util.padRightDownCorner(imageToTest, 64, 128)

		input_img = np.transpose(np.float32(imgToTest_padded[:, :, :, np.newaxis]),
		                         (3, 2, 0, 1)) / 255 - 0.5  # required shape (1, c, h, w)

		input_var = torch.autograd.Variable(torch.from_numpy(input_img).cuda())

		# get the features
		heat = model(input_var)
		# heat = model(input_var)

		# get the heatmap
		heatmap = heat.data.cpu().numpy()
		heatmap = np.transpose(np.squeeze(heatmap), (1, 2, 0))  # (h, w, c)
		heatmap = cv2.resize(heatmap, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
		heatmap = heatmap[:imgToTest_padded.shape[0] - pad[2], :imgToTest_padded.shape[1] - pad[3], :]
		heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
		heatmap_avg = heatmap_avg + heatmap / len(multiplier)

	all_peaks = []  # all of the possible points by classes.
	peak_counter = 0
	thre1 = 0.1
	for part in range(25 - 1):
		x_list = []
		y_list = []
		map_ori = heatmap_avg[:, :, part]
		map = gaussian_filter(map_ori, sigma=3)

		map_left = np.zeros(map.shape)
		map_left[1:, :] = map[:-1, :]
		map_right = np.zeros(map.shape)
		map_right[:-1, :] = map[1:, :]
		map_up = np.zeros(map.shape)
		map_up[:, 1:] = map[:, :-1]
		map_down = np.zeros(map.shape)
		map_down[:, :-1] = map[:, 1:]

		peaks_binary = np.logical_and.reduce(
			(map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > thre1))
		peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])  # note reverse
		peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
		id = range(peak_counter, peak_counter + len(peaks))
		peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

		all_peaks.append(peaks_with_score_and_id)
		peak_counter += len(peaks)

	# sort by score
	for i in range(24):
		all_peaks[i] = sorted(all_peaks[i], key=lambda ele : ele[2],reverse = True)

	canvas = oriImg.copy()
	# draw points
	for i in range(24):
		for j in range(len(all_peaks[i])):
			if j is 0:
				cv2.circle(canvas, all_peaks[i][j][0:2], 4, [0, 0, 255], thickness=-1)
			else:
				cv2.circle(canvas, all_peaks[i][j][0:2], 4, [255, 0, 0], thickness=-1)

	keypoints = -1*np.ones((24, 3))
	for i in range(24):
		if len(all_peaks[i]) == 0:
			continue
		else:
			keypoints[i,0], keypoints[i,1], keypoints[i,2] = all_peaks[i][0][0], all_peaks[i][0][1], 1

	return  keypoints, canvas


def write_csv(name, results):
	import csv
	with open(name, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(results)

def prepare_row(ann, keypoints):
	# cls
	image_name = ann[0]
	category = ann[1]
	keypoints_str = []
	for i in range(24):
		cell_str = str(int(keypoints[i][0])) + '_' + str(int(keypoints[i][1])) + '_' + str(int(keypoints[i][2]))
		keypoints_str.append(cell_str)
	row = [image_name, category] + keypoints_str
	return row

def read_csv(ann_file):
	info = []
	anns = []
	with open(ann_file, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			anns.append(row)
	info = anns[0]
	anns = anns[1:]
	return info, anns

def euclidean_distance(a, b):
	return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def criterion(ann_gt, ann_dt):
	category = ann_gt[1]
	gt_kpt = -1 * np.ones((24, 3))
	for i in range(len(gt_kpt)):
		x_str, y_str, vis_str = ann_gt[i + 2].split('_')
		gt_kpt[i][0], gt_kpt[i][1], gt_kpt[i][2] = int(x_str), int(y_str), int(vis_str)

	dt_kpt = -1 * np.ones((24, 3))
	for i in range(len(dt_kpt)):
		x_str, y_str, vis_str = ann_dt[i + 2].split('_')
		dt_kpt[i][0], dt_kpt[i][1], dt_kpt[i][2] = int(x_str), int(y_str), int(vis_str)

	if category in ['blouse','outwear','dress']: # armpit distance
		thre = euclidean_distance(gt_kpt[5], gt_kpt[6])
	elif category in ['trousers', 'skirt']: # waistband distance
		thre = euclidean_distance(gt_kpt[7], gt_kpt[8])
	if thre == 0:
		return []
	score = []
	for i in range(len(gt_kpt)):
		if gt_kpt[i][2] == 1:
			#if dt_kpt[i][2] == -1:
			#	score.append(2)
			#else:
			score.append(1.0* euclidean_distance(gt_kpt[i],dt_kpt[i])/ thre)
	return score
	#print('score = {}'.format(score))



def evaluate(gt_file, dt_file, num_imgs):
	info_gt, anns_gt = read_csv(gt_file)
	info_dt, anns_dt = read_csv(dt_file)
	anns_gt = anns_gt[:num_imgs]
	assert len(anns_gt) == len(anns_dt)
	scores = []
	for i in range(len(anns_gt)):
		ann_gt = anns_gt[i]
		ann_dt = anns_dt[i]
		score = criterion(ann_gt, ann_dt)
		scores += score
	value = sum(scores)/len(scores)
	print('score = {}'.format(value))

def eval():
	gt_file = '../FashionAI/data/train/Annotations/val.csv'
	# dt_file = 'val_result.csv'
	dt_file = 'modify.csv'

	num_imgs = 500
	evaluate(gt_file, dt_file,num_imgs)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # --------------------------- model -------------------------------------------------------------------------------
    import models.CPM_FPN
    pytorch_model = '../FashionAI/Heatmap/experiments/CPM_FPN/160000.pth.tar'
    model = models.CPM_FPN.pose_estimation(class_num=25, pretrain=False)
    # -----------------------------------------------------------------------------------------------------------------

    img_dir = '../FashionAI/data/train/'
    ann_path = '../FashionAI/data/train/Annotations/val.csv'
    # ann_path = '/data/xiaobing.wang/xiangyu.zhu/FashionAI/data/train/Annotations/trainminusval.csv'
    result_name = 'val_result.csv'
    # scale_search = [0.5, 0.7, 1.0, 1.3]  # [0.5, 1.0, 1.5]
    scale_search = [0.5, 0.7, 1.0, 1.3]
    boxsize = 384
# -------------------------- pytorch model------------------
    state_dict = torch.load(pytorch_model)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
# --------------------------------------------------------
    anns = []
    with open(ann_path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            anns.append(row)
    info=anns[0]
    anns = anns[1:]
    #---------------------------------------------------------
    num_imgs =100# len(anns)
    results = []
    results.append(info)

    for i in range(num_imgs):
        print('{}/{}'.format(i, num_imgs))
        ann = anns[i]
        image_path = os.path.join(img_dir, ann[0])
        oriImg = cv2.imread(image_path)
        # multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
        multiplier = scale_search
        keypoints, canvas = apply_model(oriImg, model, multiplier)
        # cv2.imwrite(os.path.join('./result', ann[0].split('/')[-1]), canvas)
        row = prepare_row(ann, keypoints)
        results.append(row)
    write_csv(result_name, results)
    evaluate(ann_path, result_name,num_imgs)

if __name__ == '__main__':
    main()
    # eval()