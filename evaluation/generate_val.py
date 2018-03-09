# split train to trainminusval and val (500)
import csv
import os, random

train_ann_path = '/data/xiaobing.wang/xiangyu.zhu/FashionAI/data/train/Annotations/train.csv'
output_dir = '/data/xiaobing.wang/xiangyu.zhu/FashionAI/data/train/Annotations/'
val_num = 500

info = []
anns = []
with open(train_ann_path,'rb') as f:
	reader = csv.reader(f)
	for row in reader:
		anns.append(row)
info = anns[0]
anns = anns[1:]

random.shuffle(anns)
trainminusval_anns = [info]
val_anns = [info]
trainminusval_anns = trainminusval_anns + anns[:-500]
val_anns = val_anns +anns[-500:]

with open(os.path.join(output_dir,'trainminusval.csv'), 'w') as f:
	writer = csv.writer(f)
	writer.writerows(trainminusval_anns)

with open(os.path.join(output_dir, 'val.csv'), 'w') as f:
	writer = csv.writer(f)
	writer.writerows(val_anns)


