import csv


def cell_str2init(cell):
	[x_str, y_str, vis_str] = cell.split('_')
	x, y, vis = int(x_str), int(y_str), int(vis_str)
	return [x,y,vis]

# csv_file = 'val_result.csv'
csv_file = 'result_0309_23.16%.csv'
anns = []
with open(csv_file, 'rb') as f:
	reader = csv.reader(f)
	for row in reader:
		anns.append(row)
info = anns[0]
anns = anns[1:]

center_pair = [[0,3,5,7,13,15,17],
               [1,4,6,8,14,16,18]]

near_pair = [[ 9,11,20,22],
             [10,12,21,23]]

for i in range(len(anns)):
	ann = anns[i]
	center_x = 0
	count = 0
	for j in range(2, len(ann)):
		cell = ann[j]
		[x, y, vis] = cell_str2init(cell)
		center_x += x
		count += 1
	center_x = int(1.0*center_x/count)
	for j in range(len(near_pair[0])):
		indexA = near_pair[0][j] + 2
		indexB = near_pair[1][j] + 2
		[x_str_A, y_str_A, vis_str_A] = ann[indexA].split('_')
		x_A, y_A, vis_A = int(x_str_A), int(y_str_A), int(vis_str_A)

		[x_str_B, y_str_B, vis_str_B] = ann[indexB].split('_')
		x_B, y_B, vis_B = int(x_str_B), int(y_str_B), int(vis_str_B)

		if (vis_A == -1 and vis_B == -1) or (vis_A == 1 and vis_B == 1):
			continue
		if (vis_A == 1 and vis_B == -1):
			vis_B = 1
			x_B = x_A
			y_B = y_A
		elif (vis_B == 1 and vis_A == -1):
			vis_A = 1
			x_A = x_B
			y_A = y_B
		anns[i][indexA] = str(x_A) + '_' + str(y_A) + '_' + str(vis_A)
		anns[i][indexB] = str(x_B) + '_' + str(y_B) + '_' + str(vis_B)

	for j in range(len(center_pair[0])):
		indexA = center_pair[0][j] + 2
		indexB = center_pair[1][j] + 2
		[x_str_A, y_str_A, vis_str_A] = ann[indexA].split('_')
		x_A, y_A, vis_A = int(x_str_A), int(y_str_A), int(vis_str_A)

		[x_str_B, y_str_B, vis_str_B] = ann[indexB].split('_')
		x_B, y_B, vis_B = int(x_str_B), int(y_str_B), int(vis_str_B)

		if (vis_A == -1 and vis_B== -1) or (vis_A == 1 and vis_B == 1):
			continue
		if (vis_A == 1 and vis_B == -1):
			vis_B = 1
			x_B = abs(2*center_x - x_A)
			y_B = y_A
		elif (vis_B == 1 and vis_A == -1):
			vis_A = 1
			x_A = abs(2*center_x - x_B)
			y_A = y_B
		anns[i][indexA] = str(x_A) + '_' + str(y_A) + '_' + str(vis_A)
		anns[i][indexB] = str(x_B) + '_' + str(y_B) + '_' + str(vis_B)

results = [info]
results = results + anns

with open('modify.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerows(results)


'''
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