# Heatmap
Heatmap approach for Fashion AI keypoint

Preprocessing
1. split train to trainminusval and val

train
1. cd ./experiments/CPM/
2. python ./train_net

eval
1. cd ./evaluation
2. python ./csv_evaluation.py or python ./submit.py

experiments
1.  CPM  -> 23% on leaderboard
2.  CPM_ResNet 17.9% on valset
3.  CPM_FPN + data_aug -> 8% on valset, 12% on leaderboard
