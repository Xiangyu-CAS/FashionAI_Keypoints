import torch
import torch.nn as nn
import os
import sys
import math
import torchvision.models as models

def make_net_dict():

    feature = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1': [2, 2, 0]},
            {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2': [2, 2, 0]},
            {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]}, {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3': [2, 2, 0]},
            {'conv4_1': [256, 512, 3, 1, 1]}, {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_cpm': [512, 256, 3, 1, 1]}, {'conv4_4_cpm': [256, 128, 3, 1, 1]}]


    block1 = [{'conv5_1_CPM': [128, 128, 3, 1, 1]},{'conv5_2_CPM': [128, 128, 3, 1, 1]},{'conv5_3_CPM': [128, 128, 3, 1, 1]},
             {'conv5_4_CPM': [128, 512, 1, 1, 0]}]


    block2 = [{'Mconv1': [128+25, 128, 7, 1, 3]}, {'Mconv2': [128, 128, 7, 1, 3]},
              {'Mconv3': [128, 128, 7, 1, 3]},{'Mconv4': [128, 128, 7, 1, 3]},
              {'Mconv5': [128, 128, 7, 1, 3]},
              {'Mconv6': [128, 128, 1, 1, 0]}
              ]
    predict_layers_stage1 = [{'predict_L1': [512, 25, 1, 1, 0]}]

    predict_layers_stageN = [{'predict_L1': [128, 25, 1, 1, 0]}]

    net_dict = [feature,block1,predict_layers_stage1,block2,predict_layers_stageN]

    return net_dict


class CPM(nn.Module):

    def __init__(self, net_dict, batch_norm=False):

        super(CPM, self).__init__()

        self.feature = self._make_layer(net_dict[0])

        self.block = self._make_layer(net_dict[1])

        self.predict = self._make_layer(net_dict[2])

        # repeate
        self.block_stage2 = self._make_layer(net_dict[3])

        self.predict_stage2 = self._make_layer(net_dict[4])

        self.block_stage3 = self._make_layer(net_dict[3])

        self.predict_stage3 = self._make_layer(net_dict[4])

        self.block_stage4 = self._make_layer(net_dict[3])

        self.predict_stage4 = self._make_layer(net_dict[4])

        self.block_stage5 = self._make_layer(net_dict[3])

        self.predict_stage5 = self._make_layer(net_dict[4])

        self.block_stage6 = self._make_layer(net_dict[3])

        self.predict_stage6 = self._make_layer(net_dict[4])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, net_dict, batch_norm=False):
        layers = []
        length = len(net_dict)
        for i in range(length):
            one_layer = net_dict[i]
            key = one_layer.keys()[0]
            v = one_layer[key]

            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            elif 'predict' in key:
	             conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
	             layers += [conv2d]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, x):
        # define forward flow
        feature = self.feature(x)

        out_stage1 = self.block(feature)
        L1_stage1 = self.predict(out_stage1)


        concat_stage2 = torch.cat([L1_stage1,  feature], 1)
        out_stage2 = self.block_stage2(concat_stage2)
        L1_stage2 = self.predict_stage2(out_stage2)

        concat_stage3 = torch.cat([L1_stage2, feature], 1)
        out_stage3 = self.block_stage3(concat_stage3)
        L1_stage3 = self.predict_stage3(out_stage3)


        concat_stage4 = torch.cat([L1_stage3, feature], 1)
        out_stage4 = self.block_stage4(concat_stage4)
        L1_stage4 = self.predict_stage4(out_stage4)

        concat_stage5 = torch.cat([L1_stage4, feature], 1)
        out_stage5 = self.block_stage5(concat_stage5)
        L1_stage5 = self.predict_stage5(out_stage5)

        concat_stage6 = torch.cat([L1_stage5, feature], 1)
        out_stage6 = self.block_stage6(concat_stage6)
        L1_stage6 = self.predict_stage6(out_stage6)

        return L1_stage1, L1_stage2, L1_stage3, L1_stage4, L1_stage5, L1_stage6

def PoseModel(num_point, num_stages=6, batch_norm=False, pretrained=False):
    net_dict = make_net_dict()
    model = CPM(net_dict, batch_norm)

    if pretrained:
        parameter_num = 10
        if batch_norm:
            vgg19 = models.vgg19_bn(pretrained=True)
            parameter_num *= 6
        else:
            vgg19 = models.vgg19(pretrained=True)
            parameter_num *= 2

        vgg19_state_dict = vgg19.state_dict()
        vgg19_keys = vgg19_state_dict.keys()

        model_dict = model.state_dict()
        from collections import OrderedDict
        weights_load = OrderedDict()

        for i in range(parameter_num):
            weights_load[model.state_dict().keys()[i]] = vgg19_state_dict[vgg19_keys[i]]
        model_dict.update(weights_load)
        model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    print PoseModel(25, 6, batch_norm=False)
