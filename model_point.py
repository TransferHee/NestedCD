from __future__ import print_function
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self, num_points=2048):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, args):
        parameter = args['parameter']
        num_points = parameter['num_points']

        super(PointNetfeat, self).__init__()
        self.args = args
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, args['argument']['feat_dims'], 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(args['argument']['feat_dims'])
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        local = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(local)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)

        glf = x.view(-1, self.args['argument']['feat_dims'], 1)

        return glf, local


class PointDecoder(nn.Module):
    def __init__(self, args, seg_class=0):
        super(PointDecoder, self).__init__()
        parameter = args['parameter']
        num_points = parameter['num_points']

        self.num_points = num_points
        self.seg_class = seg_class

        self.conv1 = torch.nn.Conv1d(args['argument']['feat_dims'] + args['argument']['local_dims'],
                                     512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 3 + seg_class, 1)

    def forward(self, glf, local):
        glf = glf.repeat(1, 1, self.num_points)
        x = torch.cat([glf, local], dim=1)

        x = F.leaky_relu_(self.conv1(x), 0.2)
        x = F.leaky_relu_(self.conv2(x), 0.2)
        x = F.leaky_relu_(self.conv3(x), 0.2)
        x = self.conv4(x)

        if self.seg_class == 0:
            return x
        else:
            recons = x[:, :3, :]
            seg = x[:, 3:, :].transpose(2, 1)
            seg = F.log_softmax(seg, dim=-1)

            return recons, seg


class PointGridDecoder(nn.Module):
    def __init__(self, args, seg_class=0):
        super(PointGridDecoder, self).__init__()
        parameter = args['parameter']
        argument = args['argument']

        self.meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        num_points = parameter['num_points']

        self.num_points = num_points
        self.k = 3
        self.seg_class = seg_class

        self.decoder1 = nn.Sequential(
            nn.Conv1d(argument['feat_dims'] + 2, 512, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 256, 1),
            nn.LeakyReLU(0.2),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 3 + seg_class, 1)
        )

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))

        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, x):
        x = x.repeat(1, 1, self.num_points)  # (batch_size, feat_dims, num_points)
        points = self.build_grid(x.shape[0]).transpose(1, 2)
        # points shape (batch_size, 2, num_points) or (batch_size, 3, num_points)

        if x.get_device() != -1:
            points = points.cuda(x.get_device())
        cat1 = torch.cat((x, points), dim=1)
        # cat 1 (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)

        x = self.decoder1(cat1)
        x = self.decoder2(x)

        if self.seg_class == 0:
            return x
        else:
            recons = x[:, :3, :]
            seg = x[:, 3:, :].transpose(2, 1)
            seg = F.log_softmax(seg, dim=-1)

            return recons, seg


class BiasDecoder(nn.Module):
    def __init__(self, args, seg_class=0):
        super(BiasDecoder, self).__init__()
        parameter = args['parameter']
        num_points = parameter['num_points']

        self.num_points = num_points
        self.k = 3
        self.seg_class = seg_class

        self.decoder1 = nn.Sequential(
            nn.Conv1d(args['argument']['feat_dims'], 512, 1, bias=False),
            BiasLayer(512, num_points),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 1, bias=False),
            BiasLayer(512, num_points),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 1, bias=False),
            BiasLayer(512, num_points),
            nn.LeakyReLU(0.2),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            BiasLayer(512, num_points),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 3 + seg_class, 1, bias=False),
            BiasLayer(3 + seg_class, num_points),
        )

    def forward(self, glf):
        glf = glf.repeat(1, 1, self.num_points)

        x = self.decoder1(glf)
        x = self.decoder2(x)

        if self.seg_class == 0:
            return x
        else:
            recons = x[:, :3, :]
            seg = x[:, 3:, :].transpose(2, 1)
            seg = F.log_softmax(seg, dim=-1)

            return recons, seg


class FCDecoder(nn.Module):
    def __init__(self, args, seg_class=0):
        super(FCDecoder, self).__init__()
        parameter = args['parameter']
        num_points = parameter['num_points']

        self.num_points = num_points
        self.k = 3
        self.seg_class = seg_class

        self.fc1 = FCLayer(args['argument']['feat_dims'], 32)
        self.fc2 = FCLayer(32, 8)
        self.fc3 = FCLayer(8, 3 + seg_class)

    def forward(self, glf):
        glf = glf.repeat(1, 1, self.num_points)
        x = self.fc1(glf)
        x = self.fc2(x)
        x = self.fc3(x)

        if self.seg_class == 0:
            return x
        else:
            recons = x[:, :3, :]
            seg = x[:, 3:, :].transpose(2, 1)
            seg = F.log_softmax(seg, dim=-1)

            return recons, seg


class FCLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCLayer, self).__init__()

        self.layer_arr = nn.ModuleList([nn.Conv1d(in_channels, out_channels, 1)
                                        for _ in range(2048)])

    def forward(self, x):
        output_arr = []

        for i in range(2048):
            temp = x[:, :, i]
            temp = torch.unsqueeze(temp, 2)
            t = self.layer_arr[i](temp)
            output_arr.append(t)

        output = torch.cat(output_arr, dim=2)

        return output


class BiasLayer(nn.Module):
    def __init__(self, channel, num_points):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.ones(channel, num_points))
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, x):
        x = x + self.bias

        return x


if __name__ == '__main__':
    zz = torch.ones((2, 3, 5))

    # bl = BiasLayer(3, 5)
    # print(bl(zz))
