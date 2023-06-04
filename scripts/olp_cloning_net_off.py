from itertools import count
from platform import release
from pyexpat import features, model
import numpy as np
import os
import time
import pandas as pd
import roslib
from typing import List, Tuple
import ast
from PIL import Image

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import load
from tqdm import tqdm

# HYPER PARAM
BATCH_SIZE = 512

# DATASET_NAME = '100train_100eval2'
DATASET_NAME = '600train_400eval2_fix'
DATASET_PATH = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/' + DATASET_NAME

TOTAL_EPOCH = 100
LOSS_RATIO = 0.1 # this param is weight of velocity

class MyDataset(Dataset):
    def __init__(self, csv_path, transforms, device) -> None:
        super().__init__()
        self.transforms = transforms
        self.device = device
        # csv process
        df_scan = np.loadtxt(csv_path + '/dataset_100_scan.csv', delimiter=',', dtype='float64')
        df = pd.read_csv(csv_path + '/dataset_100.csv')
        self.action_v_list = df.iloc[:, 3]
        self.action_w_list = df.iloc[:, 4]
        self.target_list = df.iloc[:, 5]
        self.scan_list = df_scan

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.target_list[index]
        action_v = self.action_v_list[index]
        action_w = self.action_w_list[index]
        scan_data = self.scan_list[index]

        # target = np.array(target)
       
        target = target.strip('[]')
        target = target.split()
        target = [float(element) for element in target]

        t_cos = np.cos(target[1])
        t_sin = np.sin(target[1])
        target[1] = t_cos
        target[2] = t_sin
        target = np.array(target)

        # TARGET ELEMENT
        # [0]: disatance [1]:cos [2]:sin

        # print(target, type(target))

        scan_data = torch.tensor(scan_data, dtype=torch.float32, device=self.device).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        # target = torch.tensor([target[0], target[1]], dtype=torch.float32, device=self.device)
        action = torch.tensor([float(action_v), float(action_w)], dtype=torch.float32, device=self.device)

        return scan_data, target, action

    def __len__(self) -> int:
        return len(self.action_v_list)

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # network
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.bn5 = nn.BatchNorm1d(64)

        self.avg_pool = nn.AvgPool1d(3)
        self.max_pool = nn.MaxPool1d(3)

        # 2496 + 3
        self.fc1 = nn.Linear(2499, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

        self.relu = nn.ReLU(inplace=True)

    # Weight set
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)
        torch.nn.init.kaiming_normal_(self.conv3.weight)
        torch.nn.init.kaiming_normal_(self.conv4.weight)
        torch.nn.init.kaiming_normal_(self.conv5.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        # torch.nn.init.kaiming_normal_(self.fc7.weight)
        
        self.flatten = nn.Flatten()

    def forward(self, x, target):
        # CNN
        # print(x.shape)
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.max_pool(x1)

        # x2 >> shorcut
        x3 = self.relu(self.bn2(self.conv2(x2)))
        x4 = self.bn3(self.conv3(x3))

        # x4 >> shortcut
        
        # shorcut x2
        # x5 = torch.cat([x4, x2], dim=2)
        x5 = x4 + x2
        x6 = self.relu(x5)
        x7 = self.relu(self.bn4(self.conv4(x6)))
        x8 = self.bn5(self.conv5(x7))

        # shorcut x4
        x9 = x8 + x4
        # x9 = torch.cat([x8, x4], dim=2)
        x10 = self.relu(x9)
        x11 = self.avg_pool(x10)
        x12 = self.flatten(x11)

        # FC
        # print(x12.shape, target.shape)
        x13 = torch.cat([x12, target], dim=1)
        x14 = self.relu(self.fc1(x13))
        x15 = self.relu(self.fc2(x14))
        x16 = self.fc3(x15)

        return x16

class deep_learning:
    def __init__(self, n_channel=3, n_action=1):
        # tensor device choiece
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        self.net = Net(n_channel, n_action).to(self.device)
        print(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), eps=1e-2, weight_decay=5e-4, lr=0.001)
        pg = self.optimizer.param_groups[0]
        print(pg['lr'])
        self.totensor = transforms.ToTensor()
        self.n_action = n_action
        self.count = 0
        self.accuracy = 0
        self.results_train = {}
        self.results_train['loss'], self.results_train['accuracy'] = [], []
        self.loss_list = []
        self.acc_list = []
        self.dir_list = []
        self.datas = []
        self.target_angles = []
        self.criterion = nn.MSELoss()
        self.transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.first_flag = True
        self.save_path = roslib.packages.get_pkg_dir('olp_cloning') + '/data/'
        torch.backends.cudnn.benchmark = False
        # log_path = "./log"
        # self.writer = SummaryWriter(log_dir="log")
        # self.cmd_img = ['center']
        # self.cmd_img = ['center', 'left', 'right']

        self.writer = SummaryWriter(log_dir='/home/fmasa/catkin_ws/src/olp_cloning/runs')

    def offline_training(self):

        # <Training mode>
        # self.net.train()

        # <make dataset>
        csv_path = DATASET_PATH  

        dataset = MyDataset(csv_path, transforms = self.transform, device = self.device)

        for epoch in range(1, TOTAL_EPOCH+1):
            # <dataloder>
            # train_dataset = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu').manual_seed(0), shuffle=True)
            # dataset = MyDataset(csv_path, transforms = self.transform, device = self.device)
            train_dataset = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu').manual_seed(0), shuffle=True)

            loss_sum = 0
            total = 0

            with tqdm(total = len(train_dataset), unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch}/{TOTAL_EPOCH}]")

                for scan_train, target_train, action_train in train_dataset:
                    self.net.train()

                    # print(scan_train.shape, target_train.shape, action_train.shape)

                    self.optimizer.zero_grad()

                    y_train = self.net(scan_train, target_train)
                    loss = self.criterion(y_train, action_train)


                    # loss = loss1 + loss2
                    # loss1 = self.criterion(y_train[0], action_train[0])
                    # loss2 = self.criterion(y_train[1], action_train[1])

                    # loss_ratio = LOSS_RATIO
                    # loss = loss_ratio * loss1 + (1 - loss_ratio) * loss2
                    

                    loss.backward()
                    self.optimizer.step()

                    # print(self.count)
                    self.count += 1

                    total += scan_train.size(0)
                    loss_sum += loss.item()
                    running_loss = loss_sum/total

                    self.writer.add_scalar("loss", loss.item(), self.count)

                    pbar.set_postfix({"loss":loss.item()})
                    pbar.update(1)      

        self.save(self.save_path)

        return True

    def act(self, scan_data, target):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        scan_test = torch.tensor(scan_data, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        # print(scan_test.shape)
        target_test = np.array(target)
        target_test = torch.tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        # <test phase>
        action_value_test = self.net(scan_test, target_test)

        # print(action_value_test)
        #print("act = " ,action_value_test.item())
        return action_value_test[0][0].item(), action_value_test[0][1].item()

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S") + '_' + str(DATASET_NAME) + '_' + str(LOSS_RATIO) + 'ratio_' + str(TOTAL_EPOCH) + 'ep' + '_' + str(BATCH_SIZE) + 'ba'
        os.makedirs(path)
        torch.save(self.net.state_dict(), path + '/model_gpu.pt')
        print("save_model")

    def load(self, load_path):
        # <model load>
        self.net.load_state_dict(torch.load(load_path))
        print("load_model =", load_path)


if __name__ == '__main__':
    dl = deep_learning()
    dl.offline_training()