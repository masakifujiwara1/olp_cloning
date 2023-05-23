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
BATCH_SIZE = 16
# MAX_DATA = 100000
DATASET_PATH = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/20230522_19:53:13'
# DATASET_PATH = '/home/fmasa/catkin_ws/src/create_dataset/dataset/real_environment_4hz'

TOTAL_EPOCH = 100

class MyTransforms:
    def __init__(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x = torch.from_numpy(x.astype(np.float32))
        # transforms.AugMix()
        return x

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

    # ここで取り出すデータを指定している
    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.target_list[index]
        action_v = self.action_v_list[index]
        action_w = self.action_w_list[index]
        scan_data = self.scan_list[index]

        # target = np.array(target)
        print(target, type(target))

        scan_data = torch.tensor(scan_data, dtype=torch.float32, device=self.device)
        target = torch.tensor(target, dtype=torch.float32, device=self.device)
        action = torch.tensor([float(action_v), float(action_w)], dtype=torch.float32, device=self.device)
        # action_w = torch.tensor([], dtype=torch.float32, device=self.device)

        # print(cmd, target, number)

        # print(index, data)

        return scan_data, target, action

    # この method がないと DataLoader を呼び出す際にエラーを吐かれる
    def __len__(self) -> int:
        return len(self.action_v_list)

class Net(nn.Module):
    def __init__(self, n_channel, n_out):
        super().__init__()
    # network
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm1d(64)

        self.avg_pool = nn.AvgPool1d(3)
        self.max_pool = nn.MaxPool1d(3)

        # 7630 + 3
        self.fc1 = nn.Linear(7363, 1024)
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
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.max_pool(x1)

        # x2 >> shorcut
        x3 = self.relu(self.bn2(self.conv2(x2)))
        x4 = self.bn3(self.conv3(x3))

        # x4 >> shortcut
        
        # shorcut x2
        x5 = torch.cat([x4, x2], dim=2)
        x6 = self.relu(x5)
        x7 = self.relu(self.bn4(self.conv4(x6)))
        x8 = self.bn5(self.conv5(x7))

        # shorcut x4
        x9 = torch.cat([x8, x4], dim=2)
        x10 = self.relu(x9)
        x11 = self.avg_pool(x10)
        x12 = self.flatten(x11)

        # FC
        x13 = torch.cat([x12, target], dim=1)
        x14 = self.relu(self.fc1(x13))
        x15 = self.relu(self.fc2(x14))
        x16 = self.relu(self.fc3(x15))

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
        # self.transform = transforms.Compose([MyTransforms(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.Compose([MyTransforms(), transforms.AugMix()])
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

        for epoch in range(1, TOTAL_EPOCH+1):
            # <dataloder>
            # train_dataset = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu').manual_seed(0), shuffle=True)
            dataset = MyDataset(csv_path, transforms = self.transform, device = self.device)
            train_dataset = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu').manual_seed(0), shuffle=True)

            loss_sum = 0
            total = 0

            with tqdm(total = len(train_dataset), unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch}/{TOTAL_EPOCH}]")

                for scan_train, target_train, action_train in train_dataset:
                    self.net.train()

                    self.optimizer.zero_grad()

                    y_train = self.net(scan_train, target_train)
                    loss = self.criterion(y_train, action_train)

                    loss.backward()
                    self.optimizer.step()

                    # print(self.count)
                    self.count += 1

                    total += x_train.size(0)
                    loss_sum += loss.item()
                    running_loss = loss_sum/total

                    self.writer.add_scalar("loss", loss.item(), self.count)

                    pbar.set_postfix({"loss":loss.item()})
                    pbar.update(1)      

        self.save(self.save_path)

        return True

    def act(self, img, dir_cmd):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        # x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        c_test = torch.tensor(dir_cmd, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        # print(x_test_ten.shape,x_test_ten.device,c_test.shape,c_test.device)
        # <test phase>
        action_value_test = self.net(x_test_ten, c_test)

        #print("act = " ,action_value_test.item())
        return action_value_test[0][0].item()

    def result(self):
        accuracy = self.accuracy
        return accuracy

    def save(self, save_path):
        # <model save>
        path = save_path + time.strftime("%Y%m%d_%H:%M:%S")
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