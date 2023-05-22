from asyncore import write
from itertools import count
from platform import release
from pyexpat import features, model
import numpy as np
import matplotlib as plt
import os
import time
from os.path import expanduser

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yaml import load

# HYPER PARAM
BATCH_SIZE = 8
MAX_DATA = 100000

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
        torch.nn.init.kaiming_normal_(self.fc4.weight)
        torch.nn.init.kaiming_normal_(self.fc5.weight)
        torch.nn.init.kaiming_normal_(self.fc6.weight)
        torch.nn.init.kaiming_normal_(self.fc7.weight)
        
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
        # self.device = 'cpu'
        self.net = Net(n_channel, n_action).to(self.device)
        # self.net.to(self.device)
        print(self.device)
        # print(self.net)
        self.optimizer = optim.Adam(
            self.net.parameters(), eps=1e-2, weight_decay=5e-4)
        self.totensor = transforms.ToTensor()
        self.transform_color = transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)
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
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.first_flag = True
        torch.backends.cudnn.benchmark = False

        #self.writer = SummaryWriter(log_dir="/home/haru/nav_ws/src/nav_cloning/runs",comment="log_1")

    def act_and_trains(self, img, dir_cmd, target_angle, times = 1):

        # <Training mode>
        self.net.train()

        if self.first_flag:
            self.x_cat = torch.tensor(
                img, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.x_cat = self.x_cat.permute(0, 3, 1, 2)
            self.c_cat = torch.tensor(
                dir_cmd, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.t_cat = torch.tensor(
                [target_angle], dtype=torch.float32, device=self.device).unsqueeze(0)
            self.first_flag = False

        # x= torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        # <To tensor img(x),cmd(c),angle(t)>
        x = torch.tensor(img, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # print('-'*50)                         
        # print(x.shape)
        # print(x.size)
        # <(Batch,H,W,Channel) -> (Batch ,Channel, H,W)>
        x = x.permute(0, 3, 1, 2)
        c = torch.tensor(dir_cmd, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        t = torch.tensor([target_angle], dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        # print('-'*50)                         
        # print(t.shape)
        # print(x)
        self.x_cat = torch.cat([self.x_cat, x], dim=0)
        self.c_cat = torch.cat([self.c_cat, c], dim=0)
        self.t_cat = torch.cat([self.t_cat, t], dim=0)

        for i in range(times):
            self.x_cat = torch.cat([self.x_cat, x], dim=0)
            self.c_cat = torch.cat([self.c_cat, c], dim=0)
            self.t_cat = torch.cat([self.t_cat, t], dim=0)

            if self.x_cat.size()[0] > MAX_DATA:
                self.x_cat = self.x_cat[1:]
                self.c_cat = self.c_cat[1:]
                self.t_cat = self.t_cat[1:]

        # <make dataset>
        #print("train x =",x.shape,x.device,"train c =" ,c.shape,c.device,"tarain t = " ,t.shape,t.device)
        dataset = TensorDataset(self.x_cat, self.c_cat, self.t_cat)
        # <dataloder>
        train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(
            'cpu').manual_seed(0), shuffle=True)
        #train_dataset = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator('cpu'),pin_memory=True,num_workers=2,shuffle=True)

        # <split dataset and to device>
        for x_train, c_train, t_train in train_dataset:
            x_train.to(self.device, non_blocking=True)
            c_train.to(self.device, non_blocking=True)
            t_train.to(self.device, non_blocking=True)
            break
        # <use data augmentation>
        # x_train = self.transform_color(x_train)
        # <learning>
        self.optimizer.zero_grad()
        y_train = self.net(x_train, c_train)
        # print("y_train=",y_train.shape,"t_train",t_train.shape)
        loss = self.criterion(y_train, t_train)
        loss.backward()
        self.optimizer.step()
        self.count += 1
        # self.writer.add_scalar("loss",loss,self.count)

        # <test>
        self.net.eval()
        action_value_training = self.net(x, c)
        # self.writer.add_scalar("angle",abs(action_value_training[0][0].item()-target_angle),self.count)
        #print("action=" ,action_value_training[0][0].item() ,"loss=" ,loss.item())

        # if self.first_flag:
        #     self.writer.add_graph(self.net,(x,c))
        # self.writer.close()
        # self.writer.flush()
        # <reset dataset>
        if self.x_cat.size()[0] > MAX_DATA:
            print("reset dataset")
            self.x_cat = self.x_cat[1:]
            self.c_cat = self.c_cat[1:]
            self.t_cat = self.t_cat[1:]

        # print(self.x_cat.size()[0])

        return action_value_training[0][0].item(), loss.item()

    def trains(self, iteration):
        # if not self.first_flag:
        dataset = TensorDataset(self.x_cat, self.c_cat, self.t_cat)
        train_iter = DataLoader(dataset, batch_size=BATCH_SIZE, generator=torch.Generator(
            'cpu').manual_seed(0), shuffle=True)
        for i in range(iteration):
            for x_train, c_train, t_train in train_iter:
                x_train.to(self.device, non_blocking=True)
                c_train.to(self.device, non_blocking=True)
                t_train.to(self.device, non_blocking=True)
                break

            # x_train = self.transform_color(x_train)
            self.optimizer.zero_grad()

            y_train = self.net(x_train, c_train)

            loss = self.criterion(y_train, t_train)
            loss.backward()

            self.optimizer.step()

            self.count += 1

            print("trains: " + str(self.count))
            # print(self.x_cat.size()[0])

            if self.x_cat.size()[0] > MAX_DATA:
                self.x_cat = self.x_cat[1:]
                self.c_cat = self.c_cat[1:]
                self.t_cat = self.t_cat[1:]

    def act(self, img, dir_cmd):
        self.net.eval()
        # <make img(x_test_ten),cmd(c_test)>
        # x_test_ten = torch.tensor(self.transform(img),dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = torch.tensor(
            img, dtype=torch.float32, device=self.device).unsqueeze(0)
        x_test_ten = x_test_ten.permute(0, 3, 1, 2)
        c_test = torch.tensor(dir_cmd, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        # print(x_test_ten.shape)                              
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