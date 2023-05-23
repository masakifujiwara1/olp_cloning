# !/usr/bin/env python3
import pandas as pd
import roslib
import numpy as np

def main():
    csv_path = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/20230523_01:00:22/dataset_100_scan.csv'
    # df = pd.read_csv(csv_path, header=None)
    
    df = np.loadtxt(csv_path, delimiter=',', dtype='float64')

    # print(len(df))
    print(df.shape)
    # print(df[0])
    for i in range(1):
        for j in range(10):
            print(df[i][j])

    # scan_data = df.iloc[:, :]

    # print(len(df))
    # print(df[:1])
    # print(len(scan_data))
    # print(scan_data)

if __name__=='__main__':
    main()