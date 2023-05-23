# !/usr/bin/env python3
import pandas as pd
import roslib
import numpy as np

def main():
    csv_path = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/20230522_19:53:13/dataset_100_scan.csv'
    csv_path_data = roslib.packages.get_pkg_dir('olp_cloning') + '/dataset/20230522_19:53:13/dataset_100.csv'
    df_data = pd.read_csv(csv_path_data, header=None)
    

    action = df_data.iloc[:, [3, 4]]
    print(action)


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