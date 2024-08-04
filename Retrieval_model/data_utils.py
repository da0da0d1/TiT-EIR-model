import os
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(type='Corel10-a'):
    folder_path = '/root/autodl-tmp/EViT-main/data/features/difffeature_matrix'
    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=lambda x: int(x.split('.')[0]))
    all_arrays = []
    # 遍历文件夹中的.npy文件
    for file_name in file_list:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            array = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays.append(array)
    LBP_features = np.concatenate(all_arrays, axis=0)

    folder_path2 = '/root/autodl-tmp/EViT-main/data/bianyuan_features/difffeature_matrix'
    file_list2 = os.listdir(folder_path2)
    file_list2 = sorted(file_list2, key=lambda x: int(x.split('.')[0]))
    all_arrays2 = []
    # 遍历文件夹中的.npy文件
    for file_name in file_list2:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path2, file_name)
            array2 = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays2.append(array2)
    LBP_features_b = np.concatenate(all_arrays2, axis=0)

    folder_path3 = '/root/autodl-tmp/EViT-main/data/features/huffman_feature'
    file_list3 = os.listdir(folder_path3)
    file_list3 = sorted(file_list3, key=lambda x: int(x.split('.')[0]))
    all_arrays3 = []
    # 遍历文件夹中的.npy文件
    for file_name in file_list3:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path3, file_name)
            array3 = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays3.append(array3)
    col_huffman = np.stack(all_arrays3, axis=0)


    folder_path4 = '/root/autodl-tmp/EViT-main/data/bianyuan_features/huffman_feature'
    file_list4 = os.listdir(folder_path4)
    file_list4 = sorted(file_list4, key=lambda x: int(x.split('.')[0]))
    all_arrays4 = []
    # 遍历文件夹中的.npy文件
    for file_name in file_list4:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path4, file_name)
            array4 = np.load(file_path, allow_pickle=True).astype(np.float32)
            all_arrays4.append(array4)
    col_huffman_b = np.stack(all_arrays4, axis=0)

    label = []
    for i in range(100):
        for j in range(100):
            label.append(i)
    if type == 'Corel10K-a':
        train_data = LBP_features[:7000, :, :]
        train_data_b = LBP_features_b[:7000, :, :]
        train_label = label[:7000]

        test_data = LBP_features[7000:, :, :]
        test_data_b = LBP_features_b[7000:, :, :]
        test_label = label[7000:]

        train_huffman_feature = col_huffman[:7000, :]
        train_huffman_feature_b = col_huffman_b[:7000, :]
        test_huffman_feature = col_huffman[7000:, :]
        test_huffman_feature_b = col_huffman_b[7000:, :]

    else:
        # Corel10K-b
        train_data, test_data, train_label, test_label = train_test_split(LBP_features, label, test_size=0.3, stratify=label, random_state=20240)
        train_data_b, test_data_b, train_label, test_label = train_test_split(LBP_features_b, label, test_size=0.3, stratify=label, random_state=20240)

        train_huffman_feature, test_huffman_feature, train_label, test_label = train_test_split(col_huffman, label, test_size=0.3, stratify=label, random_state=20240)
        train_huffman_feature_b, test_huffman_feature_b, train_label, test_label = train_test_split(col_huffman_b, label, test_size=0.3, stratify=label, random_state=20240)

    return train_data,train_data_b, test_data,test_data_b, train_huffman_feature,train_huffman_feature_b,test_huffman_feature,test_huffman_feature_b, train_label, test_label


