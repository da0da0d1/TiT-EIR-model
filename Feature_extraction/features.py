import glob
from global_feature import global_feature
from local_feature import local_feature_all_component
from Encryption_algorithm.encryption_utils import loadEncBit
import numpy as np
import os
import multiprocessing as mul
import datetime

def main(path):
    # if not os.path.exists("../data/features"):
    #     os.mkdir("../data/features")
    # if not os.path.exists("../data/features/difffeature_matrix"):
    #     os.mkdir("../data/features/difffeature_matrix")
    # if not os.path.exists("../data/features/huffman_feature"):
    #     os.mkdir("../data/features/huffman_feature")

    bitstream = loadEncBit(path).item()  # load encrypted bitstream

    local_feature = local_feature_all_component(bitstream['dccofY'], bitstream['accofY'], bitstream['dccofCb'],
                                                  bitstream['accofCb'], bitstream['dccofCr'], bitstream['accofCr'], bitstream['size'])
    np.save("../data/features/difffeature_matrix/" + path.split('/')[-1].split('.')[0] + ".npy", local_feature)

    global_Huffman_feature = global_feature(bitstream['dccofY'], bitstream['accofY'], bitstream['dccofCb'],
                                            bitstream['accofCb'], bitstream['dccofCr'], bitstream['accofCr'])
    np.save("../data/features/huffman_feature/" + path.split('/')[-1].split('.')[0] + ".npy", global_Huffman_feature)

    print(path + ' ' + 'process success!')


if __name__ == '__main__':
    bit_path = '../data/JPEGBitStream/*.npy'
    bitFiles = glob.glob(bit_path)
    now_time = datetime.datetime.now()
    print(now_time)
    pool = mul.Pool(10)
    rel = pool.map(main, bitFiles)
    now_time = datetime.datetime.now()
    print(now_time)
    print('finish')
