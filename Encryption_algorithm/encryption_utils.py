import numpy as np
import copy
import glob

def yates_shuffle(plain, key):
    p = copy.copy(plain)
    n = len(p)
    p.insert(0, 0)
    bit_len = len(bin(int(str(n), 10))) - 1
    key = '0' + key
    key_count = 1
    for i in range(n, 1, -1):
        num = int('0b' + key[key_count:key_count + bit_len], 2) + 1
        index = num % i + 1
        temp = p[i]
        p[i] = p[index]
        p[index] = temp
        key_count = key_count + 1
    del p[0]
    return p


def loadEncBit(path):
    bitstream_dic = np.load(path, allow_pickle=True)
    return bitstream_dic


def loadImageFiles(srcFiles):
    return glob.glob(srcFiles)


def loadImageSet(srcFiles):
    imageFiles = loadImageFiles(srcFiles)
    plainimages = []
    for imageName in imageFiles:
        img = cv2.imread(imageName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plainimages.append(img)
    return plainimages


