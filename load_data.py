import numpy as np
import h5py


def load_data(data_path):

    fname1 = data_path
    hf = h5py.File(fname1, 'r')
    # for name in hf:
    #     print(name)
    perm = hf['perm'][:]
    label = hf['cond'][:]
    perm = perm
    label = label / 3
    hf.close()

    return perm, label


if __name__ == '__main__':
    perm, label = load_data('perm_15000.h5')
    print(perm.shape, label.shape)
