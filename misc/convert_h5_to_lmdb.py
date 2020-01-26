import h5py
import lmdb
from tqdm import tqdm
import numpy as np
import six

h5_file_path = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.h5'
lmdb_file_path = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.lmdb'
NUM_IMGS_ESTIMATE = 150000
MAX_IMG_SIZE = (70, 2048)


def copy_data():
    img_size = np.zeros(MAX_IMG_SIZE, dtype=np.float32).nbytes
    in_f = h5py.File(h5_file_path, 'r')
    print('Reading h5py keys...')
    in_keys = in_f.keys()
    print('Finished reading h5py keys!')
    with lmdb.open(lmdb_file_path, subdir=False, map_size=img_size * NUM_IMGS_ESTIMATE) as out_env:
        with out_env.begin(write=True) as txn:
            print('Starting to copy!')
            for key in tqdm(in_keys):
                data = in_f[key][()]
                bytes_file = six.BytesIO()
                np.save(bytes_file, data)
                bytes_file.seek(0)
                bytes_data = bytes_file.read()
                txn.put(key.encode(), bytes_data)


# check data is indeed written successfully
def validate_data():
    with lmdb.open(lmdb_file_path, subdir=False, readonly=True, lock=False, readahead=True,
                   meminit=False) as in_env:
        with in_env.begin(write=False) as txn:
            with txn.cursor() as curs:
                print('Reading!')
                for key, value in curs:
                    print(f'KEY IS: {key}')
                    feats = np.load(six.BytesIO(value))
                    print(f'DATA SHAPE IS {feats.shape}')


def main():
    copy_data()


if __name__ == '__main__':
    main()
