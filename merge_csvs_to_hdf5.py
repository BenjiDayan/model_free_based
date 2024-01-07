import os
import h5py
import pandas as pd
import numpy as np

def arr_to_kastner_format(arr):
    """
    arr: N x 200 x 4 arr of uint8, choice1, stage2, choice2, reward
    Converts to kastner format, which is:
    400 x 2 x N
    i.e. 2 rows per trial, each of 2 cols: 2nd col is reward 0/1
    1st col is 0/1 for choice1, and then 2/3 for stage2 that is end up in
    """
    arr = arr.copy()
    # N x 200
    rews = arr[:, :, 3]
    # N x 400
    arr2 = arr[:, :, 0:2].reshape(arr.shape[0], -1)
    arr2.shape
    arr2[:, 1::2] = arr2[:, 1::2] + 2  # 0/1 stage 2 -> 2/3
    # N x 400
    rews2 = np.zeros(arr2.shape)
    rews2[:, 1::2] = rews
    # N x 400 x 2
    arr3 = np.stack((arr2, rews2), axis=-1)
    return arr3.transpose(1, 2, 0)  # 400 x 2 x N


path = './kastner_generation_05_01_2024'

# Get all the files in the folder
files = os.listdir(path)
files = [os.path.join(path, file) for file in files]

out_dfs = []
for file in files:
    with open(file, 'rb') as f:
        df = pd.read_pickle(f)
        out_dfs.append(df)

out_df = pd.concat(out_dfs, axis=0)
# give it a new index
out_df.reset_index(inplace=True, drop=True)


# choice1, stage2, choice2, reward, (N x 4 array) uint8 array
arr = np.stack(out_df.loc[:, 'episode_data'].values)
arr = arr_to_kastner_format(arr)
arr = arr.astype(np.uint8)

with h5py.File('kastner_generation_05_01_2024.hdf5', 'w') as f:
    f.create_dataset('episode_data', data=arr)


out_df.drop('episode_data', axis=1, inplace=True)
# model_params dataset:
# for some reason columns aren't easily findable, but they are:
# Index(['beta_mb', 'beta_mf', 'alpha', 'beta_stage2', 'beta_stick',
#       'Q_MB_rare_prob', 'episode_reward_rate'], dtype='object')
out_df.to_hdf('kastner_generation_05_01_2024.hdf5', key='model_params', mode='a', format='table')
