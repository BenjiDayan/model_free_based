import os
import h5py
import pandas as pd
import numpy as np

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
with h5py.File('kastner_generation_05_01_2024.hdf5', 'a') as f:
    f.create_dataset('episode_data', data=arr)


out_df.drop('episode_data', axis=1, inplace=True)
# model_params dataset:
# for some reason columns aren't easily findable, but they are:
# Index(['beta_mb', 'beta_mf', 'alpha', 'beta_stage2', 'beta_stick',
#       'Q_MB_rare_prob', 'episode_reward_rate'], dtype='object')
out_df.to_hdf('kastner_generation_05_01_2024.hdf5', key='model_params', mode='w', format='table')


# for some reason columns aren't easily findable, but they are:
# Index(['beta_mb', 'beta_mf', 'alpha', 'beta_stage2', 'beta_stick',
#       'Q_MB_rare_prob', 'episode_reward_rate'], dtype='object')


def arr_to_kastner_format(arr):
    """
    arr: N x 200 x 4 arr of uint8, choice1, stage2, choice2, reward
    Converts to kastner format, which is:
    
    400 x 2 x N
    i.e. 2 rows per trial, each of 2 cols: 2nd col is reward 0/1
    1st col is 
    """
    pass