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


def df_to_hdf5(out_df, hdf5_fn, sub_grp=''):
    """
    Writes 'episode_data', and 'model_params' datasets to hdf5_fn, optionally under
    sub_grp a la sub_grp/episode_data etc.
    """
    out_df = out_df.copy()
    # choice1, stage2, choice2, reward, (N x 4 array) uint8 array
    arr = np.stack(out_df.loc[:, 'episode_data'].values)
    arr = arr_to_kastner_format(arr)
    arr = arr.astype(np.uint8)

    with h5py.File(hdf5_fn, 'a') as f:
        key = os.path.join(sub_grp, 'episode_data') if sub_grp else 'episode_data'
        f.create_dataset(key, data=arr)


    out_df.drop('episode_data', axis=1, inplace=True)
    # model_params dataset:
    # for some reason columns aren't easily findable, but they are:
    # Index(['beta_mb', 'beta_mf', 'alpha', 'beta_stage2', 'beta_stick',
    #       'Q_MB_rare_prob', 'episode_reward_rate'], dtype='object')
    key = os.path.join(sub_grp, 'model_params') if sub_grp else 'model_params'
    out_df.to_hdf(hdf5_fn, key=key, mode='a', format='table')


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

AB = out_df.loc[out_df.alpha==0.2]
C = out_df.loc[out_df.alpha != 0.2]
A_MF = AB.loc[AB.beta_mb==0.0]
A_MB = AB.loc[AB.beta_mf==0.0]

# B is AB without A_MF and A_MB
B = AB.loc[list(set(AB.index) - set(A_MF.index) - set(A_MB.index))]

AB.shape, A_MF.shape, A_MB.shape, B.shape, C.shape

# move the first row of A_MF and A_MB to B
a1, a2 = A_MF.iloc[0].to_frame().T, A_MB.iloc[0].to_frame().T
for col in a1.columns:
    a1[col] = a1[col].astype(B[col].dtype)
    a2[col] = a2[col].astype(B[col].dtype)
B = pd.concat([B, a1, a2], axis=0)

# remove from A_MF and A_MB
A_MF = A_MF.iloc[1:]
A_MB = A_MB.iloc[1:]

hdf5_fn = 'kastner_generation_09_01_2024.hdf5'

df_to_hdf5(A_MF, hdf5_fn, sub_grp='MF_500')
df_to_hdf5(A_MB, hdf5_fn, sub_grp='MB_500')
df_to_hdf5(B, hdf5_fn, sub_grp='MF_to_MB_5001')
df_to_hdf5(C, hdf5_fn, sub_grp='MF_to_MB_201_alpha_100')