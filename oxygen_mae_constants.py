import pickle

import numpy as np

# data columns
# NOTE @^@ order matters in encoding
cat_cols = ['SST', 'CHL', 'PAR',
            'U', 'V',
            'MLD_CLM', 'SAL_CLM', 'SST_CLM',
            ]
date_loc_cols = ['YR', 'DY', 'X', 'Y']


# be careful the consistency
dt_path = "/data1/zuchuan/data/Oxygen_Ocean/pretraining_data_2024-09-27"
f_name = dt_path + '/pretraining_metadata_2024-10-09.pickle'
with open(f_name, 'rb') as fid:
    dt_meta = pickle.load(fid)


# Get the shift of encoding
def get_encode_shift(par):
    level_cum = 0
    for ii in cat_cols:
        if ii == par:
            break
        else:
            level_cum += dt_meta['LEVEL_NUM'][ii]
    return level_cum


# get column min
def get_column_min_par(par):
    return get_encode_shift(par)


# get column min
def get_column33344111_min_idx(idx):
    indices = np.array([3, 3, 3, 4, 4, 1, 1, 1])
    tmp = np.where(idx <= np.cumsum(indices)-1)[0]
    idx = indices.sum() if len(tmp) == 0 else tmp[0]
    return get_column_min_par(cat_cols[idx])

