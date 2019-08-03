import numpy as np
import pandas as pd
import psutil
import os
import warnings  
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
import gc
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

column_types_train = {
    'id': 'int32',
    'molecule_name': 'category',
    'atom_index_0': 'int16',
    'atom_index_1': 'int16',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
column_types_test = {
    'id': 'int32',
    'molecule_name': 'category',
    'atom_index_0': 'int16',
    'atom_index_1': 'int16',
    'type': 'category',
}
column_types_DM = {
     'molecule_name': 'category',
     'X': 'float16',
     'Y': 'float16',
     'Z': 'float16',
}
column_types_PE = {
     'molecule_name': 'category',
     'potential_energy': 'float32'
}
column_types_SCC = {
     'id': 'int32',
     'molecule_name': 'category',
     'atom_index_0': 'int8',
     'atom_index_1': 'int8',
     'type': 'category',
     'fc': 'float16',
     'sd': 'float16',
     'pso': 'float16',
     'dso': 'float16',
}
column_types_MC = {
     'id': 'int32',
     'atom_index': 'int16',
     'molecule_name': 'category',
     'mulliken_charge': 'float32'
}
column_types_MST = {
     'id': 'int32',
     'molecule_name': 'category',
     'atom_index': 'int16',
     'XX': 'float32',
     'YX': 'float32',
     'ZX': 'float32',
     'XY': 'float32',
     'YY': 'float32',
     'ZY': 'float32',
     'XZ': 'float32',
     'YZ': 'float32',
     'ZZ': 'float32',
}
column_types_STRUCTURES = {
    'id': 'int32',
    'molecule_name': 'category',
    'atom_index': 'int32',
    'atom': 'category',
    'X': 'float32',
    'Y': 'float32',
    'Z': 'float32',
}
train = pd.read_csv('../input/train.csv', dtype=column_types_train)
test = pd.read_csv('../input/test.csv', dtype=column_types_test)
dipole_moments = pd.read_csv('../input/dipole_moments.csv', dtype=column_types_DM)
potential_energy = pd.read_csv('../input/potential_energy.csv', dtype=column_types_PE)
scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv', dtype=column_types_SCC)
mulliken_charges = pd.read_csv('../input/mulliken_charges.csv', dtype=column_types_MC)
magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv', dtype=column_types_MST)
structures = pd.read_csv('../input/structures.csv', dtype=column_types_STRUCTURES)

train.info()

def reduce_mem(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

df_train = reduce_mem(train)
df_test = reduce_mem(test)
df_DM = reduce_mem(dipole_moments)
df_PE = reduce_mem(potential_energy)
df_MC = reduce_mem(mulliken_charges)
df_MST = reduce_mem(magnetic_shielding_tensors)
df_struct = reduce_mem(structures)
df_scc = reduce_mem(scalar_coupling_contributions)

def map_atom_info(df_1, df_2, atom_idx):
    df = pd.merge(
        df_1,
        df_2.drop_duplicates(subset=['molecule_name', 'atom_index']),
        how = 'left',
        left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
        right_on = ['molecule_name', 'atom_index']
    )
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

def show_ram_usage():
    py = psutil.Process(os.getpid())
    print('RAM usage: {} GB'.format(py.memory_info()[0]/2. ** 30))
    
show_ram_usage()


for atom_idx in [0, 1]:
    df_train = map_atom_info(df_train, df_struct, atom_idx)
    df_test = map_atom_info(df_test, df_struct, atom_idx)
    
df_train = create_master_data(df_train, df_MC)
df_train = create_master_data(df_train, df_DM)
df_train = create_master_data(df_train, df_PE)
df_train = create_master_data(df_train, df_MST)
df_train = create_master_data(df_train, df_struct)

df_train.info(memory_usage="deep")
def create_basic_features(df, atom_idx):
    df[f'c_x_{atom_idx}'] = df.groupby('molecule_name')[f'x_{atom_idx}'].transform('mean')
    df[f'c_y_{atom_idx}'] = df.groupby('molecule_name')[f'y_{atom_idx}'].transform('mean')
    df[f'c_z_{atom_idx}'] = df.groupby('molecule_name')[f'z_{atom_idx}'].transform('mean')
    df[f'min_x_{atom_idx}'] = df.groupby('molecule_name')[f'x_{atom_idx}'].transform('min')
    df[f'min_y_{atom_idx}'] = df.groupby('molecule_name')[f'y_{atom_idx}'].transform('min')
    df[f'min_z_{atom_idx}'] = df.groupby('molecule_name')[f'z_{atom_idx}'].transform('min')
    df[f'max_x_{atom_idx}'] = df.groupby('molecule_name')[f'x_{atom_idx}'].transform('max')
    df[f'max_y_{atom_idx}'] = df.groupby('molecule_name')[f'y_{atom_idx}'].transform('max')
    df[f'max_z_{atom_idx}'] = df.groupby('molecule_name')[f'z_{atom_idx}'].transform('max')
    df[f'count_x_{atom_idx}'] = df.groupby('molecule_name')[f'x_{atom_idx}'].transform('count')
    df[f'count_y_{atom_idx}'] = df.groupby('molecule_name')[f'y_{atom_idx}'].transform('count')
    df[f'count_z_{atom_idx}'] = df.groupby('molecule_name')[f'z_{atom_idx}'].transform('count')
    df[f'std_x_{atom_idx}'] = df.groupby('molecule_name')[f'x_{atom_idx}'].transform('std')
    df[f'std_y_{atom_idx}'] = df.groupby('molecule_name')[f'y_{atom_idx}'].transform('std')
    df[f'std_z_{atom_idx}'] = df.groupby('molecule_name')[f'z_{atom_idx}'].transform('std')
    df[f'distance_{atom_idx}'] = (df[f'c_x_{atom_idx}'] ** 2) + (df[f'c_y_{atom_idx}'] ** 2) + (df[f'c_z_{atom_idx}'] ** 2)
    return df

for atom_idx in [0,1]:
    df_train = create_basic_features(df_train, atom_idx)
    df_test = create_basic_features(df_test, atom_idx)
    
df_train['x_diff'] = df_train['x_1'] - df_train['x_0']
df_train['y_diff'] = df_train['y_1'] - df_train['y_0']
df_train['z_diff'] = df_train['z_1'] - df_train['z_0']

df_test['x_diff'] = df_test['x_1'] - df_test['x_0']
df_test['y_diff'] = df_test['y_1'] - df_test['y_0']
df_test['z_diff'] = df_test['z_1'] - df_test['z_0']

def create_complex_features(df, atom_idx):
    df[f'mean_dev_x_{atom_idx}'] = df[f'x_{atom_idx}'] - df[f'c_x_{atom_idx}'] # standard deviation
    df[f'mean_dev_y_{atom_idx}'] = df[f'y_{atom_idx}'] - df[f'c_y_{atom_idx}']
    df[f'mean_dev_z_{atom_idx}'] = df[f'z_{atom_idx}'] - df[f'c_z_{atom_idx}']
    df[f'var_x_{atom_idx}'] = df[f'std_x_{atom_idx}'] ** 2 # variance
    df[f'var_y_{atom_idx}'] = df[f'std_y_{atom_idx}'] ** 2
    df[f'var_z_{atom_idx}'] = df[f'std_z_{atom_idx}'] ** 2
    df[f'vec_center_{atom_idx}_x'] = (df[f'x_{atom_idx}'] - df[f'c_x_{atom_idx}']) / (df[f'distance_{atom_idx}'])
    df[f'vec_center_{atom_idx}_y'] = (df[f'y_{atom_idx}'] - df[f'c_y_{atom_idx}']) / (df[f'distance_{atom_idx}'])
    df[f'vec_center_{atom_idx}_z'] = (df[f'z_{atom_idx}'] - df[f'c_z_{atom_idx}']) / (df[f'distance_{atom_idx}'])
    return df

    
for atom_idx in [0, 1]:
    df_train = create_complex_features(df_train, atom_idx)
    df_test = create_complex_features(df_test, atom_idx)

df_train = df_train.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'atom_0', 'atom_1'], axis=1) # dont drop the id or scalar coupling constant because you will use it to generate features with featuretools
df_train = reduce_mem(df_train)

df_test = df_test.drop(['molecule_name', 'atom_index_0', 'atom_index_1', 'atom_0', 'atom_1'], axis=1) # dont drop the id because you will use it to generate features with featuretools
df_test = reduce_mem(df_test)
gc.collect()

sc = StandardScaler()
df_train = sc.fit_transform(df_train)
df_test = sc.transform(df_test)

# after learning to train them by type, scale the data before training

# from EDA, it has been noticed that different types have different distributions so models will be trained by types
# find a way to apply standard scalar on the data

d_train = lgb.Dataset(X, label=y)
d_train.save_binary('d_train.bin')
train_data = lgb.Dataset('d_train.bin')
params = {
    'learning_rate': 0.03,
    'min_child_samples': 79, # 79
    'boosting_type' : 'gbdt',
    'objective': 'regression_l2',
    'subsample': 0.9, # should be corrected to subsample
    'metric': 'mae',
    'sub_feature': 0.5,
    'num_leaves': 256, # 256
    'min_data_in_leaf': 700, # 700
    'max_depth': 20,
    'n_estimators': 5000, # 6000
    'reg_alpha': 0.1,
    'reg_lambda': 0.3,
    'colsample_bytree': 0.9,
    'verbosity': -1
}

response = {}
result = [] # put all the responses inside the result array then loop through it and add them to a dataframe
X_current = pd.DataFrame({})
for t in df_train['type'].unique():
    print(f'Training of type {t}')
    train_t = df_train.loc[df_train['type'] == t]
    test_t = df_test.loc[df_test['type'] == t]
#     print(test_t)
    pred_id_t = test_t['id'].values
    print(pred_id_t[0])
    y_t = train_t['scalar_coupling_constant']
    train_t = train_t.drop(['id', 'scalar_coupling_constant', 'type'], axis=1)
    pred_t = test_t.drop(['id', 'type'], axis=1)
    sc = StandardScaler()
    train_t_std = sc.fit_transform(train_t)
    pred_t_std = sc.transform(pred_t)
    d_train = lgb.Dataset(train_t, label=y_t)
    clf_t = lgb.train(params, d_train, 100)
    print(f'Predicting for type {t}')
    predictions_t = clf_t.predict(pred_t)
#     print(predictions_t)
    # I know it's not a good practice to use for inside for but abeg free me
    res_range = len(predictions_t)
    print(predictions_t)
    for i in range(res_range):
        result.append({'id': pred_id_t[i], 'predictions': predictions_t[i]})
