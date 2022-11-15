import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import lib.biosignal.signal_process.signalgo as signalgo
from .lib import signalgo as signalgo
sns.set(rc={'figure.figsize': (25, 15),
            'lines.linewidth': 2})


def sliceValues(np_values, upper_bound, lower_bound):
    np_mean = np.mean(np_values, axis=0)
    iszero = np.where(np_mean == 0)
    print(iszero)
    np_values = np.delete(np_values, iszero, axis=1)
    print('new np_values: ', np_values.shape)

    np_mean = np.mean(np_values, axis=0)
    np_argsort = np.argsort(np_mean)
    upper_index = np_argsort[:upper_bound]

    lower_index = np_argsort[-lower_bound:]
    delete_index = np.append(upper_index, lower_index)
    print(f'lower_index: {lower_index}, upper_index: {upper_index}')
    print(f'delete_index: {delete_index}')

    sliced_values = np.delete(np_values, delete_index, axis=1)

    print(f'sliced values: {sliced_values.shape}')

    return sliced_values


def plotEachFSR(path_source_list, folder_name_list,
                start_point=0, end_point=2000,
                upper_bound=0, lower_bound=2,
                isPlot=False):

    for path_source in path_source_list:
        for i, folder_name in enumerate(folder_name_list):
            print(f'i: {i}')
            if i >= 1:
                break
            current_data = os.path.join(path_source, folder_name)
            path_data_fsr = f'{current_data}/fsr.csv'
            print(path_data_fsr)

            if not os.path.exists(path_data_fsr):
                continue
            else:
                df_fsr = pd.read_csv(path_data_fsr)

            df_fsr['sensor'] = df_fsr['sensor'].apply(
                lambda x: np.array(x.split(' '), dtype=float))

            _values = np.stack(df_fsr.sensor.values, axis=0)
            _values = _values[start_point:end_point, :]
            print(f'values: {_values.shape}')

            sliced_values = sliceValues(_values, upper_bound, lower_bound)

            if isPlot:
                sns.lineplot(data=_values, palette='bright').set(
                    title=path_data_fsr)
                plt.show()
                sns.lineplot(data=sliced_values, palette='bright').set(
                    title=path_data_fsr)
                plt.show()

            # display(df_fsr)


def convert_df_withoutKey(df_fsr):
    # __min = df_key.timestamp.min()
    # __max = df_key.timestamp.max()
    # condition = (df_fsr.timestamp >= __min) & (df_fsr.timestamp <= __max)
    # df_fsr = df_fsr[condition]

    df_fsr.index = df_fsr.timestamp
    # df_fsr = df_fsr.set_index('timestamp')
    # df_key = df_key.set_index('timestamp')
    # df_key_re = df_key.reindex(df_fsr.index, method='ffill')  # nearest

    # df_fsr['key'] = df_key_re['key']
    # df_fsr = df_fsr.dropna(subset=['key']).reset_index(drop=True)
    print('amp: ', df_fsr.amp_sensor.unique())
    print('sens: ', df_fsr.sens_sensor.unique())
    # display(df_fsr.head())

    df = df_fsr.copy()
    df['time_sec'] = (df.timestamp - df.timestamp.min()) / 1000
    df['$sensor'] = df['sensor']
    df['sensor'] = df['sensor'].apply(
        lambda x: np.array(x.split(' '), dtype=float))

    # df = df[df['sensor'].sum(axis=0) > 0]
    _values = np.stack(df.sensor.values, axis=0)

    df['sensor_sum'] = df['sensor'].apply(lambda x: np.sum(x, dtype=int))
    # print(df.shape)
    df = df[df['sensor_sum'] > 0]
    print(df.shape)

    return df


def convert_df(df_fsr, df_key):
    __min = df_key.timestamp.min()
    __max = df_key.timestamp.max()
    condition = (df_fsr.timestamp >= __min) & (df_fsr.timestamp <= __max)
    df_fsr = df_fsr[condition]

    df_fsr.index = df_fsr.timestamp
    # df_fsr = df_fsr.set_index('timestamp')
    df_key = df_key.set_index('timestamp')
    df_key_re = df_key.reindex(df_fsr.index, method='ffill')  # nearest

    df_fsr['key'] = df_key_re['key']
    df_fsr = df_fsr.dropna(subset=['key']).reset_index(drop=True)
    # display(df_fsr.head())
    print(df_fsr.amp_sensor.unique())
    print(df_fsr.sens_sensor.unique())

    df = df_fsr.copy()
    df['time_sec'] = (df.timestamp - df.timestamp.min()) / 1000
    df['$sensor'] = df['sensor']
    df['sensor'] = df['sensor'].apply(
        lambda x: np.array(x.split(' '), dtype=float))

    return df


def calculate_statistics(df,
                         ratio=0.5464516840547095, without_ratio=False,
                         start_point=None, end_point=None):
    _values = np.stack(df.sensor.values, axis=0)

    if start_point is not None and end_point is not None:
        _values = _values[start_point:end_point]

    # get non-zero mean
    __mean_nz = np.apply_along_axis(lambda x: np.mean(x[x != 0]), 1, _values)
    __min_nz = np.apply_along_axis(lambda x: np.min(x[x != 0]), 1, _values)
    __max_nz = np.apply_along_axis(lambda x: np.max(x[x != 0]), 1, _values)
    __std_nz = np.apply_along_axis(lambda x: np.std(x[x != 0]), 1, _values)

    # print(len(__mean_nz))
    # print(df.shape)

    if start_point is not None and end_point is not None:
        timestamp = df['timestamp'].values[start_point:end_point]
    else:
        timestamp = df['timestamp'].values

    df = pd.DataFrame({
        'timestamp': timestamp,
        'mean': __mean_nz,
        'min': __min_nz,
        'max': __max_nz,
        'std': __std_nz
    })
    # display(df)

    __mean_sub_ema = signalgo.subtract_ema(__mean_nz, 0.01)
    __mean_ema = signalgo.ema(__mean_sub_ema, 0.05)
    # peaks = signalgo.get_peaks(__mean_ema)

    plt.figure(figsize=(16, 14))
    plt.subplot(4, 1, 1)
    plt.plot(__mean_nz, color='red')
    plt.title('temperature XX degree of celcius - non-zero mean')
    plt.subplot(4, 1, 2)
    # plt.plot(__mean_sub_ema)
    plt.plot(__mean_ema, color='red')
    plt.title('(filtered by ema)')

    if without_ratio:
        __mean_rt = __mean_nz
    else:
        __mean_rt = __mean_nz * ratio  # RATIO
    __mean_rt_sub_ema = signalgo.subtract_ema(__mean_rt, 0.01)
    __mean_rt_ema = signalgo.ema(__mean_rt_sub_ema, 0.05)

    if not without_ratio:
        plt.subplot(4, 1, 3)
        plt.plot(__mean_rt, color='blue')
        plt.title('reduced by result of regression - non-zero mean')
        plt.subplot(4, 1, 4)
        plt.plot(__mean_rt_ema, color='blue')
        plt.title('(filtered by ema)')
