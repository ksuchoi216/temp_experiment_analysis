import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import lib.biosignal.signal_process.signalgo as signalgo
from .lib_external import signalgo as signalgo


def get_data_path(cfg, degree=80, FolderName=None):
    degrees = [-30, 25, 80]

    if degree not in degrees:
        return print('Wrong degree!!!')

    if degree < 0:
        degree_keyword = 'm'+str(-degree)
    else:
        degree_keyword = str(degree)

    config_keyword = 'path_21v_'+degree_keyword

    PATH_SOURCE = cfg['PATH'][config_keyword]

    if FolderName is None:
        PATH_DATA_FSR = PATH_SOURCE
    else:
        CURRENT_DATA = os.path.join(PATH_SOURCE, FolderName)
        PATH_DATA_FSR = f'{CURRENT_DATA}/fsr.csv'

    print(PATH_DATA_FSR)

    return PATH_DATA_FSR


sns.set(rc={'figure.figsize': (20, 10),
            'lines.linewidth': 2})


# def sliceValues(np_values, upper_bound, lower_bound):
#     np_mean = np.mean(np_values, axis=0)
#     iszero = np.where(np_mean == 0)
#     print(iszero)
#     np_values = np.delete(np_values, iszero, axis=1)
#     # print('new np_values: ', np_values.shape)

#     np_mean = np.mean(np_values, axis=0)
#     print('FSR mean: ', np_mean)
#     # ascending sort
#     np_argsort = np.argsort(np_mean)
#     print('ascending sort FSR: \n', np_argsort.reshape(-1, 5))
#     upper_index = np_argsort[-upper_bound:]
#     lower_index = np_argsort[:lower_bound]
#     delete_index = np.append(upper_index, lower_index)
#     print(f'lower_index: {lower_index}, upper_index: {upper_index}')
#     # print(f'delete_index: {delete_index}')

#     sliced_values = np.delete(np_values, delete_index, axis=1)

#     print(f'sliced values: {sliced_values.shape}')

#     print(f'sliced values mean: {np.mean(sliced_values)}')

#     return sliced_values


def sliceValuesWithSelectedFSR(np_values, selected_FSR):
    sliced_values = np_values[:, selected_FSR]

    print(f'sliced values: {sliced_values.shape}')
    print(f'sliced values mean: {np.mean(sliced_values)}')

    return sliced_values


def plotEachFSR(path_source_list,
                folder_name_list,
                data_range_list=None,
                upper_bound=None, lower_bound=None,
                selected_FSR=None,
                isPlot=False):

    for path_source in path_source_list:
        for i, folder_name in enumerate(folder_name_list):
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

            if data_range_list is not None:
                data_start_point, data_end_point = data_range_list[i]
                _values = _values[data_start_point:data_end_point, :]

            print(f'values: {_values.shape}')

            np_mean = np.mean(_values, axis=0)
            print('np mean:\n ', np.sort(np_mean))
            np_argsort = np.argsort(np_mean)
            print('ascending sort FSR:\n', np_argsort)

            if upper_bound is not None and lower_bound is not None:
                sliced_values = sliceValues(_values, upper_bound, lower_bound)
            else:
                sliced_values = sliceValuesWithSelectedFSR(
                    _values, selected_FSR
                )

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


def sliceValuesWithoutSelectedFSR(np_values, deleted_FSR_list: list):
    np_mean = np.mean(np_values, axis=0)
    # iszero = np.where(np_mean == 0)
    # np_values = np.delete(np_values, iszero, axis=1)

    sliced_values = np.delete(np_values, deleted_FSR_list, axis=1)

    print(f'sliced values: {sliced_values.shape}')

    print(f'sliced values mean: {np.mean(sliced_values)}')

    return sliced_values


def calculate_statistics(df,
                         ratio=0.5464516840547095, without_ratio=False,
                         deleted_FSR_list=None,
                         selected_FSR_list=None,
                         start_point=None, end_point=None):

    _values = np.stack(df.sensor.values, axis=0)

    if start_point is not None and end_point is not None:
        _values = _values[start_point:end_point, :]

    print('org values dimension: ', _values.shape)

    if deleted_FSR_list is not None:
        print('deleted FSR list: ', deleted_FSR_list)
        _values = np.delete(_values, deleted_FSR_list, axis=1)
        print('deleted_values dimension: ', _values.shape)
    elif selected_FSR_list is not None:
        print('selected FSR list: ', selected_FSR_list)
        _values = _values[:, selected_FSR_list]
        print('selected_values dimension: ', _values.shape)

    time_mean = np.mean(_values, axis=1)
    # print('np_mean shape: ', np_mean.shape)

    # get non-zero mean
    # __mean_nz = np.apply_along_axis(lambda x: np.mean(x[x != 0]), 1, _values)
    # __min_nz = np.apply_along_axis(lambda x: np.min(x[x != 0]), 1, _values)
    # __max_nz = np.apply_along_axis(lambda x: np.max(x[x != 0]), 1, _values)
    # __std_nz = np.apply_along_axis(lambda x: np.std(x[x != 0]), 1, _values)

    # print(len(__mean_nz))
    # print(df.shape)

    sensor_mean = np.mean(_values, axis=0)
    print('sensor_mean shape: ', sensor_mean.shape)
    zero_indice = np.argwhere(sensor_mean == 0).reshape(-1)
    print('zero_indice: ', zero_indice)

    print('org mean: ', np.mean(sensor_mean))

    if start_point is not None and end_point is not None:
        timestamp = df['timestamp'].values[start_point:end_point]
    else:
        timestamp = df['timestamp'].values

    df = pd.DataFrame({
        'timestamp': timestamp,
        'mean': time_mean,
        # 'min': __min_nz,
        # 'max': __max_nz,
        # 'std': __std_nz
    })

    __mean_sub_ema = signalgo.subtract_ema(time_mean, 0.01)
    __mean_ema = signalgo.ema(__mean_sub_ema, 0.05)
    # peaks = signalgo.get_peaks(__mean_ema)

    plt.figure(figsize=(16, 14))
    plt.subplot(2, 1, 1)
    plt.plot(time_mean, color='red')
    # plt.title('temperature XX degree of celcius - non-zero mean')
    # plt.subplot(4, 1, 2)
    # plt.plot(__mean_sub_ema)
    # plt.plot(__mean_ema, color='red')
    # plt.title('(filtered by ema)')

    if without_ratio:
        __mean_rt = time_mean
    else:
        __mean_rt = time_mean * ratio  # RATIO

    compensated_mean = np.mean(__mean_rt)

    print(f'Compensation mean: {compensated_mean}')
    __mean_rt_sub_ema = signalgo.subtract_ema(__mean_rt, 0.01)
    __mean_rt_ema = signalgo.ema(__mean_rt_sub_ema, 0.05)

    if not without_ratio:
        plt.subplot(2, 1, 2)
        plt.plot(__mean_rt, color='blue')
        # plt.title('reduced by result of regression - non-zero mean')
        # plt.subplot(4, 1, 4)
        # plt.plot(__mean_rt_ema, color='blue')
        # plt.title('(filtered by ema)')
        plt.show()

    return compensated_mean
