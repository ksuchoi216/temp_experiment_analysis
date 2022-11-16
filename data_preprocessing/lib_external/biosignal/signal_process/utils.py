import os
import time
import numpy as np
import pandas as pd
import fitparse
from datetime import timedelta


def str2numpy(src):
    # convert list of string to numpy array
    return np.asarray([s.split() for s in src], dtype=int)


def timestamp_in_seconds(src):
    # return timestamp in seconds
    src = np.asarray(src, dtype=float)
    if len(str(src[0])) >= 13:
        src = src / 1000
    return src


def get_file_list(root, exp='csv', log=False):
    temp = sorted(os.listdir(root))
    files = []
    for f in temp:
        if f[-len(exp):] == exp:
            files.append(os.path.join(root, f))
    if log:
        for i, f in enumerate(files):
            print(i, f)
    return files


def decode_ppgbcg(df):
    size = len(df)
    times = []
    sensors = []
    datetime = []
    ch_data = df['CH_data']
    timestamps = df['timestamp']
    for i in range(size):
        sensor = ch_data[i]
        sensors.append(np.asarray(sensor.split(), dtype=int))
        timestamp = timestamps[i]
        lt = time.localtime(timestamp // 1000)
        tf = '%04d%02d%02d-%02d%02d%02d' % (lt.tm_year, lt.tm_mon,
                                            lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)
        datetime.append(tf)

    sensors = np.asarray(sensors)

    df_new = pd.DataFrame({
        'timestamp': df['timestamp'] / 1000.0,
        'datetime': datetime,
        'PC': df['PC'],
        'PCD': df['PCD'],
        'CH1': sensors[:, 0],
        'CH2': sensors[:, 1],
        'CH3': sensors[:, 2],
        'CH4': sensors[:, 3],
        'CH5': sensors[:, 4],
        'CH6': sensors[:, 5],
        'CH7': sensors[:, 6],
        'CH8': sensors[:, 7],
    })

    return df_new


def get_ppgbcg(root):
    return decode_ppgbcg(pd.read_csv(root))


fsr_position = [1, 5, 20, 24,
                2, 6, 19, 23,
                3, 7, 18, 22,
                4, 8, 17, 21,
                9, 13, 28, 0,
                10, 14, 27, 31,
                11, 15, 26, 30,
                12, 16, 25, 29]


def decode_fsr(df):
    size = len(df)
    sensors = []
    datetime = []
    ch_data = df['sensor']
    timestamps = df['timestamp']
    for i in range(size):
        sensor = ch_data[i]
        sensors.append(np.asarray(sensor.split(), dtype=int))
        timestamp = timestamps[i]
        lt = time.localtime(timestamp // 1000)
        tf = '%04d%02d%02d-%02d%02d%02d' % (lt.tm_year, lt.tm_mon,
                                            lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)
        datetime.append(tf)

    sensors = np.asarray(sensors)
    sensors = sensors[:, fsr_position].copy()
    sensors_sum = sensors.sum(1)

    df_new = pd.DataFrame({
        'timestamp': df['timestamp'] / 1000.0,
        'datetime': datetime,
        'amp_sensor': df['amp_sensor'],
        'sens_sensor': df['sens_sensor'],
        'sensor_sum': sensors_sum,
        'sensor': df['sensor']
    })

    return df_new


def get_fsr(root):
    return decode_fsr(pd.read_csv(root))


def get_fit(fitfile_path):
    # parse fit file

    fitfile = fitparse.FitFile(fitfile_path)
    delta_KST = timedelta(hours=9)

    values = []
    for record in fitfile.get_messages('record'):
        value = {}
        for data in record:
            value[data.name] = data.value
            values.append(value)

    df = pd.DataFrame.from_records(values)

    dt = df['timestamp']
    timestamps = []
    for i in range(len(dt)):
        timestamp = dt[i].timestamp()
        timestamps.append(timestamp)

    df_new = pd.DataFrame({
        'datetime': dt + delta_KST,
        'timestamp': timestamps,
        'heart_rate': df['unknown_136'],
        'temperature': df['temperature'],
        'resp_rate': df['unknown_108'] / 100,
        'stress': df['unknown_116'] / 100
    })
    return df_new
