import os
import gc
import re
import time
import pandas as pd
import numpy as np
import psutil
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from pprint import pprint


_data_subpath = 'data/devices'
_data_path = os.path.join(os.getcwd(), _data_subpath)

_seizure_ratio = 0.4
_homology_dimensions = [0]
_max_edge_length = 0.01
_n_jobs = 4


def get_devices():
    ''' Extract name and number of channels of each EEG device

        Returns
        -------
        List of tuples of the form (device, n_channels)
        - device: str
        - n_channels: int
    '''
    with open('Electroencefalografos.txt', 'r') as f:
        devices_info_txt = f.read()

    devices_info_list = devices_info_txt.split('\n\n')[1:]
    prog_name = re.compile('^\d- (\S+)')

    devices = list()

    for device_info_txt in devices_info_list:
        rows = device_info_txt.split('\n')
        rows = list(map(lambda x: x.strip(), rows))
    
        match = prog_name.search(rows[0])
        name = match.group(1)
        
        pairs_start = rows.index('Pares:') + 1
        pairs_end = rows.index('Sampling rate:')
        
        channels = pairs_end - pairs_start

        devices.append((name, channels))

    return devices


def is_seizure(df):
    ''' Determines wether the given dataframe contains enough
        frames with seizure information given _seizure_ratio
    '''
    seizure_frames = df.loc[df['seizure'] == 1]
    seizure_length = seizure_frames.shape[0]
    window_length = df.shape[0]
        
    ratio = seizure_length / window_length

    return ratio >= _seizure_ratio


def get_persistent_entropy(point_clouds):
    ''' Creates Vietoris Rips Filtration and calculates Persistent Entropy

        Returns
        -------
        List with persistent entropy of 0th homology group for each series
    '''
    vietorisrips_tr = VietorisRipsPersistence(
        metric='manhattan',
        homology_dimensions=_homology_dimensions,
        max_edge_length=_max_edge_length,
        n_jobs=_n_jobs,
    )
    diagrams = vietorisrips_tr.fit_transform(point_clouds)

    entropy_tr = PersistenceEntropy()
    features = entropy_tr.fit_transform(diagrams)

    return features


def get_last_test(file_path, case):
    ''' Locates the latest test processed from the given `case`

        Returns
        -------
        Integer indicating index of last processed test
        will return 0 if no test has been processed
    '''
    if os.path.isfile(file_path):
        df = pd.read_hdf(file_path)
        tests = df.loc[df['case'] == case, 'test']
        test = 0 if tests.empty else tests.max()
    else:
        test = 0

    return test


def get_last_window(file_path, case, test):
    ''' Locates the latest window processed from the given `case` and `test`

        Returns
        -------
        Integer indicating index of last processsed window
        will return -1 if no window has been processed
    '''
    if os.path.isfile(file_path):
        df = pd.read_hdf(file_path)
        windows = df.loc[(df['case'] == case) & (df['test'] == test), 'window']
        window = -1 if windows.empty else windows.max()
    else:
        window = -1

    return window


def aggregate_data(device, case, n_channels):
    ''' Reduces data to a single row from each window
        Writes to disk each row
    '''
    print('Processing data for case:', case)

    case_data_path = os.path.join(_data_path, device, f'{case}.hdf')
    df = pd.read_hdf(case_data_path)
    case_tests = df['test'].max()
    
    device_file_path = os.path.join(_data_path, device, f'{device}.hdf')
    last_test = get_last_test(device_file_path, case)

    channels = list(df)[:n_channels]

    for test in range(last_test, case_tests):
        print('Processing test:', test)
        df_test = df.loc[df['test'] == test]

        last_window = get_last_window(device_file_path, case, test)
        print(last_window)
        test_windows = int(df_test['window'].max())

        for window in range(last_window + 1, test_windows):
            # Measure performance
            psutil.cpu_percent()
            start = time.time()

            df_window = df_test.loc[df_test['window'] == window]
            seizure = is_seizure(df_window)

            row = {
                'case': case,
                'test': test,
                'window': window,
                'seizure': seizure,
            }
            
            point_clouds = [df_window[channel].to_numpy().reshape(-1, 1) 
                            for channel in channels]
            entropies = get_persistent_entropy(point_clouds)

            for i in range(n_channels):
                row.update({channels[i]: entropies[i]})

            # Measure performance
            end = time.time()

            row.update({'elapsed time': end - start})
            row.update({'cpu usage': psutil.cpu_percent()})
            row.update({'memory usage': psutil.virtual_memory()[3]})
            row = pd.DataFrame(row)

            row.to_hdf(device_file_path, 'df', append=True)
        gc.collect()


def main():
    devices = get_devices()
    n_devices = len(devices)

    # Find checkpoint of processed devices
    for i in reversed(range(n_devices)):
        device, _ = devices[i]
        device_file_path = os.path.join(_data_path, device, f'{device}.hdf')
        if os.path.isfile(device_file_path):
            if devices[:i]:
                print('Skipping the following devices:')
                print(devices[:i])
            break

    devices = devices[i:]

    for device, n_channels in devices:
        print('Processing data for device:', device)
        print('Number of channels:', n_channels)
        
        device_file_path = os.path.join(_data_path, device, f'{device}.hdf')
        
        start = 1

        # Determine last case
        if os.path.isfile(device_file_path):
            df_device = pd.read_hdf(device_file_path)
            last_case = df_device['case'].max()
            start = int(last_case[3:])
            del df_device
            gc.collect()
            if start > 1:
                print('Skipping cases before case', last_case)

        for i in range(start, 25):
            case_num = '0' + str(i) if i < 10 else str(i)
            case = f'chb{case_num}'
            aggregate_data(device, case, n_channels)


if __name__ == "__main__":
    main()
