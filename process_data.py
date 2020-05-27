import os
import gc
import re
import time
import pandas as pd
import numpy as np
from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from pprint import pprint


_data_subpath = 'data/devices'
_cwd = os.getcwd()
_data_path = os.path.join(_cwd, _data_subpath)

_seizure_ratio = 0.5
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
    vietorisrips_tr = VietorisRipsPersistence(
        metric='manhattan',
        homology_dimensions=_homology_dimensions,
        max_edge_length=_max_edge_length,
        n_jobs=_n_jobs,
    )
    print('Creating Vietoris Rips Complex')
    diagrams = vietorisrips_tr.fit_transform(point_clouds)
    entropy_tr = PersistenceEntropy()

    print('Calculating Persistent Entropy')
    features = entropy_tr.fit_transform(diagrams)
    return features


def aggregate_data(device, case, n_channels):
    ''' Reduces data to a single row from each window
        Writes to disk
    '''
    device_file_path = os.path.join(_data_path, device, f'{device}.hdf')

    # Locating checkpoint
    if os.path.isfile(device_file_path):
        df_device = pd.read_hdf(device_file_path)
        last_test = int(df_device['test'].max())
        last_window = int(df_device.loc[df_device['test'] == last_test, 'window'].max())
    else:
        last_test = 0
        last_window = 0

    print('Processing data for case:', case)

    data_path = os.path.join(_data_path, device, f'{case}.hdf')

    print('Reading file')
    df = pd.read_hdf(data_path)
    case_tests = int(df['test'].max())

    channels = list(df)[:n_channels]

    for test in range(last_test, case_tests):
        print('Processing test:', test)
        df_test = df.loc[df['test'] == test]
        test_windows = int(df_test['window'].max())
        for window in range(last_window + 1, test_windows):
            start = time.time()
            print('Processing window:', window)
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

            end = time.time()

            row.update({'elapsed_time': end - start})
            row = pd.DataFrame(row)

            row.to_hdf(device_file_path, 'df', append=True)
            print(row)
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
