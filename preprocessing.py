import gc
import os
import mne
import glob
import numpy as np
import pandas as pd
from pprint import pprint


_sample_rate = 256
_window_seconds = 10


def extract_case_metadata(case: str):
    ''' Get metadata of `case`'s EEG recordings

        Returns: List with dicts with corresponding metadata
        Data:
        - File Name: 
        - File Start Time
        - File End Time
        - Number of Seizures in File
        If Number of Seizures in File > 0:
        - Seizure i Start Time
        - Seizure i End Time
    '''
    summary_file = f'./data/chb{case}/chb{case}-summary.txt'
    eegs = glob.glob(f'./chb{case}/*.edf')

    with open(summary_file, 'r') as f:
        summary = f.read()

    summary_parts = summary.split('\n\n')[:-1]
    info_eegs_raw = filter(lambda part: part.startswith('File'), summary_parts)
    
    info_eegs = list()
    
    for file_info in info_eegs_raw:
        info_eeg = dict()
        for row in file_info.split('\n'):
            key, value = row.split(': ')
            value.strip()
            while key in info_eeg:
                key += '_'
            info_eeg[key] = value
        info_eegs.append(info_eeg)

    return info_eegs


def get_windows(eeg, columns, frames, length, overlay):
    ''' Partition the eeg into overlapping windows

        Returns: DataFrame
        Columns:
        - *Channels
        - seizure: boolean
            1 if the corresponding frame is part of a seizure
        - frame: index of the frame in the test
            may repeat twice in a given test because of the overlay of windows
        - window: index of the window in a test
    '''
    dfs = []
    for i in range(length//overlay - 1):
        start = i * overlay
        end = start + frames

        data = eeg[:, start:end].T
        frame = np.arange(start, end).reshape(-1, 1)
        window = np.full((end - start, 1), i)

        df = pd.DataFrame(
            data=np.hstack((data, frame, window)),
            columns=columns)

        dfs.append(df)

    return dfs


def flag_seizures(df, n_seizures, seizures_keys, info):
    ''' Flag the frames with seizure information
    '''
    seizures = [[] for i in range(n_seizures)]
    for i, key in enumerate(seizures_keys):
        value_str = info[key].strip()
        value_seconds = int(value_str.split(' ')[0])
        value_frames = value_seconds * _sample_rate
        seizures[i//2].append(value_frames)
    
    for s_start, s_end in seizures:
        df.loc[(df['frame'] >= s_start) & (df['frame'] <= s_end), 'seizure'] = 1

    return df


def optimize_memory(df, columns):
    types = dict()
    for col in columns:
        types[col] = 'float16'

    types.update({'seizure': bool, 'window': 'int16', 'frame': 'int32'})

    df = df.astype(types)
    return df


def get_data(case, channels=None, window_seconds=_window_seconds):
    ''' Gathers all EEG data of the given case into a DataFrame
        Selecting only the required channels of the EEG

        Returns: DataFrame
        Columns:
        - *Channels
        - seizure: boolean
            1 if the corresponding frame is part of a seizure
        - frame: index of the frame in the test
            may repeat twice in a given test because of the overlay of windows
        - window: index of the window in a test
    '''
    frames = window_seconds * _sample_rate
    overlay = int(frames/2)

    info_eegs = extract_case_metadata(case)
    info_eegs_filt = filter(lambda info: int(info['Number of Seizures in File']) > 0, info_eegs)
    info_eegs = list(info_eegs_filt)

    Df = pd.DataFrame()
    for test, info in enumerate(info_eegs):
        print(f'Reading test {test} of {len(info_eegs)}')
        eeg_file = f"./data/chb{case}/{info['File Name']}"
        eeg = mne.io.read_raw_edf(eeg_file)
        eeg_channels = eeg.ch_names

        # Correct channels
        if channels:
            try:
                channels = correct_channels(channels, eeg_channels)
                raw_eeg = eeg.get_data(picks=channels)
            except ValueError as e:
                print(e)
                continue
        else:
            raw_eeg =  eeg.get_data()

        length = raw_eeg.shape[1]

        columns = channels.copy() if channels else eeg_channels.copy()
        columns.extend(['frame', 'window'])
        
        dfs = get_windows(raw_eeg, columns, frames, length, overlay)
        df = pd.concat(dfs)
        
        del dfs
        gc.collect()

        df['test'] = test
        df['seizure'] = 0

        n_seizures = int(info['Number of Seizures in File'])

        seizures_keys = filter(lambda k: k.startswith('Seizure'), info)

        df = flag_seizures(df, n_seizures, seizures_keys, info)

        df = optimize_memory(df, columns)
        
        Df = Df.append(df)

    return Df


def correct_channels(channels, available_channels):
    for i in range(len(channels)):
        channel = channels[i]
        if channel not in available_channels:
            similar_channels = list(filter(lambda x: x.startswith(channel),
                                           available_channels))
            if len(similar_channels) == 0:
                raise ValueError('No suitable channel to replace')
            else:
                channels[i] = similar_channels[0]
    return channels
