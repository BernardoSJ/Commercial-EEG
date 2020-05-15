import os
import mne
import glob
import pandas as pd
from pprint import pprint


_sample_rate = 256


def extract_patient_metadata(patient: str):
    ''' Get metadata of `patient`'s EEG recordings

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
    summary_file = f'./data/chb{patient}/chb{patient}-summary.txt'
    eegs = glob.glob(f'./chb{patient}/*.edf')

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
            info_eeg[key] = value
        info_eegs.append(info_eeg)

    return info_eegs


def get_data(patient, channels, window_seconds=30)
    info_eegs = extract_patient_metadata(patient)
    frames = window_seconds * _sample_rate
    overlay = int(frames/2)
    for info in info_eegs:
        eeg_file = f"./data/chb{patient}/{info['File Name']}"
        eeg = mne.io.read_raw_edf(eeg_file)
        eeg_channels = eeg.ch_names

        raw_eeg = eeg.get_data(picks=channels)
        length = raw_eeg.shape[1]

        dfs = []

        for i in range(length//overlay - 1):
            start = i * overlay
            end = start + frames
            data = raw_eeg[:, start:end].T
            frame = np.arange(start, end).reshape(-1, 1)
            index = np.full((end - start, 1), i)
            columns = channels.copy()
            columns.append('frame')
            columns.append('window')

            df_i = pd.DataFrame(
                data=np.hstack((data, frame, index)),
                columns=columns)

            dfs.append(df_i)

        df = pd.concat(dfs)

        df['seizure'] = 0

        n_seizures = int(info['Number of Seizures in File'])

        if n_seizures > 0:
            seizures_keys = filter(lambda k: k.startswith('Seizure'), info)
            seizures = [[]*n_seizures]
            for i, key in enumerate(seizures_keys):
                value_seconds = int(info[key].split(' ')[0])
                value_frames = value_seconds * sample_rate
                seizures[i//2].append(value_frames)
            
            for s_start, s_end in seizures:
                df.loc[(df['frame'] >= s_start) & (df['frame'] <= s_end), 'seizure'] = 1

        else:
            df['seizure'] = 0

    return df
