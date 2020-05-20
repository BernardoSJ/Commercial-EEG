import os
import re
import preprocessing
from pprint import pprint


_data_subpath = 'data/devices'
_cwd = os.getcwd()
_data_path = os.path.join(_cwd, _data_subpath)


def get_devices_info():
    with open('Electroencefalografos.txt', 'r') as f:
        devices_info_txt = f.read()

    devices_info_list = devices_info_txt.split('\n\n')[1:]
    prog_name = re.compile('^\d- (\S+)')

    devices_info = list()

    for device_info_txt in devices_info_list:
        rows = device_info_txt.split('\n')
        rows = list(map(lambda x: x.strip(), rows))
    
        match = prog_name.search(rows[0])
        name = match.group(1)
        
        pairs_start = rows.index('Pares:') + 1
        pairs_end = rows.index('Sampling rate:')
        
        pairs = rows[pairs_start: pairs_end]
        
        device_info = {
            'name': name,
            'channels': pairs,
        }
        devices_info.append(device_info)

    return devices_info


def make_dirs(devices):
    for device in devices:
        name = device['name']
        device_path = os.path.join(_data_path, name)
        os.makedirs(device_path, exist_ok=True)


def preprocess(device, channels):
    device_path = os.path.join(_data_path, device)
    print('Processing data for device:', device)
    print('Number of channels:', len(channels))

    if(len(channels) == 0):
        print('Invalid channel selection, Skipping')
        return

    for i in range(1, 25):
        case_num = '0' + str(i) if i < 10 else str(i)
        case = f'chb{case_num}'
        file_path = os.path.join(device_path, f'{case}.hdf')

        if os.path.isfile(file_path):
            print('file')
            print('Skipping case')
            continue

        print('Processing case:', case)
        df = preprocessing.get_data(case_num, channels=channels)
        print(df)

        print('Writing to disk:', case)
        print('File path:', file_path)
        df.to_hdf(file_path, 'df')


def main():
    devices_info = get_devices_info()
    make_dirs(devices_info)
    for device in devices_info:
        name = device['name']
        channels = device['channels']
        preprocess(name, channels)


if __name__ == '__main__':
    main()
