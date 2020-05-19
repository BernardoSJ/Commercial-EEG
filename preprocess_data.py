import os
import re
from pprint import pprint


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


def main():
    devices_info = get_devices_info()
    print(devices_info)


if __name__ == '__main__':
    main()
