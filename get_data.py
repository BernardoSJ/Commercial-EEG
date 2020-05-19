import os

_record_download_command = 'gsutil -m cp -r gs://chbmit-1.0.0.physionet.org/{case}/{record} data/{case}/'

_summary_download_command = 'gsutil -m cp -r gs://chbmit-1.0.0.physionet.org/{case}/{case}-summary.txt data/{case}/'

_data_subpath = 'data'
_cwd = os.getcwd()
_data_path = os.path.join(_cwd, _data_subpath)

_download_tries = 5


def make_dirs():
    ''' Create the directories required to locally store the EEG data
        as well as download the summary file for each case
    '''
    for i in range(1, 25):
        num = '0' + str(i) if i < 10 else str(i)
        case = f'chb{num}'
        case_path = os.path.join(_data_path, case)
        os.makedirs(case_path, exist_ok=True)

        file_path = os.path.join(case_path, f'{case}-summary.txt')
        if not os.path.isfile(file_path):
            print('Downloading summary file for case: ', case)
            result = os.system(_summary_download_command.format(case=case))


def main():
    print('Creating data directory and subdirectories')
    make_dirs()
    
    with open('Records.txt', 'r') as f:
        records_txt = f.read()
    records = records_txt.split('\n')
    records = filter(lambda x: x, records)

    print('Downloading records')
    for row in records:
        print('Downloading record:', row)
        case, record = row.split('/')
        
        file_path = os.path.join(_data_path, row)
        if os.path.isfile(file_path):
            print('Skiping record: ', record)
            print('Record has been downloaded previously')
            continue

        result = 1
        tries = 0
        
        while result != 0:
            if tries < _download_tries:
                tries += 1
                result = os.system(
                    _record_download_command.format(case=case, record=record))
            else:
                print('Skiping record: ', record)
                print('Record took too many tries to download')
                result = 0


if __name__ == '__main__':
    main()
