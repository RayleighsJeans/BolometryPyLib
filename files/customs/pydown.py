from tqdm import tqdm
import requests
import os
import links

def download():
    os.chdir(r'\\sv-it-fs-1\Downloads$\pih\Downloads\PYDOWN')

    li = links.linklist()
    names = ['mrE08_1.7.mkv',
             'mrE09_1.8.mkv',
             'mrE10_1.9.mkv',
             'mrE11_2.01.mkv',
             'mrE12_2.02.mkv',
             'mrE13_2.1.mkv',
             'mrE14_2.2.mkv'
            ]

    for link, name in zip(li, names):

        if not os.path.isfile(name):
            print('>> file doesn\'t exist, downloading', name, '...')

            response = requests.get(link, stream=True)
            with open(name, "wb") as handle:
                for data in tqdm(response.iter_content()):
                    handle.write(data)

        elif os.path.isfile(name):
            print('>> file already exists:', name)

            if (os.stat(name).st_size / 1e6 < 500.0):
                print('\t>> file is is small:', 
                      os.stat(name).st_size / 1e6, 'MB\n' + 
                      '\t\t ... deleting')
                os.remove(name)

                response = requests.get(link, stream=True)
                with open(name, "wb") as handle:
                    for data in tqdm(response.iter_content()):
                        handle.write(data)

        else:
            print('\t>> file exists:',
                  os.stat(name).st_size / 1e6 < 500.0, 'MB')
            pass

    return