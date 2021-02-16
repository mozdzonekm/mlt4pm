import os
from zipfile import ZipFile
import requests

archive_name = 'computers_train.zip'
destination_folder = 'computers_train'

print('Downloading the data...\n')
url = f'http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/trainsets/{archive_name}'
r = requests.get(url)
open(archive_name, 'wb').write(r.content)

print('Extracting...\n')
with ZipFile(archive_name, 'r') as zip:
    zip.printdir()
    zip.extractall(destination_folder)
os.remove(archive_name)
print()

print('Done!')


