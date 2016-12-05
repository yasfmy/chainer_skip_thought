from urllib.request import urlopen
from urllib.parse import urljoin
import os

host = 'https://raw.githubusercontent.com'

url_list = (
                urljoin(host, 'tomsercu/lstm/master/data/ptb.train.txt'),
                urljoin(host, 'tomsercu/lstm/master/data/ptb.valid.txt'),
                urljoin(host, 'tomsercu/lstm/master/data/ptb.test.txt')
            )

def save_file_from_url(url, filename=None):
    response = urlopen(url)
    filename = os.path.basename(url) if not filename else filename
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(str(response.read()))

if __name__ == '__main__':
    for url in url_list:
        save_file_from_url(url)
