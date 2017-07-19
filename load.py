import os
import csv
import sys
from urllib.request import urlretrieve

def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    """
    last_percent_reported = 0
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
      
        last_percent_reported = percent

def load_or_download(filename, url_i='assistments_train'):
    urls = {
        'assistments_train' : 'https://raw.githubusercontent.com/siyuanzhao/2016-EDM/master/data/0910_c_train.csv',
        'assistments_test' : 'https://raw.githubusercontent.com/siyuanzhao/2016-EDM/master/data/0910_c_test.csv'
    }
    if not os.path.exists(filename):
        assert url_i in urls, "file {} does not exist and no url supplied".format(filename)
        print("Attempting to download " + filename)
        filename, _ = urlretrieve(urls[url_i], filename, reporthook=download_progress_hook)
        print("\nDownload complete")
    return filename

def read_data_from_csv_file(filename, url_i='assistments_train'):
    tuples = []
    targets = []
    seen_probs = []
    filename = load_or_download(filename, url_i = url_i)
    print(filename)
    n_seq = None
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        while not n_seq:
            n_seq = next(csvreader)
        n_seq = int(n_seq[0])
        pids = next(csvreader)
        targets = next(csvreader)
        while targets:
            try:
                if len(pids) >= 2:
                    seen_probs += [p for p in set(pids) if p not in seen_probs]
                    tuples.append((n_seq,list(zip(map(int,pids),map(int,targets)))))
                n_seq = int(next(csvreader)[0])
                pids = next(csvreader)
                targets = next(csvreader)
            except StopIteration:
                tuples.append((n_seq,list(zip(map(int,pids),map(int,targets)))))
                targets = None
    print("The number of students is ", len(tuples))

    id2idx = { int(j): int(i) for i, j in enumerate(seen_probs)}

    return tuples, id2idx