#!/usr/local/bin python3

import os
import sys

import gzip
import shutil

from urllib.request import urlopen


# Adapted to return file from https://stackoverflow.com/a/2030027
def chunk_report(bytes_so_far, chunk_size, total_size):
    percent = float(bytes_so_far) / total_size
    percent = round(percent*100, 2)
    sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" %
                     (bytes_so_far, total_size, percent))

    if bytes_so_far >= total_size:
        sys.stdout.write('\n')


def chunk_read(response, chunk_size=8192, report_hook=None):
    total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0
    data = b""

    while 1:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            break

        if report_hook:
            report_hook(bytes_so_far, chunk_size, total_size)

        data += chunk

    return data
# -

def filepath_generator():
    base_url = "https://cs.nyu.edu/%7Eylclab/data/norb-v1.0/"
    base_name_training = "norb-5x46789x9x18x6x2x108x108"
    base_name_testing = "norb-5x01235x9x18x6x2x108x108"

    for dataset in ("training", "testing"):
        for datatype in ("cat", "dat", "info"):
            for filenum in range(1, 11):
                if (dataset == "testing"):
                    base_name = base_name_testing
                    if (filenum > 2):
                        pass
                else:
                    base_name = base_name_training

                filenum_str = str(filenum).zfill(2)

                filename = "{}-{}-{}-{}.mat.gz".format(base_name,
                                                   dataset, filenum_str,
                                                   datatype)

                yield base_url, filename


def download_files():
    os.makedirs(os.path.join("norb_data", "archives"), exist_ok=True)

    out_path = None
    try:
        for base_url, filename in filepath_generator():
            filepath = "{}{}".format(base_url, filename)

            out_path = os.path.join("norb_data", "archives", filename)

            if not os.path.exists(out_path):
                print("Downloading: {}".format(filepath))

                with open(out_path, "wb") as f:
                    response = urlopen(filepath)
                    data_read = chunk_read(response, report_hook=chunk_report)

                    f.write(data_read)
    except:
        if out_path:
            os.remove(out_path)
        raise ValueError("Download failed, aborting!")

    print("Finished downloading files!")


def extract():
    os.makedirs(os.path.join("norb_data", "extracted"), exist_ok=True)

    out_path = None
    last_path = None
    try:
        for _, filename in filepath_generator():
            print("Extracting {}...".format(filename))
            last_path = out_path

            in_path = os.path.join("norb_data", "archives", filename)
            out_path = os.path.join("norb_data", "extracted",
                                    filename.rstrip((".gz")))

            with gzip.open(in_path, "rb") as f_in:
                with open(out_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
    except:
        if last_path:
            os.remove(last_path)
        raise ValueError("Extraction failed, aborting!")


def main():
    download_files()

    extract()


if __name__ == '__main__':
    sys.exit(main())
