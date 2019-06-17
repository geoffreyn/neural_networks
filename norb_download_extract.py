#!/usr/bin/env python3

import datetime
import os
import sys

import gzip
import shutil
from urllib.request import urlopen


def size_to_string(bytes_):
    if (bytes_ > 2**11) and (bytes_ < 2**21):
        byte_str = "{:0.2f} KB".format(bytes_ / 2**10)
    elif (bytes_ > 2**21) and (bytes_ < 2**31):
        byte_str = "{:0.2f} MB".format(bytes_ / 2**20)
    elif (bytes_ > 2**31) and (bytes_ < 2**41):
        byte_str = "{:0.2f} GB".format(bytes_ / 2**30)
    elif bytes_ > 2**41:
        byte_str = "{:0.2f} TB".format(bytes_ / 2**30)
    else:
        byte_str = "{:0.1f} B".format(bytes_)

    return byte_str


# Adapted to return file from https://stackoverflow.com/a/2030027
def chunk_report(bytes_so_far, chunk_size, total_size, last_chunk_time):
    this_chunk_time = datetime.datetime.now()
    percent = bytes_so_far / total_size * 100

    byte_str = size_to_string(bytes_so_far)
    total_byte_str = size_to_string(total_size)

    time_delta = (this_chunk_time - last_chunk_time).microseconds / 10**6
    data_rate = max(chunk_size / time_delta, 1)  # Avoid divide-by-zero

    data_rate_str = size_to_string(data_rate)

    sys.stdout.write("Downloaded {} of {}: {:0.2f}% ({}/s)            \r".
        format(byte_str, total_byte_str, percent, data_rate_str))

    if bytes_so_far >= total_size:
        sys.stdout.write('\n')

    last_chunk_time = this_chunk_time
    return last_chunk_time


def chunk_read(response, chunk_size=8192, report_hook=None):
    total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0
    data = b""

    last_chunk_time = datetime.datetime.now()
    while 1:
        chunk = response.read(chunk_size)

        bytes_so_far += len(chunk)

        if not chunk:
            break

        if report_hook:
            last_chunk_time = report_hook(bytes_so_far, chunk_size,
                              total_size, last_chunk_time)

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
                    data_read = chunk_read(response,
                                           chunk_size=2**15,
                                           report_hook=chunk_report)

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
