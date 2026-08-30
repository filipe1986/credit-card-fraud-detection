import os
import tarfile
import urllib.request

DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0RHPEN/data/creditcard.tgz"
ARCHIVE_NAME = "creditcard.tgz"
CSV_NAME = "creditcard.csv"


def download_and_extract_data():
    if os.path.exists(CSV_NAME):
        print(f"'{CSV_NAME}' already exists. Skipping download.")
        return

    if not os.path.exists(ARCHIVE_NAME):
        print("Downloading credit card dataset...")
        urllib.request.urlretrieve(DATA_URL, ARCHIVE_NAME)
        print("Download complete.")

    print("Extracting dataset archive...")
    with tarfile.open(ARCHIVE_NAME, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete. Dataset ready!")


if __name__ == "__main__":
    download_and_extract_data()
    