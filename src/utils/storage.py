from google.cloud import storage
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BUCKET_NAME = 'idp-models'


def upload_file(path, destination_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_name)
    blob.upload_from_filename(path)


def download_file(filename, destination):
    """Downloads a file from the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.download_to_filename(destination)
