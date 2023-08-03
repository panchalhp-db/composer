from __future__ import annotations

import logging
import os
import pathlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, List
from urllib.parse import urlparse
import json
import pyarrow.parquet as pq

import requests
import uuid

from composer.utils.import_helpers import MissingConditionalImportError
from composer.utils.object_store.object_store import ObjectStore
from composer.utils.object_store.s3_object_store import S3ObjectStore

log = logging.getLogger(__name__)


@dataclass
class S3Credentials:
    bucket: str
    parquet_files: List[str]
    aws_access_key_id: str
    aws_session_token: str
    aws_secret_access_key: str


class Operation(Enum):
    READ = 1
    READ_WRITE = 2


def timed(func):

    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        end = time.time()
        total_time_sec = int(end - start)
        log.warn(f"Function `{func.__module__}.{func.__name__}(..)` took {total_time_sec} seconds to execute.")

    return inner


class UCTableObjectStore(ObjectStore):
    """
    Utility for uploading and downloading models from Unity Catalog Tables
    """

    def __init__(
        self,
        table_name: str,
        databricks_host_name: Optional[str] = None,
        databricks_token: Optional[str] = None,
    ) -> None:
        try:
            import databricks
        except ImportError as e:
            raise MissingConditionalImportError('databricks') from e

        if not databricks_host_name:
            databricks_host_name = os.environ['DATABRICKS_HOST']

        # sanitize the host name to remove the workspace id
        parse_result = urlparse(databricks_host_name)
        self.databricks_host_name = f'{parse_result.scheme}://{parse_result.netloc}'

        if not databricks_token:
            databricks_token = os.environ['DATABRICKS_TOKEN']
        self.databricks_token = databricks_token

        if len(table_name.split('.')) != 3:
            raise ValueError(f'Invalid UC Table name. Table should be of the format: <catalog>.<schema>.<table_name>')

        self.table_name = table_name
        self.catalog, self.schema, self.table = self.table_name.split('.')
        log.info(
            f'Initialized UCTableObjectStore with table_name={self.table_name} and host={self.databricks_host_name}')

    def get_temporary_credentials(self, operation: Operation) -> S3Credentials:
        from databricks.sdk import WorkspaceClient

        ws_client = WorkspaceClient()  # auth automatically picked up from env variables
        table_info = ws_client.tables.get(self.table_name)

        log.info(f'Fetched table_info={table_info}')

        storage_location = table_info.storage_location
        parse_result = urlparse(storage_location)
        backend, bucket, storage_prefix = parse_result.scheme, parse_result.netloc, parse_result.path
        if storage_prefix:
            storage_prefix = storage_prefix.strip('/')
        storage_prefix += '/'

        if backend != 's3':
            raise ValueError(f'The remote backend {backend} is not supported for UCTables')

        # fetch temporary creds
        url = f'{self.databricks_host_name}/api/2.1/unity-catalog/temporary-table-credentials'
        data = {'table_id': table_info.table_id, 'operation': operation.name}

        resp = requests.post(url, json=data, headers={'Authorization': f'Bearer {self.databricks_token}'})
        if resp.status_code != 200:
            raise Exception(f'Calling {url} resulted in status_code={resp.status_code} with message {resp.raw}')

        parsed_resp = resp.json()
        expiration_time_seconds = int(parsed_resp['expiration_time'] / 1000)
        aws_credentials = parsed_resp['aws_temp_credentials']

        epoch_time_now = int(time.time())
        log.info(f'UC credentials expire in {expiration_time_seconds - epoch_time_now} seconds.')

        import boto3
        from botocore.config import Config
        client = boto3.session.Session().client(
            "s3",
            config=Config(),
            endpoint_url=None,
            aws_access_key_id=aws_credentials['access_key_id'],
            aws_session_token=aws_credentials["session_token"],
            aws_secret_access_key=aws_credentials["secret_access_key"],
        )

        parquet_files = [
            content["Key"]
            for content in client.list_objects(Bucket=bucket, Prefix=storage_prefix)["Contents"]
            if content["Key"].endswith("parquet")
        ]
        log.info(f"Parquet files listed are: {parquet_files}")

        return S3Credentials(
            bucket=bucket,
            parquet_files=parquet_files,
            aws_access_key_id=aws_credentials['access_key_id'],
            aws_session_token=aws_credentials['session_token'],
            aws_secret_access_key=aws_credentials['secret_access_key'],
        )

    @timed
    def upload_object(self,
                      object_name: str,
                      filename: str | pathlib.Path,
                      callback: Callable[[int, int], None] | None = None) -> None:
        pass

    def download_object(self,
                        object_name: str,
                        filename: str | pathlib.Path,
                        overwrite: bool = False,
                        callback: Callable[[int, int], None] | None = None) -> None:

        if not object_name.endswith(".jsonl"):
            return FileNotFoundError

        # get temporary credentials w/ details on the exact parquet files
        s3_credentials = self.get_temporary_credentials(Operation.READ)
        s3_client = S3ObjectStore(bucket=s3_credentials.bucket,
                                  aws_access_key_id=s3_credentials.aws_access_key_id,
                                  aws_session_token=s3_credentials.aws_session_token,
                                  aws_secret_access_key=s3_credentials.aws_secret_access_key)

        json_list = []
        for parquet_file in s3_credentials.parquet_files:
            local_tmp_file = f"/tmp/{str(uuid.uuid4())[:8]}.parquet"
            s3_client.download_object(object_name=parquet_file,
                                      filename=local_tmp_file,
                                      overwrite=overwrite,
                                      callback=callback)
            for _, row in pq.read_table(local_tmp_file).to_pandas().iterrows():
                data = {"prompt": row["prompt"], "response": row["response"]}
                json_list.append(json.dumps(data))

        with open(filename, "w+") as f:
            for row in json_list:
                f.write(f"{row}\n")

    def get_object_size(self, object_name: str) -> int:
        s3_client = self.get_temporary_s3_client(Operation.READ)
        return s3_client.get_object_size(object_name=object_name)

    def get_uri(self, object_name: str) -> str:
        return f'uc://{self.table_name}/{object_name}'
