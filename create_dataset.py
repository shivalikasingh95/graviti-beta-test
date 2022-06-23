import logging
import os
logging.basicConfig(level=logging.INFO)
from graviti import DataFrame, Workspace
from graviti.utility import File
import graviti.portex as pt

ACCESS_KEY = os.environ.get("secret.accesskey")
dataset_name = "FLOWERS_MODEL"
ws = Workspace(ACCESS_KEY)
try:
    ws.datasets.create(dataset_name)
    logging.info(f"Created dataset {dataset_name} Successfully")
except:
    logging.info(f"{dataset_name} aleady exists.")