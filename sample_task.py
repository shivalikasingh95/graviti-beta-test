import logging
import torch
import sys

logging.basicConfig(level=logging.INFO)
logging.info("hello")
MODEL_NAME = sys.argv[0]
logging.info(f"Model name: {MODEL_NAME}")
EPOCHS = sys.argv[1]
logging.info(f"Epoch: {EPOCHS}")