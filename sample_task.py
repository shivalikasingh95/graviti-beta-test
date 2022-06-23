import logging
import torch
from PIL import Image
from graviti import DataFrame, Workspace
from graviti.utility import File
import graviti.portex as pt
from torchvision import transforms
logging.basicConfig(level=logging.INFO)
logging.info("hello taskAA!")
