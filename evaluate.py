import torch
import logging
import os

from PIL import Image
from graviti import DataFrame, Workspace
from graviti.utility import File
import graviti.portex as pt

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
import torchvision.transforms as T


logging.basicConfig(level=logging.INFO)

import torch.nn as nn
import torch.nn.functional as F

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    cat_dict = {0:'roses', 1: 'sunflowers', 2: 'daisy', 3: 'dandelion', 4: 'tulips'}
    print("preds[0]:",preds[0])
    print("preds[0].item():",preds[0].item())
    return cat_dict[preds[0].item()]


# Building a Network Architecture.
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '', 
            result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)   
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)    
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))   
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                        nn.Flatten(),     
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))    
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out



if __name__ == "__main__":

    ACCESS_KEY = os.environ.get("secret.accesskey")
    ws = Workspace(ACCESS_KEY)
    flower_model = ws.datasets.get("FLOWERS_MODEL")

    dataset = ws.datasets.get("Beta-Test-1")

    device = get_default_device()
    logging.info(f"Using {device} device")

    test_data = dataset['test_data']

    model_df = flower_model["final_model"]
    model_file = model_df.loc[0]['model_file']
    print(model_file)
    modelcurr = to_device(ResNet9(3, 5), device)
    with open(f"./model_out.pth", "wb") as fp:  # Path where data is stored locally
            fp.write(model_file.open().read())

    modelcurr.load_state_dict(torch.load("model_out.pth", map_location=torch.device("cuda")))
    img_size = 64
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize((img_size, img_size)),
                            T.RandomCrop(64, padding=4, padding_mode='reflect'),
                            T.RandomHorizontalFlip(),
                            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            T.ToTensor(),
                            T.Normalize(*stats,inplace=True)])

    for idx in range(len(test_data)):
        data_img = test_data.loc[idx]['image']

        with data_img.open() as fp:
            img = transform(Image.open(fp))
        label = test_data.loc[idx]['category']

        #img, label = valid_ds[200]
        #plt.imshow(img.permute(1, 2, 0).clamp(0, 1))
        cat_dict = {0:'roses', 1: 'sunflowers', 2: 'daisy', 3: 'dandelions', 4: 'tulips'}
        cat_dict_orig = {'roses': 0, 'sunflowers': 1, 'daisy':2,'dandelion':3, 'tulips':4}
        pred = predict_image(img, modelcurr)
        print('Label:', cat_dict_orig[label], ', Predicted:', pred )
        logging.info(f"Predicted: {pred}")