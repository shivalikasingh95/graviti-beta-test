import logging
import os

import torch
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

import torch

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

# Read Datasets from Graviti platform.
class FlowerSegment(Dataset):
    """class for wrapping Flower Dataset Segment."""

    def __init__(self, dataset, sheet_name, transform):
        super().__init__()
        self.dataset = dataset
        self.sheet = self.dataset[sheet_name]
        self.transform = transform

    def __len__(self):
        return len(self.sheet)

    def __getitem__(self, idx):
        data_img = self.sheet.loc[idx]['image']
        cat_dict = {'roses': 0, 'sunflowers': 1, 'daisy':2,'dandelion':3, 'tulips':4}
        data_category = self.sheet.loc[idx]['category']
        with data_img.open() as fp:
            image_tensor = self.transform(Image.open(fp))
            
        return image_tensor, cat_dict[data_category]

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                    weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    
    torch.save(model.state_dict(), "model.pth")
    logging.info("Saved PyTorch Model State to model.pth")
    return history

if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 3
    ACCESS_KEY = os.environ.get("secret.accesskey")
    ws = Workspace(ACCESS_KEY)
    flower_dataset = ws.datasets.get("Beta-Test-1") 

    img_size = 64
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = T.Compose([T.Resize((img_size, img_size)),
                            T.RandomCrop(64, padding=4, padding_mode='reflect'),
                            T.RandomHorizontalFlip(),
                            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            T.ToTensor(),
                            T.Normalize(*stats,inplace=True)])

    from torch.utils.data import random_split

    random_seed = 43
    torch.manual_seed(random_seed)

    val_pct = 0.1


    train_ds = FlowerSegment(flower_dataset, sheet_name="train_data", transform=transform)
    valid_ds = FlowerSegment(flower_dataset, sheet_name="val_data", transform=transform)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE*2, num_workers=3, pin_memory=True)


    device = get_default_device()
    logging.info(f"Using {device} device")
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    model = to_device(ResNet9(3, 5), device)


    logging.info(model)

    epochs = 2
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    history = [evaluate(model, valid_dl)]
    print(history)
    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                                grad_clip=grad_clip, 
                                weight_decay=weight_decay, 
                                opt_func=opt_func)


    logging.info("Done!")

    dataset = ws.datasets.get("FLOWERS_MODEL")
    
    draft = dataset.drafts.create("upload_model")


    ob = pt.build_openbytes("main")

    schema = pt.record(
        {
            "model_name": pt.string(),
            "model_file": ob.file.RemoteFile(),
        }
    )
    data = []
    row_data = {
        "model_name": "flowers",
        "model_file": File("./model.pth")
    }
    data.append(row_data)

    draft["final_model"] = DataFrame(data=data, schema=schema) 
    draft.upload()
    draft.commit("uploaded flower model file")
    logging.info("Uploaded model!")