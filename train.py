import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from model.efficientdet import EfficientDetTrain
import pytorch_lightning as pl

# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./data',
    filename='detection-{epoch:02d}-{val_loss:.2f}',
    save_top_k=10,
    save_last = True,
    verbose = True,
    mode='min')

from dataset.dataset import get_train_transforms, get_valid_transforms, zaloDataset, collate_fn

train_dataset = zaloDataset(root_path = os.path.join(os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/images')),
                    file_name = os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json'),
                    transforms=get_train_transforms())
valid_dataset = zaloDataset(root_path = os.path.join(os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/images')),
                    file_name = os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json'),
                    transforms=get_valid_transforms())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle =True, num_workers=8, collate_fn=collate_fn)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size = 10, shuffle = False, num_workers = 8, collate_fn = collate_fn)

if __name__ == '__main__':
    eff = EfficientDetTrain(model_name = 'tf_efficientdet_d0',
                            num_classes = 7)
    trainer = pl.Trainer(max_epochs = 100, gpus = -1, callbacks = [checkpoint_callback], accelerator='dp')
    trainer.fit(eff, train_loader, valid_loader)

