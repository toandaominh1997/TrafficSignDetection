import os

import pytorch_lightning as pl
import torch
from models.efficientdet import EfficientDetTrain
import pytorch_lightning as pl

from dataset.dataset import get_train_transforms, zaloDataset, collate_fn

train_dataset = zaloDataset(root_path = os.path.join(os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/images')),
                    file_name = os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json'),
                    transforms=get_train_transforms())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle =True, num_workers=8, collate_fn=collate_fn)


if __name__ == '__main__':
    eff = EfficientDetTrain(model_name = 'tf_efficientdet_d0',
                            num_classes = 1)
    trainer = pl.Trainer(max_epochs = 100, gpus = 1)
    trainer.fit(eff, train_loader, train_loader)

