import os
import sys
sys.path.append('/home/tonne/code/TrafficSignDetection/data/thirdparty/efficientdet-pytorch/')
import torch

import torch
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator
from effdet.efficientdet import EfficientDet, HeadNet
from effdet.config import get_efficientdet_config
from effdet.anchors import Anchors, AnchorLabeler
from effdet.loss import DetectionLoss
from effdet.data import resolve_input_config, SkipSubset

from dataset.dataset import zaloDataset, get_train_transforms

# model_name = 'tf_efficientdet_d1'
# config = get_efficientdet_config(model_name)
# model = EfficientDet(config)
# inputs  = torch.randn(3, 3, 512, 512)
# out = model(inputs)
# print('out: ', out)


import pytorch_lightning as pl

train_dataset = zaloDataset(root_path = os.path.join(os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/images')),
                    file_name = os.path.join(os.getcwd(), 'data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json'),
                    transforms=get_train_transforms())
def collate_fn(batch):
    return tuple(zip(*batch))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 12, shuffle =True, num_workers=8, collate_fn=collate_fn)

class EfficientDetTrain(pl.LightningModule):
    def __init__(self, model_name, num_classes, create_labeler=True):
        super().__init__()
        config = get_efficientdet_config(model_name)

        self.model = EfficientDet(config)
        self.model.reset_head(num_classes=num_classes)
        self.num_levels = self.model.config.num_levels
        self.num_classes = self.model.config.num_classes
        self.anchors = Anchors.from_config(self.model.config)
        self.anchor_labeler = None
        if create_labeler:
            self.anchor_labeler = AnchorLabeler(self.anchors, self.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(self.model.config)

    def forward(self, x):
        class_out, box_out = self.model(x)
        return class_out, box_out
    def training_step(self, batch, batch_idx):
        x, targets, idx = batch
        x = torch.stack(x, dim = 0)
        class_out, box_out = self.forward(x)

        # target['bbox'] = torch.stack(target['bbox'], dim = 1)
        # target['cls'] = torch.stack(target['cls'], dim = 1)

        bbox = [tar['bbox'].float() for tar in targets]
        clses = [tar['cls'].float() for tar in targets]
        target = {}
        target['bbox'] = bbox
        target['cls'] = clses
        if self.anchor_labeler is None:
            # target should contain pre-computed anchor labels if labeler not present in bench
            assert 'label_num_positives' in target
            cls_targets = [target[f'label_cls_{l}'] for l in range(self.num_levels)]
            box_targets = [target[f'label_bbox_{l}'] for l in range(self.num_levels)]
            num_positives = target['label_num_positives']
        else:
            cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
                target['bbox'], target['cls'])

        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
        return loss + class_loss + box_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

eff = EfficientDetTrain(model_name = 'tf_efficientdet_d0',
                        num_classes = 1)
trainer = pl.Trainer(max_epochs = 1, gpus = 1)
trainer.fit(eff, train_loader, train_loader)

