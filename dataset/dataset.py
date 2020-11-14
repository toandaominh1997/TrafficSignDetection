import os
import cv2
import pandas as  pd
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import json

import random

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

def get_train_transforms():
    return A.Compose(
        [
            # A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            # A.OneOf([
            #     A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
            #                          val_shift_limit=0.2, p=0.9),
            #     A.RandomBrightnessContrast(brightness_limit=0.2,
            #                                contrast_limit=0.2, p=0.9),
            # ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )

def collate_fn(batch):
    return tuple(zip(*batch))

class zaloDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, file_name, transforms = None):
        if os.path.exists('./data.csv') == False:
            self.df = self.load_json(file_name)
            self.df.to_csv('./data.csv')
        else:
            self.df = pd.read_csv('./data.csv')
        self.df = self.df.loc[self.df['file_name'].notnull()]
        self.root_path = root_path
        self.transforms = transforms

    @staticmethod
    def load_json(path):
        data = json.load(open(path))
        df_cate = pd.DataFrame()
        for cate in data['categories']:
            df_cate.append(cate, ignore_index=True)

        df_image = pd.DataFrame(columns = ['file_name', 'height', 'width', 'id', 'street_id'])
        for image in data['images']:
            # image['image_id'] = image['file_name'].split('.')[0]
            df_image = df_image.append(image, ignore_index = True)

        df_annotation = pd.DataFrame(columns = ['area', 'iscrowd', 'image_id', 'x', 'y', 'w', 'h', 'category_id', 'id'])

        for annot in data['annotations']:
            bbox = annot['bbox']
            x, y, w, h = bbox

            im = {'area': annot['area'], 'iscrowd': annot['iscrowd'], 'image_id': annot['image_id'], 'x': x, 'y': y, 'h': h, 'w': w, 'category_id': annot['category_id'], 'id': annot['id']}
            df_annotation = df_annotation.append(im, ignore_index=True)
        df_image['id'] = df_image['id'].astype(str)
        df_annotation['image_id'] = df_annotation['image_id'].astype(str)
        df = pd.merge(df_annotation, df_image, left_on = 'image_id', right_on = 'id', how='left')
        # df_annotation['image_id'] = df_annotation['image_id'].astype(str)
        # df_image['image_id'] = df_image['image_id'].astype(str)
        # df = pd.merge(df_annotation, df_image, on = 'image_id', how='left')
        return df
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_id = self.df.loc[index, 'image_id']
        file_name = os.path.join(self.root_path, self.df.loc[index, 'file_name'])
        image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        boxes = self.df.loc[self.df['image_id'] == image_id, ['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        labels = torch.ones((boxes.shape[0], ), dtype = torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([int(image_id)])
        if self.transforms:
            sample = self.transforms(**{
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            })
            if len(sample['bboxes']) > 0:
                image = sample['image']
                target['boxes'] = torch.from_numpy(np.array(sample['bboxes']))
                # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  #yxyx: be warning
                # target['boxes'] = target['boxes'].view(1, -1, 4)
        target_eff = {'bbox': target['boxes'], 'cls': target['labels']}
        return image, target_eff, image_id



if __name__=='__main__':
    dataset = zaloDataset(
        root_path = '/home/tonne/code/TrafficSignDetection/data/za_traffic_2020/traffic_train/images',
        file_name = '/home/tonne/code/TrafficSignDetection/data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json')
    print(dataset.__getitem__(1))
