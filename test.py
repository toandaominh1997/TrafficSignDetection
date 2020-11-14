import sys
sys.path.append('/home/tonne/code/TrafficSignDetection/data/thirdparty/efficientdet-pytorch/')
from typing import Optional, Dict, List
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model.efficientdet import EfficientDetTrain

from dataset.dataset import get_train_transforms, zaloDataset, collate_fn, zaloDatasetInfer, get_valid_transforms


from effdet.anchors import Anchors, AnchorLabeler, generate_detections, MAX_DETECTION_POINTS


def _post_process(
        cls_outputs: List[torch.Tensor],
        box_outputs: List[torch.Tensor],
        num_levels: int,
        num_classes: int,
        max_detection_points: int = MAX_DETECTION_POINTS,
):
    """Selects top-k predictions.
    Post-proc code adapted from Tensorflow version at: https://github.com/google/automl/tree/master/efficientdet
    and optimized for PyTorch.
    Args:
        cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width, num_anchors].
        box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width, num_anchors * 4].
        num_levels (int): number of feature levels
        num_classes (int): number of output classes
    """
    batch_size = cls_outputs[0].shape[0]
    cls_outputs_all = torch.cat([
        cls_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, num_classes])
        for level in range(num_levels)], 1)

    box_outputs_all = torch.cat([
        box_outputs[level].permute(0, 2, 3, 1).reshape([batch_size, -1, 4])
        for level in range(num_levels)], 1)

    _, cls_topk_indices_all = torch.topk(cls_outputs_all.reshape(batch_size, -1), dim=1, k=max_detection_points)
    indices_all = cls_topk_indices_all // num_classes
    classes_all = cls_topk_indices_all % num_classes

    box_outputs_all_after_topk = torch.gather(
        box_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, 4))

    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all, 1, indices_all.unsqueeze(2).expand(-1, -1, num_classes))
    cls_outputs_all_after_topk = torch.gather(
        cls_outputs_all_after_topk, 2, classes_all.unsqueeze(2))

    return cls_outputs_all_after_topk, box_outputs_all_after_topk, indices_all, classes_all
@torch.jit.script
def _batch_detection(
        batch_size: int, class_out, box_out, anchor_boxes, indices, classes,
        img_scale: Optional[torch.Tensor] = None, img_size: Optional[torch.Tensor] = None):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        img_scale_i = None if img_scale is None else img_scale[i]
        img_size_i = None if img_size is None else img_size[i]
        detections = generate_detections(
            class_out[i], box_out[i], anchor_boxes, indices[i], classes[i], img_scale_i, img_size_i)
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)

test_dataset = zaloDatasetInfer(root_path = "/home/tonne/code/TrafficSignDetection/data/za_traffic_2020/traffic_train/images",
                    transforms=get_valid_transforms())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 2, shuffle =False, num_workers=8, collate_fn=collate_fn)
model = EfficientDetTrain(model_name = 'tf_efficientdet_d0',
                          num_classes=1)
model.load_from_checkpoint('/home/tonne/code/TrafficSignDetection/lightning_logs/version_1/checkpoints/epoch=24.ckpt',
                           model_name = 'tf_efficientdet_d0', num_classes = 1)
for batch in test_loader:
    x, idx = batch
    x = torch.stack(x, dim = 0)
    class_out, box_out = model(x)
    class_out, box_out, indices, classes = _post_process(
            class_out, box_out, num_levels=model.num_levels, num_classes=model.num_classes)
    img_info = None
    if img_info is None:
        img_scale, img_size = None, None
    else:
        img_scale, img_size = img_info['img_scale'], img_info['img_size']
    output =  _batch_detection(x.shape[0], class_out, box_out, model.anchors.boxes, indices, classes, img_scale, img_size)

    print('output: ', output.shape)
    visualization = True

    if visualization:
        score_threshold = 0.2
        for i in range(len(output)):
            image = x[i].permute(1, 2, 0).cpu().numpy()
            classes = output[i].detach().cpu().numpy()[:, -1]
            scores = output[i].detach().cpu().numpy()[:, 4]
            print('scores: ', scores)
            indexes = np.where(scores > score_threshold)[0]
            boxes = output[i].detach().cpu().numpy()[:, :4][indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            for box in boxes:
                print('box: ', box)
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 1)
            plt.figure(figsize=(12, 12))
            plt.axis('off')
            plt.imshow(image)
            plt.show()


    break


