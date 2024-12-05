# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList, DetDataSample
from mmengine.structures import InstanceData
from ..layers import (DetrTransformerDecoder, DetrTransformerEncoder,
                      SinePositionalEncoding)
from .base_detr import DetectionTransformer
from .base import BaseDetector

from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector

# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from torch import Tensor


@MODELS.register_module()
class DFINE(BaseDetector):
    def __init__(self, backbone: ConfigType, 
                 decoder: OptConfigType = None, 
                 encoder: OptConfigType = None, 
                 criterion: OptConfigType = None, 
                 init_cfg: OptMultiConfig = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        
        self.backbone = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.encoder = MODELS.build(encoder)
        self.criterion = MODELS.build(criterion)
        self.test_cfg = test_cfg
        # self._init_layers()
        
    # @abstractmethod
    # def _init_layers(self) -> None:
    #     """Initialize layers except for backbone, neck and bbox_head."""
    #     pass  
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        
        data = self._forward(batch_inputs, batch_data_samples)
        
        det_sample = []
        for data_sample in batch_data_samples:
            sample = self.detsample_to_dict(data_sample)
            det_sample.append(sample)
            
        loss = self.criterion(data, det_sample)
        
        return loss
        
        # raise NotImplementedError
        # img_feats = self.extract_feat(batch_inputs)
        # head_inputs_dict = self.forward_transformer(img_feats,
        #                                             batch_data_samples)
        # losses = self.bbox_head.loss(
        #     **head_inputs_dict, batch_data_samples=batch_data_samples)

        # return losses  
       
    def get_instance_data_list(self, data: dict, batch_data_samples: SampleList) -> DetDataSample:
        pred_logits = data.get('pred_logits', None)
        pred_boxes = data.get('pred_boxes', None)
        pred_cls = torch.argmax(pred_logits, 2, True)
        pred_scores, pred_idx = torch.max(torch.softmax(pred_logits, 2), 2)
        
        det_samples = []
        det_instances = []
        
        # bs = len(batch_data_samples)
        for i, batch_data_sample in enumerate(batch_data_samples):
            # Создаем объект DetDataSample
            det_sample = DetDataSample()
            
            # Создаем объект gt_instances из InstanceData
            pred_instances = InstanceData()
            pred_instances.scores = pred_scores[i]
            pred_instances.labels = pred_cls[i].flatten()
            pred_instances.bboxes = pred_boxes[i]
            
            # TODO: Проверить это место!!!
            w, h = batch_data_sample.ori_shape
            # Извлекаем боксы
            boxes = pred_instances.bboxes

            # Обратное преобразование: [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
            pred_instances.bboxes = torch.cat([
                boxes[:, :2] - boxes[:, 2:] / 2,  # Левый верхний угол [x_min, y_min]
                boxes[:, :2] + boxes[:, 2:] / 2   # Правый нижний угол [x_max, y_max]
            ], dim=1)

            # Масштабируем обратно (домножаем на 640)
            pred_instances.bboxes[:, 0::2] *= h  # x-координаты
            pred_instances.bboxes[:, 1::2] *= w  # y-координаты

            det_instances.append(pred_instances)
                     
        return det_instances
        
        
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
           
        x = self.backbone(batch_inputs)
        x = self.encoder(x)
        
        det_sample = []
        for data_sample in batch_data_samples:
            sample = self.detsample_to_dict(data_sample)
            det_sample.append(sample)
        
        x = self.decoder(x)
        
        # x = self.dict_to_detsamle_pred(x, batch_data_samples)
        
        
        results_list = self.get_instance_data_list(x, batch_data_samples)
        
        batch_data_samples = self.add_pred_to_datasample(
                                                batch_data_samples, results_list)
        
        return batch_data_samples
    
        # Returns:
        # list[:obj:`DetDataSample`]: Detection results of the input images.
        # Each DetDataSample usually contain 'pred_instances'. And the
        # `pred_instances` usually contains following keys.

        # - scores (Tensor): Classification scores, has a shape
        #   (num_instance, )
        # - labels (Tensor): Labels of bboxes, has a shape
        #   (num_instances, ).
        # - bboxes (Tensor): Has a shape (num_instances, 4),
        #   the last dimension 4 arrange as (x1, y1, x2, y2).
    
        # return  
        
    def dict_to_detsamle_pred(self, data: dict, batch_data_samples: SampleList) -> DetDataSample:
        pred_logits = data.get('pred_logits', None)
        pred_boxes = data.get('pred_boxes', None)
        pred_cls = torch.argmax(pred_logits, 2, True)
        pred_scores, pred_idx = torch.max(torch.softmax(pred_logits, 2), 2)
        
        det_samples = []
        
        # bs = len(batch_data_samples)
        for i, batch_data_sample in enumerate(batch_data_samples):
            # Создаем объект DetDataSample
            det_sample = DetDataSample()
            
            # Создаем объект gt_instances из InstanceData
            pred_instances = InstanceData()
            pred_instances.scores = pred_scores[i]
            pred_instances.labels = pred_cls[i].flatten()
            pred_instances.bboxes = pred_boxes[i]
            
            # TODO: Проверить это место!!!
            w, h = batch_data_sample.ori_shape
            # Извлекаем боксы
            boxes = pred_instances.bboxes

            # # Обратное преобразование: [cx, cy, w, h] -> [x_min, y_min, x_max, y_max]
            # pred_instances.bboxes = torch.cat([
            #     boxes[:, :2] - boxes[:, 2:] / 2,  # Левый верхний угол [x_min, y_min]
            #     boxes[:, :2] + boxes[:, 2:] / 2   # Правый нижний угол [x_max, y_max]
            # ], dim=1)

            # Масштабируем обратно (домножаем на 640)
            pred_instances.bboxes[:, 0::2] *= 640  # x-координаты
            pred_instances.bboxes[:, 1::2] *= 640  # y-координаты

            
            # pred_instances.img_id = batch_data_sample.img_id
            
            det_sample.pred_instances = pred_instances
                
            det_sample.set_metainfo({
                'img_id': batch_data_sample.img_id,
                'ori_shape': batch_data_sample.ori_shape,
                'pad_shape': batch_data_sample.pad_shape,
                'img_shape': batch_data_sample.img_shape,   
                'instances': batch_data_sample.gt_instances
                # 'image_id': data.get('image_id', None).item(),
                # 'orig_size': tuple(data.get('orig_size', None).tolist()),
                # 'idx': data.get('idx', None).item()
            })
            det_samples.append(det_sample)
            
        return det_samples
        
    def dict_to_detsample(self, data: dict) -> DetDataSample:
        """
        Преобразует словарь в экземпляр класса DetDataSample.

        Args:
            data (dict): Словарь с данными (например, из вашего примера).

        Returns:
            DetDataSample: Экземпляр DetDataSample с заполненными полями.
        """
        # Создаем объект DetDataSample
        det_sample = DetDataSample()

        # Создаем объект gt_instances из InstanceData
        gt_instances = InstanceData()
        gt_instances.bboxes = data.get('boxes', None)
        gt_instances.labels = data.get('labels', None)
        gt_instances.areas = data.get('area', None)
        gt_instances.iscrowd = data.get('iscrowd', None)

        # Привязываем gt_instances к DetDataSample
        det_sample.gt_instances = gt_instances

        # Добавляем метаинформацию, если необходимо
        det_sample.set_metainfo({
            'image_id': data.get('image_id', None).item(),
            'orig_size': tuple(data.get('orig_size', None).tolist()),
            'idx': data.get('idx', None).item()
        })

        return det_sample
        
    def detsample_to_dict(self, det_sample: DetDataSample) -> dict:
        """
        Преобразует экземпляр DetDataSample обратно в словарь.

        Args:
            det_sample (DetDataSample): Экземпляр DetDataSample.

        Returns:
            dict: Словарь с данными.
        """
        # Получаем gt_instances
        gt_instances = det_sample.gt_instances

        # Извлекаем метаинформацию
        meta_info = det_sample.metainfo

        # Создаем словарь с нужной структурой
        data_dict = {
            'boxes': gt_instances.bboxes if hasattr(gt_instances, 'bboxes') else None,
            'labels': gt_instances.labels if hasattr(gt_instances, 'labels') else None,
            # 'area': gt_instances.areas if hasattr(gt_instances, 'areas') else torch.tensor(10000),
            # 'iscrowd': gt_instances.iscrowd if hasattr(gt_instances, 'iscrowd') else torch.empty(0),
            'image_id': torch.tensor([meta_info.get('image_id', 0)]),
            'orig_size': torch.tensor(meta_info.get('ori_shape', None)),
            # 'idx': torch.tensor([meta_info.get('idx', 0)])
        }
        
        # TODO: Проверить это место!!!
        h, w = data_dict['orig_size']
        
        boxes = data_dict['boxes']
        data_dict['boxes'] = torch.cat([
            (boxes[:, :2] + boxes[:, 2:]) / 2,  # Центры [cx, cy]
            boxes[:, 2:] - boxes[:, :2]        # Ширина и высота [w, h]
        ], dim=1)

        # # Обновляем координаты боксов
        # data_dict['boxes'][:, :2] += data_dict['boxes'][:, 2:] / 2
        data_dict['boxes'][:, 0::2] /= 640 #w
        data_dict['boxes'][:, 1::2] /= 640 #h
        
        # x_, y_, w_, h_ = data_dict['boxes']
        # cx = x_ + w_ / 2
        # cy = y_ + h_ / 2
        # data_dict['boxes'][0] = cx
        # data_dict['boxes'][1] = cy
        # data_dict['boxes'][2] = w_
        # data_dict['boxes'][3] = h_
        # data_dict['boxes'][:, 0::2] /= w
        # data_dict['boxes'][:, 1::2] /= h

        return data_dict
        
    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        
        x = self.backbone(batch_inputs)
        x = self.encoder(x)
        
        det_sample = []
        for data_sample in batch_data_samples:
            sample = self.detsample_to_dict(data_sample)
            det_sample.append(sample)
        
        x = self.decoder(x, det_sample)
        
        return x
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        raise NotImplementedError
        # return batch_inputs
        
