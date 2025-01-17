# coding=utf-8
# Copyright 2018-2022 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
import yolov5
import numpy as np
import pandas as pd

from eva.models.catalog.frame_info import FrameInfo
from eva.models.catalog.properties import ColorSpace
from eva.udfs.pytorch_abstract_udf import PytorchAbstractClassifierUDF

try:
    import torch
    from torch import Tensor
except ImportError as e:
    raise ImportError(
        f"Failed to import with error {e}, \
        please try `pip install torch`"
    )

try:
    import torchvision
except ImportError as e:
    raise ImportError(
        f"Failed to import with error {e}, \
        please try `pip install torch`"
    )


class YoloV5(PytorchAbstractClassifierUDF):
    """
    Arguments:
        threshold (float): Threshold for classifier confidence score

    """

    @property
    def name(self) -> str:
        return "yolov5"

    def __init__(self, threshold=0.85):
        super().__init__()
        self.threshold = threshold
        self.model = yolov5.load('yolov5s.pt')
        self.model.eval()

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return [
                'person',
                'bicycle',
                'car',
                'motorcycle',
                'airplane',
                'bus',
                'train',
                'truck',
                'boat',
                'traffic light',
                'fire hydrant',
                'stop sign',
                'parking meter',
                'bench',
                'bird',
                'cat',
                'dog',
                'horse',
                'sheep',
                'cow',
                'elephant',
                'bear',
                'zebra',
                'giraffe',
                'backpack',
                'umbrella',
                'handbag',
                'tie',
                'suitcase',
                'frisbee',
                'skis',
                'snowboard',
                'sports ball',
                'kite',
                'baseball bat',
                'baseball glove',
                'skateboard',
                'surfboard',
                'tennis racket',
                'bottle',
                'wine glass',
                'cup',
                'fork',
                'knife',
                'spoon',
                'bowl',
                'banana',
                'apple',
                'sandwich',
                'orange',
                'broccoli',
                'carrot',
                'hot dog',
                'pizza',
                'donut',
                'cake',
                'chair',
                'couch',
                'potted plant',
                'bed',
                'dining table',
                'toilet',
                'tv',
                'laptop',
                'mouse',
                'remote',
                'keyboard',
                'cell phone',
                'microwave',
                'oven',
                'toaster',
                'sink',
                'refrigerator',
                'book',
                'clock',
                'vase',
                'scissors',
                'teddy bear',
                'hair drier',
                'toothbrush',
                ]


    def _get_predictions(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            tuple containing predicted_classes (List[List[str]]),
            predicted_boxes (List[List[BoundingBox]]),
            predicted_scores (List[List[float]])

        """
        # Cast hacky fix
        transform = torchvision.transforms.ToPILImage()
        img = transform(frames[0])
        predictions = self.model(img)
        outcome = pd.DataFrame()
        predictions = predictions.pandas().xyxy        
        
        for prediction in predictions:
            pred_class = [self.labels[class_index] for class_index in prediction['class'].tolist()]
            pred_boxes = [
                [l[1]['xmin'],l[1]['ymin'],l[1]['xmax'],l[1]['ymax']]
                for l in prediction.iterrows()
            ]
            pred_score = prediction['confidence'].tolist()
            valid_pred = [pred_score.index(x) for x in pred_score if x > self.threshold]

            if valid_pred:
                pred_t = valid_pred[-1]
            else:
                pred_t = -1

            pred_boxes = np.array(pred_boxes[: pred_t + 1])
            pred_class = np.array(pred_class[: pred_t + 1])
            pred_score = np.array(pred_score[: pred_t + 1])
            outcome = outcome.append(
                {"labels": pred_class, "scores": pred_score, "bboxes": pred_boxes},
                ignore_index=True,
            )
            # open('/nethome/gkakkar7/rbachkaniwala3/rajveer_fork/eva/fastrcnnlog.txt','a+').write(outcome.to_string())
        return outcome
