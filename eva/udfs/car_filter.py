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

import numpy as np
import pandas as pd

from eva.models.catalog.frame_info import FrameInfo
from eva.models.catalog.properties import ColorSpace
from eva.udfs.pytorch_abstract_udf import PytorchAbstractClassifierUDF

try:
    from torch import Tensor
    import torch
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


class CarFilter(PytorchAbstractClassifierUDF):
    """
        @Rajveer
    """

    @property
    def name(self) -> str:
        return "car_filter"

    def __init__(self, threshold=0.85):
        super().__init__()
        self.threshold = threshold
        self.model = torchvision.models.mobilenet_v3_large(num_classes=2)
        self.model = torch.nn.DataParallel(self.model).cuda()
        device = torch.device('cpu')
        # @Rajveer hard coded path
        checkpoint = torch.load('/nethome/gkakkar7/rbachkaniwala3/dev_rajveer/model_best.pth.tar',map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return [
            # @Rajveer ask about __background__
            "car",
            "not_car",
        ]

    def _get_predictions(self, frames: Tensor) -> pd.DataFrame:
        """
        Performs predictions on input frames
        Arguments:
            frames (np.ndarray): Frames on which predictions need
            to be performed

        Returns:
            @Rajveer

        """
        predictions = self.model(frames)
        # print(frames[0].shape)
        # from torchvision.utils import save_image
        # cv2.imwrite(frames.cpu().numpy(),"/nethome/gkakkar7/rbachkaniwala3/dev_rajveer/frame1.jpg")
        # save_image(frames[0].cpu(),"/nethome/gkakkar7/rbachkaniwala3/dev_rajveer/frame10.jpg")
        # exit()
        outcome = pd.DataFrame()
        for prediction in predictions:
            probabilities = torch.nn.functional.softmax(prediction,dim=0)
            pred_score = tuple(probabilities.cpu().detach().numpy())
            pred_class = ['car','not_car']            
            print(pred_score)
            valid_pred = [pred_score.index(x) for x in pred_score if x > .6]
            print(valid_pred)
            if valid_pred:
                pred_t = valid_pred[-1]
            else:
                pred_t = -1
            # print(f'pred_t: {pred_t}')

            pred_class = np.array(pred_class[: pred_t + 1])
            # print(f'pred_class: {pred_class}')
            pred_score = np.array(pred_score[: pred_t + 1])
            # print(f'pred_score: {pred_score}')
            outcome = outcome.append(
                {"labels": pred_class, "scores": pred_score},
                ignore_index=True,
            )
            print(outcome)
        return outcome
