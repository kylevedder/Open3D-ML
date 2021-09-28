#***************************************************************************************/
#
#    Based on MMDetection3D Library (Apache 2.0 license):
#    https://github.com/open-mmlab/mmdetection3d
#
#    Copyright 2018-2019 Open-MMLab.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#***************************************************************************************/

import torch
import pickle
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from functools import partial
import numpy as np
import os

from open3d.ml.torch.ops import voxelize, ragged_to_dense

from .base_model_objdet import BaseModel
from .point_pillars import PointPillars, PointPillarsScatter, PointPillarsVoxelization, PillarFeatureNet, SECOND, SECONDFPN, Anchor3DHead

from ...utils import MODEL
from ..utils.objdet_helper import Anchor3DRangeGenerator, BBoxCoder, multiclass_nms, limit_period, get_paddings_indicator, bbox_overlaps, box3d_to_bev2d
from ..modules.losses.focal_loss import FocalLoss
from ..modules.losses.smooth_L1 import SmoothL1Loss
from ..modules.losses.cross_entropy import CrossEntropyLoss
from ...datasets.utils import ObjdetAugmentation, BEVBox3D
from ...datasets.utils.operations import filter_by_min_points


class ContrastivePointPillars(PointPillars):
    """Object detection model. Based on the PointPillars architecture
    https://github.com/nutonomy/second.pytorch.

    Args:
        name (string): Name of model.
            Default to "ContrastivePointPillars".
        voxel_size: voxel edge lengths with format [x, y, z].
        point_cloud_range: The valid range of point coordinates as
            [x_min, y_min, z_min, x_max, y_max, z_max].
        voxelize: Config of PointPillarsVoxelization module.
        voxelize_encoder: Config of PillarFeatureNet module.
        scatter: Config of PointPillarsScatter module.
        backbone: Config of backbone module (SECOND).
        neck: Config of neck module (SECONDFPN).
        head: Config of anchor head module.
    """

    def __init__(self,
                 name="PointPillars",
                 device="cuda",
                 point_cloud_range=[0, -40.0, -3, 70.0, 40.0, 1],
                 voxelize={},
                 voxel_encoder={},
                 scatter={},
                 backbone={},
                 neck={},
                 slots={},
                 loss={},
                 **kwargs):

        super().__init__(name=name,
                         device=device,
                         point_cloud_range=point_cloud_range,
                         voxelize=voxelize,
                         voxel_encoder=voxel_encoder,
                         scatter=scatter,
                         backbone=backbone,
                         neck=neck,
                         loss=loss,
                         **kwargs)
        # Remove the anchor generating head.
        self.bbox_head = None
        del self.bbox_head

        # Add slot head.
        self.slot_head = SlotHead(**slots)
        self.to(device)

    def get_weighted_centroids(self, slots):
        """
        Given a Batch x Slots x Width x Height `slots`, compute the weighted centroid per slot.

        Returns a Batch x Slots x 2 tensor.
        """
        idxs_x = torch.arange(0, slots.shape[2], device=slots.device)
        vals_x = torch.sum(slots, axis=3)
        totals_x = torch.unsqueeze(torch.sum(vals_x, axis=2), axis=2)
        normed_vals_x = vals_x / totals_x
        centoids_x = torch.sum(idxs_x * normed_vals_x, axis=2)

        idxs_y = torch.arange(0, slots.shape[3], device=slots.device)
        vals_y = torch.sum(slots, axis=2)
        totals_y = torch.unsqueeze(torch.sum(vals_y, axis=2), axis=2)
        normed_vals_y = vals_y / totals_y
        centoids_y = torch.sum(idxs_y * normed_vals_y, axis=2)

        return torch.cat([
            torch.unsqueeze(centoids_x, axis=2),
            torch.unsqueeze(centoids_y, axis=2)
        ],
                         axis=2)

    def loss(self, results, inputs):
        results_inputs, results_pairs, results_negatives = results
        input_centroids = self.get_weighted_centroids(results_inputs)
        pair_centroids = self.get_weighted_centroids(results_pairs)
        negative_centroids = [
            self.get_weighted_centroids(n) for n in results_negatives
        ]

        # input_centroids.shape is Batch x Slots x 2
        # pair_centroids.shape is Batch x Slots x 2
        # negative_centroids.shape is list of len Batch, shape of NumNeg x Slots x 2
        slot_dist = nn.L1Loss()

        input_pair_dist = slot_dist(input_centroids, pair_centroids)

        neg_dist = 0
        for input_cent, neg_cents in zip(input_centroids, negative_centroids):
            num_negs = neg_cents.shape[0]
            input_cent = torch.unsqueeze(input_cent, 0)
            input_cents_repeated = input_cent.repeat(num_negs, 1, 1)
            neg_dist += slot_dist(input_cents_repeated, neg_cents)

        contrastive_loss = torch.exp(input_pair_dist / neg_dist)

        return {'loss_contrast': contrastive_loss}

    def preprocess(self, data, attr):
        if attr['split'] not in ['test', 'testing', 'val', 'validation']:
            data = self.augment_data(data, attr)

        return data

    def augment_data(self, data, attr):
        cfg = self.cfg.augment
        if cfg.get('PointShuffle', False):
            data = ObjdetAugmentation.PointShuffle(data, 'input')
            data = ObjdetAugmentation.PointShuffle(data, 'pair')
        return data

    def transform(self, data, attr):
        return data

    def forward(self, data_dict):
        inputs = data_dict.inputs
        pairs = data_dict.pairs
        negatives = data_dict.negatives

        inputs_outs = self.slot_head(self.extract_feats(inputs))
        pairs_outs = self.slot_head(self.extract_feats(pairs))
        negative_outs = [
            self.slot_head(self.extract_feats(n)) for n in negatives
        ]

        return inputs_outs, pairs_outs, negative_outs


MODEL._register_module(ContrastivePointPillars, 'torch')


class SlotHead(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self, out_channels, in_channels=384):
        super(SlotHead, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01), nn.Sigmoid())

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Input with shape (N, C, H, W).
        """
        return self.block(x)
