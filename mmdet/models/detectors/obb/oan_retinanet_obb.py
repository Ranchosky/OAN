from mmdet.models.builder import DETECTORS
from .oan_obb_single_stage import oan_OBBSingleStageDetector


@DETECTORS.register_module()
class oan_RetinaNetOBB(oan_OBBSingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(oan_RetinaNetOBB, self).__init__(backbone, neck, bbox_head, train_cfg,
                                           test_cfg, pretrained)
