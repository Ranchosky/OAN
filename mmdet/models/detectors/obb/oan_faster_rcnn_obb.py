from mmdet.models.builder import DETECTORS
from .oan_obb_two_stage import oan_OBBTwoStageDetector


@DETECTORS.register_module()
class oan_FasterRCNNOBB(oan_OBBTwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 oan,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(oan_FasterRCNNOBB, self).__init__(
            backbone=backbone,
            oan=oan,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
