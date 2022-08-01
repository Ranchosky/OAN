from mmdet.models.builder import DETECTORS
from .oan_obb_two_stage import oan_OBBTwoStageDetector


@DETECTORS.register_module()
class oan_RoITransformer(oan_OBBTwoStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(oan_RoITransformer, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
