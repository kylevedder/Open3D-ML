"""3D ML pipelines for torch."""

from .semantic_segmentation import SemanticSegmentation
from .object_detection import ObjectDetection
from .contrastive_learning import ContrastiveLearning

__all__ = ['SemanticSegmentation', 'ObjectDetection', 'ContrastiveLearning']
