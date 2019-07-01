## SSD EfficientNet in tensorflow keras version
Unofficial implementation of SSD with [EfficientNet](https://arxiv.org/abs/1905.11946) backbone using tf keras. 

### Usage Steps
1. Modify `model_builder.py` and add **SSDEfficientNetFeatureExtractor** and **SSDEfficientNetFPNFeatureExtractor**
```python
from .efficientnet_feature_extractor import SSDEfficientNetFeatureExtractor, SSDEfficientNetFPNFeatureExtractor

SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    ...
    'ssd_efficientnet': SSDEfficientNetFeatureExtractor,
    'ssd_efficientnet_fpn': SSDEfficientNetFPNFeatureExtractor,
}
```
2. Replace `ssd.proto` file under `protos` with this one
3. Install TensorFlow object detection api: see [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
4. Train model following [official steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)
