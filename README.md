## SSD EfficientNet in tensorflow keras version
Unofficial implementation of SSD with [EfficientNet](https://arxiv.org/abs/1905.11946) backbone using tf keras. 

### UPDATE
Here is the [link](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) for official EfficientNet implementation for TPU training. 

### Install Requirement
- Python 3.X
- TensorFlow 1.4 
- TensorFlow Models master branch
- Protoc 3.5.7

### Usage Steps
1. Put `efficientnet.py` and `efficient_feature_extractor.py` under `object_detection/models` directory
2. Modify `model_builder.py` and add **SSDEfficientNetFeatureExtractor** and **SSDEfficientNetFPNFeatureExtractor**
```python
from object_detection.models.efficientnet_feature_extractor import SSDEfficientNetFeatureExtractor, SSDEfficientNetFPNFeatureExtractor

SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    ...
    'ssd_efficientnet': SSDEfficientNetFeatureExtractor,
    'ssd_efficientnet_fpn': SSDEfficientNetFPNFeatureExtractor,
}
```
3. Replace `ssd.proto` file under `protos` with this one. **Then make sure to rerun `protoc object_detection/protos/ssd.proto --python_out=.`**
4. Install TensorFlow object detection api: see [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
5. Train model following [official steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md)
6. Refer to `ssd_efficientnet,config` as example config
