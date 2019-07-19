import sys
sys.path.append("/home/arkenstone/models/research")
import tensorflow as tf 
from object_detection.models import feature_map_generators
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.utils import ops, shape_utils
from .efficientnet import build_model_base_keras_model

l = tf.keras.layers

class SSDEfficientNetFPNFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor): 
    """ SSD Feature Extractor using EfficientNet as backbone and together with FPN structure"""
    def __init__(self, 
                 is_training,
                 depth_multiplier,
                 min_depth,
                 pad_to_multiple,
                 conv_hyperparams,
                 freeze_batchnorm,
                 inplace_batchnorm_update,
                 network_version='efficientnet-b0',
                 min_feature_level=3,
                 max_feature_level=7,
                 additional_layer_depth=256,
                 reuse_weights=None,
                 use_explicit_padding=None,
                 use_depthwise=False,
                 use_antialias=False,
                 override_base_feature_extractor_hyperparams=False,
                 name=None,
                 data_format="channels_last"):
        """EfficientNet Feature Extractor for SSD Models
        
        Arguments:
            is_training {bool} -- wether the network is in training mode
            depth_multiplier {float}} -- depth multiplier for feature extractor 
            min_depth {int32} -- minimum feature extractor depth
            pad_to_multiple {int32} -- the nearest multiple to zero pad the input height and 
                width dimensions to 
            conv_hyperparams -- A `hyperparams_builder.KerasLayerHyperparams` object 
                containing convolution hyperparameters for the layers added on top of
                the base feature extractor
            inplace_batchnorm_update --  Whether to update batch norm moving average
               values inplace. When this is false train op must add a control
               dependency on tf.graphkeys.UPDATE_OPS collection in order to update
               batch norm statistics.
        
        Keyword Arguments:
            reuse_weights -- whether to reuse variables. Default is None 
            use_explicit_padding -- whether to use depthwise convolutions. Default is False
            use_depthwise -- whether to use depthwise convolution
            override_base_feature_extractor_hyperparams -- Whether to override
               hyperparameters of the base feature extractor with the one from
               `conv_hyperparams`.
            name --  A string name scope to assign to the model. If 'None', Keras
               will auto-generate one from the class name
            data_format -- channels_last or channels_first
        """
        super(SSDEfficientNetFPNFeatureExtractor, self).__init__(
            is_training=is_training,
            depth_multiplier=depth_multiplier,
            min_depth=min_depth,
            pad_to_multiple=pad_to_multiple,
            conv_hyperparams=conv_hyperparams,
            freeze_batchnorm=freeze_batchnorm,
            inplace_batchnorm_update=inplace_batchnorm_update,
            use_explicit_padding=use_explicit_padding,
            use_depthwise=use_depthwise,
            override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
            name=name
        )
        self._data_format = data_format
        self._min_feature_level = min_feature_level
        self._max_feature_level = max_feature_level
        self._additional_layer_depth = additional_layer_depth
        self._network_name = network_version
        if network_version == "efficientnet-b0":
            default_nodes = ["block_4", "block_10", "block_15", "", "", "", ""] 
            self._backbone_layers = 15
        elif network_version == "efficientnet-b1":
            default_nodes = ["block_7", "block_15", "block_22", "", "", "", ""]
            self._backbone_layers = 22
        elif network_version == 'efficientnet-b2':
            default_nodes = ["block_7", "block_15", "block_22", "", "", "", ""]
            self._backbone_layers = 22
        elif network_version == "efficientnet-b3":
            default_nodes = ["block_7", "block_17", "block_25", "", "", "", ""]
            self._backbone_layers = 25
        elif network_version == "efficientnet-b4":
            default_nodes = ["block_9", "block_21", "block_31", "", "", "", ""]
            self._backbone_layers = 31
        elif network_version == "efficientnet-b5":
            default_nodes = ["block_12", "block_26", "block_38", "", "", "", ""]
            self._backbone_layers = 38
        elif network_version == "efficientnet-b6":
            default_nodes = ["block_14", "block_30", "block_44", "", "", "", ""]
            self._backbone_layers = 44
        elif network_version == "efficientnet-b7":
            default_nodes = ["block_17", "block_37", "block_54", "", "", "", ""]
            self._backbone_layers = 54
        else:
            raise ValueError("Unknown efficientnet name: {}".format(network_version))
        # default_nodes = ["reduction_3", "reduction_4", "reduction_5", "", "", ""]
        default_nodes_depth = [-1, -1, -1, 512, 256, 256, 128]
        self._used_nodes = default_nodes[min_feature_level-3:max_feature_level-2]
        self._used_nodes_depth = default_nodes_depth[min_feature_level-3:max_feature_level-2]
        self._feature_map_layout = {
            'from_layer': self._used_nodes,
            'layer_depth': self._used_nodes_depth,
            'use_depthwise': self._use_depthwise,
            'use_explicit_padding': self._use_explicit_padding
        }
        self._feature_map_generator = None 
        self._coarse_feature_layers = []
        self.net = None 
        
    def build(self, input_shape):
        model = build_model_base_keras_model(input_shape[1:], self._network_name, self._is_training)
        # inputs = tf.keras.layers.Input(input_shape[1:])
        inputs = model.inputs
        # _, endpoints = build_model_base(inputs, self._network_name, self._is_training)
        # outputs = [endpoints[x] for x in self._used_nodes if x]
        outputs = [model.get_layer(x).output for x in self._used_nodes if x]
        self.net = tf.keras.Model(inputs=inputs, outputs=outputs)
        # fpn feature map generator
        self._depth_fn = lambda d: max(int(d*self._depth_multiplier), self._min_depth)
        self._base_fpn_max_level = min(self._max_feature_level, 5)
        self._num_levels = self._base_fpn_max_level + 1 - self._min_feature_level
        self._feature_map_generator = feature_map_generators.KerasFpnTopDownFeatureMaps(
            num_levels=self._num_levels, 
            depth=self._depth_fn(self._additional_layer_depth), 
            use_depthwise=self._use_depthwise, 
            use_explicit_padding=self._use_explicit_padding,
            is_training=self._is_training,
            conv_hyperparams=self._conv_hyperparams,
            freeze_batchnorm=self._freeze_batchnorm,
            name='FeatureMaps'
        )
        # construct coarse feature layers 
        padding = 'VALID' if self._use_explicit_padding else 'SAME'
        kernel_size = 3 
        stride = 2 
        for i in range(self._base_fpn_max_level+1, self._max_feature_level+1):
            coarse_feature_layers = []
            if self._use_explicit_padding:
                _fixed_padding_fn = lambda x: ops.fixed_padding(x, kernel_size=kernel_size)
                coarse_feature_layers.append(l.Lambda(_fixed_padding_fn))
            layer_name = 'bottom_up_Conv2d_{}'.format(i-self._base_fpn_max_level+self._backbone_layers)
            conv_block = feature_map_generators.create_conv_block(self._use_depthwise, kernel_size, padding,
                                                                  stride, layer_name, self._conv_hyperparams, self._is_training, 
                                                                  self._freeze_batchnorm, self._depth_fn(self._additional_layer_depth))
            coarse_feature_layers.extend(conv_block)
            self._coarse_feature_layers.append(coarse_feature_layers)
        self.built = True 
        
    def preprocess(self, resized_inputs):
        """SSD preprocessing: map pixel values ot range [-1, 1]"""
        return resized_inputs * (2.0 / 255.0) - 1.0
    
    def _extract_features(self, preprocessed_inputs):
        """Extract features from preprocessed inputs"""        
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        image_features = self.net(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
        # layouts = {self._used_nodes[i]: image_features[i] for i, x in enumerate(self._used_nodes) if x}
        fpn_input_image_features = [(v, image_features[i]) for i, v in enumerate(self._used_nodes) if v]
        fpn_features = self._feature_map_generator(fpn_input_image_features)
        fpn_features = list(fpn_features.values())
        last_feature_map = fpn_features[-1]
        feature_maps = []
        feature_maps.extend(fpn_features)
        for coarse_feature_layers in self._coarse_feature_layers:
            for layer in coarse_feature_layers:
                last_feature_map = layer(last_feature_map)
            feature_maps.append(last_feature_map)
        return feature_maps 
        
        
class SSDEfficientNetFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor): 
    """ SSD Feature Extractor using EfficientNet """
    def __init__(self, 
                 is_training,
                 depth_multiplier,
                 min_depth,
                 pad_to_multiple,
                 conv_hyperparams,
                 freeze_batchnorm,
                 inplace_batchnorm_update,
                 network_version='efficientnet-b0',
                 min_feature_level=3,
                 max_feature_level=7,
                 additional_layer_depth=None,
                 reuse_weights=None,
                 use_explicit_padding=None,
                 use_depthwise=False,
                 use_antialias=False,
                 override_base_feature_extractor_hyperparams=False,
                 name=None,
                 data_format="channels_last"):
        """EfficientNet Feature Extractor for SSD Models
        
        Arguments:
            is_training {bool} -- wether the network is in training mode
            depth_multiplier {float}} -- depth multiplier for feature extractor 
            min_depth {int32} -- minimum feature extractor depth
            pad_to_multiple {int32} -- the nearest multiple to zero pad the input height and 
                width dimensions to 
            conv_hyperparams -- A `hyperparams_builder.KerasLayerHyperparams` object 
                containing convolution hyperparameters for the layers added on top of
                the base feature extractor
            inplace_batchnorm_update --  Whether to update batch norm moving average
               values inplace. When this is false train op must add a control
               dependency on tf.graphkeys.UPDATE_OPS collection in order to update
               batch norm statistics.
        
        Keyword Arguments:
            reuse_weights -- whether to reuse variables. Default is None 
            use_explicit_padding -- whether to use depthwise convolutions. Default is False
            use_depthwise -- whether to use depthwise convolution
            override_base_feature_extractor_hyperparams -- Whether to override
               hyperparameters of the base feature extractor with the one from
               `conv_hyperparams`.
            name --  A string name scope to assign to the model. If 'None', Keras
               will auto-generate one from the class name
            data_format -- channels_last or channels_first
        """
        super(SSDEfficientNetFeatureExtractor, self).__init__(
            is_training=is_training,
            depth_multiplier=depth_multiplier,
            min_depth=min_depth,
            pad_to_multiple=pad_to_multiple,
            conv_hyperparams=conv_hyperparams,
            freeze_batchnorm=freeze_batchnorm,
            inplace_batchnorm_update=inplace_batchnorm_update,
            use_explicit_padding=use_explicit_padding,
            use_depthwise=use_depthwise,
            override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams,
            name=name
        )
        self._data_format = data_format
        self._min_feature_level = min_feature_level
        self._max_feature_level = max_feature_level
        self._additional_layer_depth = additional_layer_depth
        self._network_name = network_version
        if network_version == "efficientnet-b0":
            default_nodes = ["block_4", "block_10", "block_15", "", "", "", ""] 
        elif network_version == "efficientnet-b1":
            default_nodes = ["block_7", "block_15", "block_22", "", "", "", ""]
        elif network_version == 'efficientnet-b2':
            default_nodes = ["block_7", "block_15", "block_22", "", "", "", ""]
        elif network_version == "efficientnet-b3":
            default_nodes = ["block_7", "block_17", "block_25", "", "", "", ""]
        elif network_version == "efficientnet-b4":
            default_nodes = ["block_9", "block_21", "block_31", "", "", "", ""]
        elif network_version == "efficientnet-b5":
            default_nodes = ["block_12", "block_26", "block_38", "", "", "", ""]
        elif network_version == "efficientnet-b6":
            default_nodes = ["block_14", "block_30", "block_44", "", "", "", ""]
        elif network_version == "efficientnet-b7":
            default_nodes = ["block_17", "block_37", "block_54", "", "", "", ""]
        else:
            raise ValueError("Unknown efficientnet name: {}".format(network_version))
        # default_nodes = ["reduction_3", "reduction_4", "reduction_5", "", "", ""]
        default_nodes_depth = [-1, -1, -1, 512, 256, 256, 128]
        self._used_nodes = default_nodes[min_feature_level-3:max_feature_level-2]
        self._used_nodes_depth = default_nodes_depth[min_feature_level-3:max_feature_level-2]
        self._feature_map_layout = {
            'from_layer': self._used_nodes,
            'layer_depth': self._used_nodes_depth,
            'use_depthwise': self._use_depthwise,
            'use_explicit_padding': self._use_explicit_padding
        }
        self._feature_map_generator = None 
        self.net = None 
        
    def build(self, input_shape):
        model = build_model_base_keras_model(input_shape[1:], self._network_name, self._is_training)
        # inputs = tf.keras.layers.Input(input_shape[1:])
        inputs = model.inputs
        # _, endpoints = build_model_base(inputs, self._network_name, self._is_training)
        # outputs = [endpoints[x] for x in self._used_nodes if x]
        outputs = [model.get_layer(x).output for x in self._used_nodes if x]
        self.net = tf.keras.Model(inputs=inputs, outputs=outputs)
        # feature map generator
        self._feature_map_generator = feature_map_generators.KerasMultiResolutionFeatureMaps(
            feature_map_layout=self._feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True, 
            is_training=self._is_training,
            conv_hyperparams=self._conv_hyperparams,
            freeze_batchnorm=self._freeze_batchnorm,
            name=None
        )
        self.built = True 
        
    def preprocess(self, resized_inputs):
        """SSD preprocessing: map pixel values ot range [-1, 1]"""
        return resized_inputs * (2.0 / 255.0) - 1.0
    
    def _extract_features(self, preprocessed_inputs):
        """Extract features from preprocessed inputs"""        
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        image_features = self.net(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
        layouts = {self._used_nodes[i]: image_features[i] for i, x in enumerate(self._used_nodes) if x}
        feature_maps = self._feature_map_generator(layouts)
        if self._additional_layer_depth:
            final_feature_map = []
            for idx, feature in enumerate(feature_maps.values()):
                feature = l.Conv2D(filters=self._additional_layer_depth,
                                    kernel_size=1,
                                    strides=[1, 1],
                                    use_bias=True,
                                    data_format=self._data_format,
                                    name='conv1x1_'+str(idx))(feature)
                feature = l.BatchNormalization()(feature, training=self._is_training)
                feature = l.ReLU(max_value=6)(feature)
                final_feature_map.append(feature)
            return final_feature_map
        else:
            return list(feature_maps.values())
