from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from cabra.environment.agent.torch_utils import init_layer
from cabra.environment.config import CNNLayerConfig, FullyConnectedLayerConfig
from cabra.environment.data_structure import NetworkType


def get_hidden_layers_units(hidden_units: Union[int, List[int]]) -> List[int]:
    if isinstance(hidden_units, list):
        return [unit for unit in hidden_units]
    else:
        return [hidden_units]


ACTIVATION_MAPPING = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'identity': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'elu': nn.ELU
}


def fully_connected(
        input_size: int,
        output_size: int,
        hidden_units: Union[int, List[int]],
        activation='relu',
        batch_normalization=False,
        add_output_layer: bool = True,
) -> Tuple[nn.Module, int]:
    hidden_layers_units: List[int] = get_hidden_layers_units(hidden_units)
    layers = []
    previous_layer = input_size
    last_units = input_size
    for units in hidden_layers_units:
        layers.append(nn.Linear(previous_layer, units))
        if batch_normalization:
            layers.append(nn.BatchNorm1d(units))
        layers.append(ACTIVATION_MAPPING[activation]())
        previous_layer = units
        last_units = units

    if add_output_layer:
        layers.append(nn.Linear(last_units, output_size))

    return nn.Sequential(*layers), last_units


def fully_connected_variable(
        input_size: int,
        output_size: int,
        hidden_layers: int = 2,
        activation='relu',
        batch_normalization=False,
        add_output_layer: bool = True,
) -> Tuple[nn.Module, int]:
    hidden_layers = hidden_layers
    modules = []
    previous_layer_output = input_size
    # create hidden layers
    for i in range(hidden_layers):
        next_layer_input = int(previous_layer_output / 2)
        next_layer_input = next_layer_input if next_layer_input > 1 else 1
        modules.append(nn.Linear(previous_layer_output, next_layer_input))
        if batch_normalization:
            modules.append(nn.BatchNorm1d(next_layer_input))
        modules.append(ACTIVATION_MAPPING[activation]())
        previous_layer_output = next_layer_input
    # finally add the output layer
    if add_output_layer:
        # finally add the output layer
        modules.append(nn.Linear(previous_layer_output, output_size))

    return nn.Sequential(*modules), previous_layer_output


NET_MAPPING = {
    NetworkType.FullyConnected: fully_connected,
    NetworkType.FullyConnectedVariable: fully_connected_variable
}


def get_network(net_type: NetworkType, input_size: int, output_size: int, **net_params) -> Tuple[nn.Module, int]:
    if net_type in NET_MAPPING:
        return NET_MAPPING[net_type](input_size, output_size, **net_params)
    else:
        raise AttributeError('net_type not available')


def cnn_model_layers(
        input_size: int,
        cnn_units: List[CNNLayerConfig],
        init_weights: bool = True
) -> Tuple[List[nn.Module], int]:
    if len(cnn_units):
        layers = []
        in_layer_size = input_size
        out_layer_size = input_size
        for cnn_layer in cnn_units:
            layers.append(init_layer(nn.Conv2d, in_layer_size, cnn_layer.out_channels,
                                     kernel_size=cnn_layer.kernel_size, stride=cnn_layer.stride,
                                     padding=cnn_layer.padding, init_weights=init_weights))
            if cnn_layer.add_batch_normalization:
                layers.append(init_layer(nn.BatchNorm2d, cnn_layer[0], init_weights=init_weights))
            layers.append(ACTIVATION_MAPPING[cnn_layer.activation]())
            in_layer_size = cnn_layer[0]
            out_layer_size = cnn_layer[0]
        return layers, out_layer_size
    else:
        return [], input_size


def conv_output_dim(cnn: nn.Module, input_dim: Tuple[int, int, int]):
    x = torch.zeros(1, *input_dim)
    x = cnn(x)
    return int(np.prod(x.shape))


def fcl_model_layers(
        input_size: int,
        output_size: int,
        fcl_units: List[FullyConnectedLayerConfig],
        add_last_layer: bool = True,
        init_weights: bool = True,
) -> Tuple[List[nn.Module], int]:
    layers = []
    previous_layer = input_size
    last_units = input_size
    for layer in fcl_units:
        layers.append(init_layer(nn.Linear, previous_layer, layer.units, init_weights=init_weights))
        if layer.add_batch_normalization:
            layers.append(init_layer(nn.BatchNorm1d, layer.units, init_weights=init_weights))
        layers.append(ACTIVATION_MAPPING[layer.activation]())
        previous_layer = layer.units
        last_units = layer.units

    if add_last_layer:
        layers.append(nn.Linear(last_units, output_size))

    return layers, last_units
