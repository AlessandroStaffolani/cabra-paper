import json
from typing import Tuple, Optional, List, Union

import numpy as np
from torch import nn as nn, Tensor

from cabra.environment.agent.distributions import make_proba_distribution, Distribution
from cabra.environment.data_structure import DistributionType
from cabra.environment.agent.pytorch_net import cnn_model_layers, conv_output_dim, fcl_model_layers
from cabra.environment.agent.torch_utils import init_layer
from cabra.environment.config import CNNLayerConfig, FullyConnectedLayerConfig


class ActorCriticModel(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            shared_cnn_units: List[CNNLayerConfig],
            shared_fcl_units: List[FullyConnectedLayerConfig],
            policy_layers: List[FullyConnectedLayerConfig],
            value_layers: List[FullyConnectedLayerConfig],
            distribution_type: DistributionType,
            distribution_dim: Union[int, List[int]],
            action_type_mapping,
            log_std_init: float = 0,
            init_weights: bool = True,
    ):
        super(ActorCriticModel, self).__init__()
        self.distribution_type: DistributionType = distribution_type
        self.log_std_init: float = log_std_init
        self.distribution_dim: Union[int, List[int]] = distribution_dim
        self.action_type_mapping = action_type_mapping
        cnn_layers, in_size = cnn_model_layers(input_size, shared_cnn_units, init_weights=init_weights)
        if len(cnn_layers) > 0:
            shared_layers = cnn_layers
            shared_layers.append(init_layer(nn.Flatten(), init_weights=init_weights))
            in_size = conv_output_dim(
                init_layer(nn.Sequential, *cnn_layers, init_weights=init_weights), (1, input_size, input_size))
        else:
            shared_layers = []

        fcl_layers, out_size = fcl_model_layers(
            in_size, output_size, shared_fcl_units, False, init_weights=init_weights)

        shared_layers += fcl_layers
        actor_layers, self.latent_dim_pi = fcl_model_layers(
            out_size, output_size, policy_layers, False, init_weights=init_weights)

        critic_layers, self.latent_dim_vf = fcl_model_layers(
            out_size, output_size, value_layers, False, init_weights=init_weights)

        # Create networks
        # Action distribution
        self.action_dist = make_proba_distribution(distribution_type, distribution_dim, dist_kwargs={
            'action_type_mapping': self.action_type_mapping,
        })
        # std layer used for gaussian distributions
        self.log_std: Optional[nn.Parameter] = None
        # If the list of layers is empty, the network will just act as an Identity module
        self.shared_net, self.actor_net, self.critic_net, self.log_std = self.init_net(
            shared_layers, actor_layers, critic_layers, init_weights=init_weights)

        self.current_distribution: Optional[Distribution] = None

    def init_net(
            self,
            shared_layers: List[nn.Module],
            actor_layers: List[nn.Module],
            critic_layers: List[nn.Module],
            init_weights: bool = True,
    ) -> Tuple[nn.Module, nn.Module, nn.Module, nn.Parameter]:
        log_std = None
        if self.distribution_type == DistributionType.Gaussian:
            action_layer, log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.latent_dim_pi, log_std_init=self.log_std_init
            )
        elif self.distribution_type in [DistributionType.Categorical, DistributionType.MultiCategorical,
                                        DistributionType.MultiBinary]:
            action_layer = self.action_dist.proba_distribution_net(
                latent_dim=self.latent_dim_pi, init_weights=init_weights)
        elif self.distribution_type == DistributionType.GaussianCategorical:
            action_layer, log_std = self.action_dist.proba_distribution_net(
                latent_dim=self.latent_dim_pi, log_std_init=self.log_std_init, init_weights=init_weights
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        actor_layers.append(action_layer)
        critic_layers.append(init_layer(nn.Linear, self.latent_dim_vf, 1, init_weights=init_weights))

        return nn.Sequential(*shared_layers), nn.Sequential(*actor_layers), nn.Sequential(*critic_layers), log_std

    def forward(
            self,
            features: Tensor,
            deterministic: bool = False,
            mask: Optional[List[Tuple[np.ndarray, ...]]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        shared_latent = self.shared_net(features)
        latent_pi = self.actor_net(shared_latent)

        values = self.critic_net(shared_latent)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if mask is not None:
            distribution.apply_masking(mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_latent_pi_and_value(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        self.current_distribution = None
        shared_latent = self.shared_net(features)
        latent_pi = self.actor_net(shared_latent)
        values = self.critic_net(shared_latent)
        self.current_distribution = self._get_action_dist_from_latent(latent_pi)
        return latent_pi, values

    def get_value(self, features: Tensor) -> Tensor:
        shared_latent = self.shared_net(features)
        values = self.critic_net(shared_latent)
        return values

    def get_log_prob(
            self,
            latent_pi: Tensor,
            actions: Tensor,
            mask: Optional[List[Tuple[np.ndarray, ...]]] = None
    ) -> Tensor:
        distribution = self._get_action_dist_from_latent(latent_pi)
        if mask is not None:
            distribution.apply_masking(mask)
        log_prob = distribution.log_prob(actions)
        return log_prob

    def get_action_from_distribution(
            self,
            latent_pi: Tensor,
            index: Optional[int] = None,
            mask: Optional[Tuple[np.ndarray, ...]] = None,
            deterministic: bool = True
    ) -> Tensor:
        distribution = self._get_action_dist_from_latent(latent_pi)
        if mask is not None:
            index_val = self.action_type_mapping[index]
            distribution.apply_masking_with_index(index, mask[index_val])
        if index is not None:
            return distribution.get_action_with_index(index, deterministic)
        else:
            return distribution.get_actions(deterministic)

    def evaluate_actions(
            self,
            states: Tensor,
            actions: Tensor,
            masks: Optional[List[Tuple[np.ndarray, ...]]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        """
        self.current_distribution = None
        shared_latent = self.shared_net(states)
        latent_pi = self.actor_net(shared_latent)

        values = self.critic_net(shared_latent)
        distribution = self._get_action_dist_from_latent(latent_pi)
        if masks is not None:
            distribution.apply_masking(masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def _get_action_dist_from_latent(self, latent_pi: Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        """
        if self.current_distribution is not None:
            return self.current_distribution
        if self.distribution_type == DistributionType.Gaussian:
            return self.action_dist.proba_distribution(latent_pi, self.log_std)
        elif self.distribution_type == DistributionType.Categorical:
            # Here latent_pi are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=latent_pi)
        elif self.distribution_type == DistributionType.MultiCategorical:
            # Here latent_pi are the flattened logits
            return self.action_dist.proba_distribution(action_logits=latent_pi)
        elif self.distribution_type == DistributionType.MultiBinary:
            # Here latent_pi are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=latent_pi)
        elif self.distribution_type == DistributionType.GaussianCategorical:
            return self.action_dist.proba_distribution(latent_space=latent_pi, log_std=self.log_std)
        else:
            raise ValueError(f"Invalid action distribution: {self.distribution_type}")
