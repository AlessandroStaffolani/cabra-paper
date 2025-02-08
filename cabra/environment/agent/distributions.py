"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch.distributions import Distribution as TorchDistribution
from torch.distributions.utils import logits_to_probs

from cabra.environment.agent.torch_utils import init_layer
from cabra.environment.data_structure import DistributionType, ActionType

SelfDistribution = TypeVar("SelfDistribution", bound="Distribution")
SelfDiagGaussianDistribution = TypeVar("SelfDiagGaussianDistribution", bound="DiagGaussianDistribution")
SelfSquashedDiagGaussianDistribution = TypeVar(
    "SelfSquashedDiagGaussianDistribution", bound="SquashedDiagGaussianDistribution"
)
SelfCategoricalDistribution = TypeVar("SelfCategoricalDistribution", bound="CategoricalDistribution")
SelfMultiCategoricalDistribution = TypeVar("SelfMultiCategoricalDistribution", bound="MultiCategoricalDistribution")
SelfBernoulliDistribution = TypeVar("SelfBernoulliDistribution", bound="BernoulliDistribution")
SelfGaussianCategoricalDistribution = TypeVar(
    "SelfGaussianCategoricalDistribution", bound="GaussianCategoricalDistribution")


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution: Optional[TorchDistribution] = None
        self.distributions: Optional[List[TorchDistribution]] = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Module, Tuple[nn.Module, nn.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self: SelfDistribution, *args, **kwargs) -> SelfDistribution:
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[th.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> th.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    def sample_with_index(self, index: Union[int, ActionType]) -> th.Tensor:
        """
        Returns a sample from the given index, used in the MultiCategorical Distribution
        """
        pass

    @abstractmethod
    def mode(self) -> th.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def mode_with_index(self, index: Union[int, ActionType]) -> th.Tensor:
        """
        Returns the most likely action (deterministic output) from the given index,
         used in the MultiCategorical Distribution
        """
        pass

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    def get_action_with_index(self, index: Union[int, ActionType], deterministic: bool = False) -> th.Tensor:
        """
        Return actions according to the probability distribution from the given index,
         used in the MultiCategorical Distribution
        """
        if deterministic:
            return self.mode_with_index(index)
        return self.sample_with_index(index)

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """

    @abstractmethod
    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        """
        Eliminate ("mask out") chosen distribution outcomes by setting their probability to 0.
        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If True, the corresponding choice's logit value is preserved. If False, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

    def apply_masking_with_index(self, index: int, masks: Optional[np.ndarray]):
        """
        Eliminate ("mask out") chosen distribution outcomes by setting their probability to 0 from the given index,
         used in the MultiCategorical Distribution
        """


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class MaskableCategorical(Categorical):
    """
    Modified PyTorch Categorical distribution with support for invalid action masking.
    To instantiate, must provide either probs or logits, but not both.
    :param probs: Tensor containing finite non-negative values, which will be renormalized
        to sum to 1 along the last dimension.
    :param logits: Tensor of unnormalized log probabilities.
    :param validate_args: Whether or not to validate that arguments to methods like lob_prob()
        and icdf() match the distribution's shape, support, etc.
    :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set to a
        large negative value, resulting in near 0 probability.
    """

    def __init__(
            self,
            probs: Optional[th.Tensor] = None,
            logits: Optional[th.Tensor] = None,
            validate_args: Optional[bool] = None,
            masks: Optional[np.ndarray] = None,
    ):
        self.masks: Optional[th.Tensor] = None
        super().__init__(probs, logits, validate_args)
        self._original_logits = self.logits
        self.apply_masking(masks)

    def apply_masking(self, masks: Optional[np.ndarray]) -> None:
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.
        :param masks: An optional boolean ndarray of compatible shape with the distribution.
            If False, the corresponding choice's logit value is preserved. If True, it is set
            to a large negative value, resulting in near 0 probability. If masks is None, any
            previously applied masking is removed, and the original logits are restored.
        """

        if masks is not None:
            device = self.logits.device
            self.masks = th.as_tensor(np.array(masks), dtype=th.bool, device=device).reshape(self.logits.shape)
            HUGE_NEG = th.tensor(-1e8, dtype=self.logits.dtype, device=device)

            logits = th.where(self.masks, HUGE_NEG, self._original_logits)
        else:
            self.masks = None
            logits = self._original_logits

        # Reinitialize with updated logits
        super().__init__(logits=logits)

        # self.probs may already be cached, so we must force an update
        self.probs = logits_to_probs(self.logits)

    def entropy(self) -> th.Tensor:
        if self.masks is None:
            return super().entropy()

        # Highly negative logits don't result in 0 probs, so we must replace
        # with 0s to ensure 0 contribution to the distribution's entropy, since
        # masked actions possess no uncertainty.
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, th.tensor(0.0, device=device), p_log_p)
        return -p_log_p.sum(-1)


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, **kwargs):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(
            self,
            latent_dim: int,
            log_std_init: float = 0.0,
            init_weights: bool = True
    ) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)
        """
        mean_actions = init_layer(nn.Linear, latent_dim, self.action_dim, init_weights=init_weights)
        # TODO: allow action dependent std
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(
            self: SelfDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfDiagGaussianDistribution:
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor,
                            deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        pass


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(
            self: SelfSquashedDiagGaussianDistribution, mean_actions: th.Tensor, log_std: th.Tensor
    ) -> SelfSquashedDiagGaussianDistribution:
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= th.sum(th.log(1 - actions ** 2 + self.epsilon), dim=1)
        return log_prob

    def entropy(self) -> Optional[th.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return th.tanh(self.gaussian_actions)

    def mode(self) -> th.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return th.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob

    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        raise NotImplementedError()


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int, **kwargs):
        super().__init__()
        self.distribution: Optional[MaskableCategorical] = None
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int, init_weights: bool = True) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.
        """
        action_logits = init_layer(nn.Linear, latent_dim, self.action_dim, init_weights=init_weights)
        return action_logits

    def proba_distribution(self: SelfCategoricalDistribution, action_logits: th.Tensor) -> SelfCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, self.action_dim)
        self.distribution = MaskableCategorical(logits=reshaped_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return th.argmax(self.distribution.probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        assert self.distribution is not None, "Must set distribution parameters"
        if masks is not None:
            self.distribution.apply_masking(masks)


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions. Supports invalid action masking.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int], action_type_mapping: Dict[ActionType, int], **kwargs):
        super().__init__()
        self.distributions: List[MaskableCategorical] = []
        self.action_dims = action_dims
        self.action_type_mapping: Dict[ActionType, int] = action_type_mapping

    def proba_distribution_net(self, latent_dim: int, init_weights: bool = True) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.
        :return:
        """

        action_logits = init_layer(nn.Linear, latent_dim, sum(self.action_dims), init_weights=init_weights)
        return action_logits

    def proba_distribution(
            self: SelfMultiCategoricalDistribution, action_logits: th.Tensor
    ) -> SelfMultiCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_logits = action_logits.view(-1, sum(self.action_dims))

        self.distributions = [
            MaskableCategorical(logits=split) for split in th.split(reshaped_logits, tuple(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        # Restructure shape to align with each categorical
        actions = actions.view(-1, len(self.action_dims))

        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([dist.sample() for dist in self.distributions], dim=1)

    def sample_with_index(self, index: Union[int, ActionType]) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        if isinstance(index, ActionType):
            index = self.action_type_mapping[index]
        return self.distributions[index].sample()

    def mode(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)

    def mode_with_index(self, index: Union[int, ActionType]) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        if isinstance(index, ActionType):
            index = self.action_type_mapping[index]
        return th.argmax(self.distributions[index].probs, dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        if masks is not None:
            for distribution, mask in zip(self.distributions, zip(*masks)):
                distribution.apply_masking(mask)

    def apply_masking_with_index(self, index: int, masks: Optional[np.ndarray]):
        if masks is not None:
            if isinstance(index, ActionType):
                index = self.action_type_mapping[index]
            self.distributions[index].apply_masking(masks)


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dims: int, **kwargs):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution(self: SelfBernoulliDistribution, action_logits: th.Tensor) -> SelfBernoulliDistribution:
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        return th.round(self.distribution.probs)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        pass


class GaussianCategoricalDistribution(Distribution):
    """
    GaussianCategoricalDistribution distribution for multi space action space with continuous and discrete sub-actions.
    Supports invalid action masking for the discrete part. The first action is the continuous sub-action
    (DiagGaussianDistribution), while the second one is discrete (MaskableCategorical)

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: Tuple[int, int, int], action_type_mapping: Dict[ActionType, int], **kwargs):
        super().__init__()
        self.distributions: Tuple[MaskableCategorical, Normal, MaskableCategorical] = tuple()
        self.action_dims: Tuple[int, int, int] = action_dims
        self.action_type_mapping: Dict[ActionType, int] = action_type_mapping

    def proba_distribution_net(
            self,
            latent_dim: int,
            log_std_init: float = 0.0,
            init_weights: bool = True,
    ) -> Tuple[nn.Module, nn.Parameter]:
        latent_space = init_layer(nn.Linear, latent_dim, sum(self.action_dims), init_weights=init_weights)
        if len(self.action_dims) == 2:
            log_std = nn.Parameter(th.ones(self.action_dims[0]) * log_std_init, requires_grad=True)
        else:
            log_std = nn.Parameter(th.ones(self.action_dims[1]) * log_std_init, requires_grad=True)
        return latent_space, log_std

    def proba_distribution(
            self: SelfGaussianCategoricalDistribution,
            latent_space: th.Tensor,
            log_std: th.Tensor
    ) -> SelfGaussianCategoricalDistribution:
        # Restructure shape to align with logits
        reshaped_latent = latent_space.view(-1, sum(self.action_dims))

        if len(self.action_dims) == 2:
            mean_actions, a_logits_2 = th.split(reshaped_latent, list(self.action_dims), dim=1)
            action_std = th.ones_like(mean_actions) * log_std.exp()
            self.distributions = (
                Normal(mean_actions, action_std),
                MaskableCategorical(logits=a_logits_2)
            )
            return self
        else:
            a_logits_1, mean_actions, a_logits_2 = th.split(reshaped_latent, list(self.action_dims), dim=1)
            action_std = th.ones_like(mean_actions) * log_std.exp()
            self.distributions = (
                MaskableCategorical(logits=a_logits_1),
                Normal(mean_actions, action_std),
                MaskableCategorical(logits=a_logits_2)
            )
            return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        if len(self.action_dims) == 2:
            gaussian_actions = actions[:, :self.action_dims[0]]
            cat_action_2 = actions[:, self.action_dims[0]:].squeeze(-1)
            log_probs = [
                sum_independent_dims(self.distributions[0].log_prob(gaussian_actions)),
                self.distributions[1].log_prob(cat_action_2)
            ]
        else:
            cat_action_1 = actions[:, :1].squeeze(-1)
            gaussian_actions = actions[:, 1:1 + self.action_dims[1]]
            cat_action_2 = actions[:, 1 + self.action_dims[1]:].squeeze(-1)
            log_probs = [
                self.distributions[0].log_prob(cat_action_1),
                sum_independent_dims(self.distributions[1].log_prob(gaussian_actions)),
                self.distributions[2].log_prob(cat_action_2)
            ]
        return th.stack(log_probs, dim=1).sum(dim=1)

    def entropy(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        if len(self.action_dims) == 2:
            entropies = [
                sum_independent_dims(self.distributions[0].entropy()),
                self.distributions[1].entropy()
            ]
        else:
            entropies = [
                self.distributions[0].entropy(),
                sum_independent_dims(self.distributions[1].entropy()),
                self.distributions[2].entropy()
            ]

        return th.stack(entropies, dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        if len(self.action_dims) == 2:
            samples = [
                self.distributions[0].rsample(),
                self.distributions[1].sample()
            ]
        else:
            samples = [
                self.distributions[0].sample(),
                self.distributions[1].rsample(),
                self.distributions[2].sample()
            ]

        return th.stack(samples, dim=1)

    def sample_with_index(self, index: Union[int, ActionType]) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        assert isinstance(index, ActionType)
        index_val = self.action_type_mapping[index]
        if index == ActionType.Target:
            return self.distributions[index_val].rsample()
        else:
            return self.distributions[index_val].sample()

    def mode(self) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        if len(self.action_dims) == 2:
            modes = [
                self.distributions[0].mean,
                th.argmax(self.distributions[1].probs, dim=1)
            ]
        else:
            modes = [
                th.argmax(self.distributions[0].probs, dim=1),
                self.distributions[1].mean,
                th.argmax(self.distributions[2].probs, dim=1)
            ]

        return th.stack(modes, dim=1)

    def mode_with_index(self, index: Union[int, ActionType]) -> th.Tensor:
        assert len(self.distributions) > 0, "Must set distribution parameters"
        assert isinstance(index, ActionType)
        index_val = self.action_type_mapping[index]
        if index == ActionType.Target:
            return self.distributions[index_val].mean
        else:
            return th.argmax(self.distributions[index_val].probs, dim=1)

    def actions_from_params(
            self,
            latent_space: th.Tensor,
            log_std: th.Tensor,
            deterministic: bool = False
    ) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(latent_space, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
            self,
            latent_space: th.Tensor,
            log_std: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(latent_space, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def apply_masking(self, masks: Optional[List[Tuple[np.ndarray, ...]]]) -> None:
        assert len(self.distributions) > 0, "Must set distribution parameters"

        if masks is not None:
            if len(self.action_dims) == 2:
                categorical_dist_2, categorical_mask_2 = list(zip(self.distributions, zip(*masks)))[1]
                categorical_dist_2.apply_masking(categorical_mask_2)
            else:
                categorical_dist_1, categorical_mask_1 = list(zip(self.distributions, zip(*masks)))[0]
                categorical_dist_1.apply_masking(categorical_mask_1)
                categorical_dist_2, categorical_mask_2 = list(zip(self.distributions, zip(*masks)))[2]
                categorical_dist_2.apply_masking(categorical_mask_2)

    def apply_masking_with_index(self, index: int, masks: Optional[np.ndarray]):
        index = self.action_type_mapping[index]
        if len(self.action_dims) == 2:
            if masks is not None and index != 0:
                self.distributions[index].apply_masking(masks)
        else:
            if masks is not None and index != 1:
                self.distributions[index].apply_masking(masks)


PROBA_DISTRIBUTION_MAPPING = {
    DistributionType.Categorical: CategoricalDistribution,
    DistributionType.MultiCategorical: MultiCategoricalDistribution,
    DistributionType.Gaussian: DiagGaussianDistribution,
    DistributionType.MultiBinary: BernoulliDistribution,
    DistributionType.GaussianCategorical: GaussianCategoricalDistribution
}


def make_proba_distribution(
        distribution_type: DistributionType,
        action_dim: Union[int, List[int]],
        dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if distribution_type in PROBA_DISTRIBUTION_MAPPING:
        return PROBA_DISTRIBUTION_MAPPING[distribution_type](action_dim, **dist_kwargs)
    else:
        raise NotImplementedError(
            f"Error: probability distribution, not implemented for type {distribution_type}."
        )


def kl_divergence(dist_true: Distribution, dist_pred: Distribution) -> th.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence

    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a PyTorch Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, MultiCategoricalDistribution):
        assert np.allclose(dist_pred.action_dims,
                           dist_true.action_dims), "Error: distributions must have the same input space"
        return th.stack(
            [th.distributions.kl_divergence(p, q) for p, q in zip(dist_true.distribution, dist_pred.distribution)],
            dim=1,
        ).sum(dim=1)

    # Use the PyTorch kl_divergence implementation
    else:
        return th.distributions.kl_divergence(dist_true.distribution, dist_pred.distribution)
