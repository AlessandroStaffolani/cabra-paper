from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, List, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from cabra.environment.data_structure import DistributionType, ActionType

WAIT_ACTION = 'WAIT'


class Action:

    def __init__(self, sub_actions: Dict[ActionType, str]):
        self._sub_actions: Dict[ActionType, str] = sub_actions

    def to_dict(self):
        d = {}
        for key, value in self._sub_actions.items():
            d[key.value] = getattr(self, value, None)
        return d

    def to_tuple(self) -> Tuple[int, ...]:
        return tuple(self.to_dict().values())

    def __dict__(self):
        return self.to_dict()

    def __eq__(self, other: 'Action'):
        equal = True
        for i, value in enumerate(self):
            if value != other[i]:
                equal = False
        return equal

    def __iter__(self):
        for value in self.__dict__().values():
            yield value

    def __getitem__(self, item):
        if isinstance(item, int):
            return tuple(self)[item]
        elif isinstance(item, ActionType):
            return self.action_from_type(item)
        else:
            return self.to_dict()[item]

    def action_from_type(self, action_type: ActionType) -> int:
        if action_type in self._sub_actions:
            return self[action_type.value]
        else:
            raise AttributeError(f'ActionType "{action_type}" not available for Action "{self.__class__.__name__}"')

    def __str__(self):
        name = self.__class__.__name__
        sub_actions = ''
        for key, value in self.to_dict().items():
            sub_actions += f'{key}={value} '
        return f'<{name} {sub_actions}>'

    def to_numpy(self) -> np.ndarray:
        return np.array([val for val in self.to_tuple() if val is not None])

    def to_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy())


class RepositionAction(Action):

    def __init__(
            self,
            wait: Optional[int] = None,
            target: Optional[int] = None,
            quantity: Optional[int] = None,
            action_vect: Optional[Union[np.ndarray, Tensor]] = None
    ):
        super(RepositionAction, self).__init__(
            {ActionType.Wait: 'wait', ActionType.Target: 'target', ActionType.Quantity: 'quantity'})
        if action_vect is None and (target is None or quantity is None):
            raise AttributeError('Action components and action_vect are both None')
        if action_vect is not None:
            if len(action_vect.shape) == 2:
                action_vect = action_vect.squeeze(0)
            wait = int(action_vect[0])
            target = int(action_vect[1])
            quantity = int(action_vect[2])
        self.wait: int = int(wait) if wait is not None else None
        self.target: int = int(target)
        self.quantity: int = int(quantity)

    def __getstate__(self):
        return self.to_tuple()

    def __setstate__(self, state):
        self.wait = state[0]
        self.target = state[1]
        self.quantity = state[2]
        self._sub_actions = {ActionType.Wait: 'wait', ActionType.Target: 'target', ActionType.Quantity: 'quantity'}


class ZoneAction(Action):

    def __init__(
            self,
            zone: Optional[int] = None
    ):
        super().__init__({ActionType.Zone: 'zone'})
        self.zone: int = int(zone)

    def __getstate__(self):
        return self.to_tuple()

    def __setstate__(self, state):
        self.zone = state[0]
        self._sub_actions = {ActionType.Zone: 'zone'}


class RawRepositionAction(Action):

    def __init__(
            self,
            wait,
            target,
            quantity,
    ):
        super(RawRepositionAction, self).__init__(
            {ActionType.Wait: 'wait', ActionType.Target: 'target', ActionType.Quantity: 'quantity'})
        self.wait = wait
        self.target = target
        self.quantity = quantity

    def __getstate__(self):
        return self.to_tuple()

    def __setstate__(self, state):
        self.wait = state[0]
        self.target = state[1]
        self.quantity = state[2]
        self._sub_actions = {ActionType.Wait: 'wait', ActionType.Target: 'target', ActionType.Quantity: 'quantity'}

    def to_numpy(self) -> np.ndarray:
        if not isinstance(self.target, int) and len(self.target) > 1:
            if self.wait is not None:
                return np.concatenate([np.array([self.wait]), self.target, np.array([self.quantity])])
            else:
                return np.concatenate([self.target, np.array([self.quantity])])
        else:
            return super().to_numpy()

    def to_tensor(self) -> torch.Tensor:
        if self.wait is not None:
            if not isinstance(self.target, int) and len(self.target) > 1:
                return torch.concatenate(
                    [torch.tensor([self.wait]), torch.tensor(self.target), torch.tensor([self.quantity])])
            elif not isinstance(self.target, int):
                return torch.tensor([[self.wait.item(), self.target.item(), self.quantity.item()]])
            else:
                return super().to_tensor()
        else:
            if not isinstance(self.target, int) and len(self.target) > 1:
                return torch.concatenate(
                    [torch.tensor(self.target), torch.tensor([self.quantity])])
            elif not isinstance(self.target, int):
                return torch.tensor([[self.target.item(), self.quantity.item()]])
            else:
                return super().to_tensor()


class SubActionSpace:

    def __init__(
            self,
            size: int,
            actions_mapping: Dict[int, Any] = None,
            add_wait_action: bool = False
    ):
        self.add_wait_action: bool = add_wait_action
        space_size = size
        self.wait_action_index: Optional[int] = None
        if self.add_wait_action:
            space_size += 1
            self.wait_action_index: int = size
        self._actions: np.ma.masked_array = np.ma.array(data=np.arange(0, space_size), mask=False)
        if actions_mapping is None:
            mapping = {a: a for a in self._actions.data}
        else:
            mapping = actions_mapping
        if self.wait_action_index is not None:
            mapping[self.wait_action_index] = WAIT_ACTION
        self.actions_mapping: Dict[int, Any] = mapping
        self.inverted_actions_mapping: Dict[Any, int] = {v: k for k, v in self.actions_mapping.items()}

    def __getitem__(self, item):
        return self._actions[item]

    def __contains__(self, item):
        return item in self._actions

    def get_available_actions(self) -> np.ndarray:
        return self._actions.compressed()

    def get_all_actions(self) -> np.ndarray:
        return self._actions.data

    def get_disabled_actions(self) -> np.ndarray:
        return self._actions.data[self._actions.mask]

    def get_mask(self) -> np.ndarray:
        return self._actions.mask

    def unmask_all(self):
        self._actions.mask = False

    def is_action_available(self, index: int):
        return not self._actions.mask[index]

    def disable_action(self, index: Union[int, slice]):
        self._actions.mask[index] = True

    def disable_wait_action(self):
        if self.add_wait_action is not None:
            self.disable_action(self.wait_action_index)

    def disable_all_except_wait(self):
        for i, _ in enumerate(self._actions.mask):
            if i != self.wait_action_index:
                self.disable_action(i)

    def disable_all(self):
        for i, _ in enumerate(self._actions.mask):
            self.disable_action(i)

    def enable_action(self, index: Union[int, slice]):
        self._actions.mask[index] = False

    def enable_wait_action(self):
        if self.add_wait_action:
            self.enable_action(self.wait_action_index)

    def size(self, no_wait=False):
        if no_wait and self.wait_action_index is not None:
            return self._actions.size - 1
        return self._actions.size

    def shape(self):
        return self._actions.shape

    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True
    ) -> Union[int, np.ndarray]:
        actions = self.get_available_actions() if compressed else self.get_all_actions()
        if random_state is None:
            return np.random.choice(actions, size=size, replace=replace, p=p)
        else:
            return random_state.choice(actions, size=size, replace=replace, p=p)

    def get_action_name(self, index: int):
        return self.actions_mapping[index]

    def is_wait_action(self, index: int) -> bool:
        return self.wait_action_index == index

    def get_action_index(self, name: str):
        return self.inverted_actions_mapping[name]

    def __str__(self):
        return f'<SubActionSpace actions={self._actions} >'


class AbstractActionSpace:

    @abstractmethod
    def raw_sub_action_size(self, action_type: ActionType) -> int:
        pass

    @property
    def raw_action_size(self) -> int:
        return 0

    @property
    def distribution_dim(self) -> Union[int, List[int]]:
        return [0]

    @property
    def distribution_dim_flatten(self) -> int:
        return sum(self.distribution_dim)

    @abstractmethod
    def unmask_all(self):
        pass

    @abstractmethod
    def disable_all_except_wait(self):
        pass

    @abstractmethod
    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True,
    ) -> Union[Action, List[Action]]:
        pass

    @abstractmethod
    def get_sizes(self) -> Dict[ActionType, int]:
        pass

    @abstractmethod
    def is_wait_action(self, action: Action) -> bool:
        pass

    @abstractmethod
    def action_to_action_value(self, action: Action) -> tuple:
        pass

    @abstractmethod
    def get_mask(self) -> Tuple[np.ndarray, ...]:
        pass

    @property
    def action_type_mapping(self) -> Dict[ActionType, int]:
        pass


class RepositionActionSpace(AbstractActionSpace):

    def __init__(
            self,
            n_nodes: int,
            n_quantities: int,
            min_quantity: int = 1,
            add_wait_action: bool = True,
            **kwargs
    ):
        self.target_space: SubActionSpace = SubActionSpace(size=n_nodes, add_wait_action=add_wait_action)
        quantity_mapping: Dict[int, int] = {}
        index = 0
        for v in range(-n_quantities, n_quantities + 1):
            if v != 0 and abs(v) > min_quantity:
                quantity_mapping[index] = v
                index += 1

        self.quantity_space: SubActionSpace = SubActionSpace(
            size=len(quantity_mapping), actions_mapping=quantity_mapping, add_wait_action=add_wait_action)
        self.action_space_distribution: DistributionType = DistributionType.MultiCategorical

    def raw_sub_action_size(self, action_type: ActionType) -> int:
        return 1

    @property
    def raw_action_size(self) -> int:
        return 2

    @property
    def distribution_dim(self) -> Union[int, List[int]]:
        return [
            self.target_space.size(),
            self.quantity_space.size(),
        ]

    @property
    def distribution_dim_flatten(self) -> int:
        return sum(self.distribution_dim)

    def unmask_all(self):
        self.target_space.unmask_all()
        self.quantity_space.unmask_all()

    def disable_all_except_wait(self):
        self.target_space.disable_all_except_wait()
        self.quantity_space.disable_all_except_wait()

    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True,
    ) -> Union[RepositionAction, List[RepositionAction]]:
        a_target = self.target_space.sample(size, replace, p, random_state, compressed)
        a_quantity = self.quantity_space.sample(size, replace, p, random_state, compressed)
        if size is None:
            return RepositionAction(target=a_target, quantity=a_quantity)
        else:
            action_tuple = list(zip(a_target, a_quantity))
            return [RepositionAction(*values) for values in action_tuple]

    def get_sizes(self) -> Dict[ActionType, int]:
        return {
            ActionType.Target: self.target_space.size(),
            ActionType.Quantity: self.quantity_space.size()
        }

    def is_wait_action(self, action: RepositionAction) -> bool:
        return self.target_space.is_wait_action(action.target) or self.quantity_space.is_wait_action(action.quantity)

    def action_to_action_value(self, action: 'RepositionAction') -> Tuple[bool, int, int]:
        """
        Take an action as input and it returns the action values. So it returns:
            - the wait flag
            - the target node index
            - the quantity to be allocated or removed
        """
        wait = self.is_wait_action(action)
        target = self.target_space.actions_mapping[action.target]
        quantity = self.quantity_space.actions_mapping[action.quantity]
        return wait, target, quantity

    def get_mask(self) -> Tuple[np.ndarray, ...]:
        return (
            self.target_space.get_mask().copy(),
            self.quantity_space.get_mask().copy()
        )

    @property
    def action_type_mapping(self) -> Dict[ActionType, int]:
        raise NotImplementedError


class ZoneActionSpace(AbstractActionSpace):

    def __init__(
            self,
            n_zones: int,
            zones_mapping: Dict[int, str],
            add_wait_action: bool = True,
            **kwargs
    ):
        self.n_zones: int = n_zones
        self.zone_space: SubActionSpace = SubActionSpace(size=n_zones, actions_mapping=zones_mapping,
                                                         add_wait_action=add_wait_action)
        self.action_space_distribution: DistributionType = DistributionType.Categorical

    def raw_sub_action_size(self, action_type: ActionType) -> int:
        return 1

    @property
    def raw_action_size(self) -> int:
        return 1

    @property
    def distribution_dim(self) -> Union[int, List[int]]:
        return self.zone_space.size()

    @property
    def distribution_dim_flatten(self) -> int:
        return self.distribution_dim

    def unmask_all(self):
        self.zone_space.unmask_all()

    def disable_all_except_wait(self):
        self.zone_space.disable_all_except_wait()

    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True,
    ) -> Union[ZoneAction, List[ZoneAction]]:
        a_zone = self.zone_space.sample(size, replace, p, random_state, compressed)
        if size is None:
            return ZoneAction(zone=a_zone)
        else:
            action_tuple = list(zip(a_zone, ))
            return [ZoneAction(*values) for values in action_tuple]

    def get_sizes(self) -> Dict[ActionType, int]:
        return {
            ActionType.Zone: self.zone_space.size(),
        }

    def is_wait_action(self, action: ZoneAction) -> bool:
        return self.zone_space.is_wait_action(action.zone)

    def action_to_action_value(self, action: ZoneAction) -> Tuple[bool, int]:
        """
        Take an action as input and it returns the action values. So it returns:
            - the wait flag
            - the zone index
        """
        wait = self.is_wait_action(action)
        zone = self.zone_space.actions_mapping[action.zone]
        return wait, zone

    def get_mask(self) -> Tuple[np.ndarray, ...]:
        return (
            self.zone_space.get_mask().copy()
        )

    @property
    def action_type_mapping(self) -> Dict[ActionType, int]:
        return {ActionType.Zone: 0}


class PPORepositionActionSpace(RepositionActionSpace):

    def __init__(
            self,
            n_nodes: int,
            n_quantities: int,
            min_quantity: int = 1,
            add_wait_space: bool = False,
            **kwargs
    ):
        super().__init__(n_nodes, n_quantities, min_quantity, add_wait_action=True)
        self.add_wait_space: bool = add_wait_space
        if add_wait_space:
            self.wait_space: SubActionSpace = SubActionSpace(size=2, add_wait_action=False)

    @property
    def action_type_mapping(self) -> Dict[ActionType, int]:
        if self.add_wait_space:
            return {ActionType.Wait: 0, ActionType.Target: 1, ActionType.Quantity: 2}
        else:
            return {ActionType.Target: 0, ActionType.Quantity: 1}

    @property
    def raw_action_size(self) -> int:
        return 3 if self.add_wait_space else 2

    @property
    def distribution_dim(self) -> Union[int, List[int]]:
        if self.add_wait_space:
            return [
                self.wait_space.size(),
                self.target_space.size(),
                self.quantity_space.size()
            ]
        else:
            return [
                self.target_space.size(),
                self.quantity_space.size()
            ]

    def unmask_all(self):
        super().unmask_all()
        if self.add_wait_space:
            self.wait_space.unmask_all()

    def disable_all_except_wait(self):
        super().disable_all_except_wait()
        if self.add_wait_space:
            self.wait_space.disable_all_except_wait()

    def sample(
            self,
            size=None,
            replace=True,
            p=None,
            random_state: np.random.RandomState = None,
            compressed=True,
    ) -> Union[RepositionAction, List[RepositionAction]]:
        a_wait = None
        if self.add_wait_space:
            a_wait = self.wait_space.sample(size, replace, p, random_state, compressed)
        a_target = self.target_space.sample(size, replace, p, random_state, compressed)
        a_quantity = self.quantity_space.sample(size, replace, p, random_state, compressed)
        if size is None:
            return RepositionAction(wait=a_wait, target=a_target, quantity=a_quantity)
        else:
            action_tuple = list(zip(a_wait, a_target, a_quantity))
            return [RepositionAction(*values) for values in action_tuple]

    def get_sizes(self) -> Dict[ActionType, int]:
        if self.add_wait_space:
            return {
                ActionType.Wait: self.wait_space.size(),
                ActionType.Target: self.target_space.size(),
                ActionType.Quantity: self.quantity_space.size()
            }
        else:
            return {
                ActionType.Target: self.target_space.size(),
                ActionType.Quantity: self.quantity_space.size()
            }

    def is_wait_action(self, action: RepositionAction) -> bool:
        if self.add_wait_space:
            return bool(action.wait) or super().is_wait_action(action)
        else:
            return super().is_wait_action(action)

    def get_mask(self) -> Tuple[np.ndarray, ...]:
        if self.add_wait_space:
            return (
                self.wait_space.get_mask().copy(),
                self.target_space.get_mask().copy(),
                self.quantity_space.get_mask().copy()
            )
        else:
            return (
                self.target_space.get_mask().copy(),
                self.quantity_space.get_mask().copy()
            )


@dataclass
class SpaceRange:
    low: float
    high: float


class ContinuousRepositionActionSpace(PPORepositionActionSpace):

    def __init__(
            self,
            n_nodes: int,
            n_quantities: int,
            min_quantity: int = 1,
            add_wait_space: bool = False,
            **kwargs
    ):
        super().__init__(n_nodes, n_quantities, min_quantity, add_wait_space)
        self.action_space_distribution: DistributionType = DistributionType.GaussianCategorical
        self.target_node_range: SpaceRange = SpaceRange(low=0, high=self.target_space.size())
        self.target_outputs: int = 2

    def raw_sub_action_size(self, action_type: ActionType) -> int:
        if action_type == ActionType.Target:
            return self.target_outputs
        else:
            return 1

    @property
    def raw_action_size(self) -> int:
        if self.add_wait_space:
            return self.target_outputs + 1 + 1
        else:
            return self.target_outputs + 1

    @property
    def distribution_dim(self) -> Union[int, List[int]]:
        if self.add_wait_space:
            return [self.wait_space.size(), self.target_outputs, self.quantity_space.size()]
        else:
            return [self.target_outputs, self.quantity_space.size()]

    def get_sizes(self) -> Dict[ActionType, int]:
        if self.add_wait_space:
            return {
                ActionType.Wait: self.wait_space.size(),
                ActionType.Target: self.target_outputs,
                ActionType.Quantity: self.quantity_space.size()
            }
        else:
            return {
                ActionType.Target: self.target_outputs,
                ActionType.Quantity: self.quantity_space.size()
            }
