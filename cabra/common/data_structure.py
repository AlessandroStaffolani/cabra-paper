from abc import abstractmethod
from dataclasses import dataclass
from enum import IntFlag
from typing import Dict, Any

from cabra.common.enum_utils import ExtendedEnum


class RunMode(str, ExtendedEnum):
    Train = 'training'
    Validation = 'validation'
    Eval = 'evaluation'


@dataclass
class BaseEntry:
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    # def __dict__(self):
    #     return self.to_dict()

    def __iter__(self):
        for value in self.to_dict().values():
            yield value


class Done(IntFlag):
    NotDone = 0
    # virtual done is used to end the episode, but it not reset the environment (nodes, trucks and generator)
    VirtualDone = 1
    # it will cause the end of the episode and the full reset of the environment
    Done = 2

    def to_bool(self) -> bool:
        return self.value > 0

    def __bool__(self):
        return self.to_bool()
