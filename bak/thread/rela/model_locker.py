import torch
from threading import Condition
from torch import jit, nn
from typing import Any, List, Text, Union


class Stack:
    def __init__(self) -> None:
        self._data: List = []
        self._cv = Condition()

    def push(self, value: Any) -> None:
        with self._cv:
            self._data.append(value)
            self._cv.notify(n=1)

    def pop(self) -> Any:
        with self._cv:
            self._cv.wait_for(predicate=lambda: len(self._data) > 0)
            value = self._data.pop()
        return value


class ModelLocker:
    def __init__(
        self,
        models: List[Union[jit.ScriptModule, nn.Module]],
        device: Text
    ) -> None:
        self._models = models
        self._device = device
        # model locks
        self._available_models = Stack()
        for i in range(len(self._models)):
            self._available_models.push(i)

    def update_model(self, model: nn.Module) -> None:
        # lock all models
        for _ in range(len(self._models)):
            self._available_models.pop()
        for m in self._models:
            m.load_state_dict(state_dict=model.state_dict())
        # release all models
        for i in range(len(self._models)):
            self._available_models.push(i)

    def lock(self) -> int:
        return self._available_models.pop()

    def unlock(self, id_: int) -> None:
        self._available_models.push(id_)

    def forward(self, query: torch.Tensor, model_id: int = -1) -> torch.Tensor:
        lock: bool = model_id == -1
        id_: int = self._available_models.pop() if lock else model_id
        inputs = query.to(self._device)
        results = self._models[id_](inputs)
        if lock:
            self._available_models.push(id_)
        # detach is needed to free the memory allocated to gradients
        # either this or torch.no_grad
        return results.detach().cpu()
