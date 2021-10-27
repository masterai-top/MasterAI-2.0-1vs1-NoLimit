import logging
import threading
import torch
from typing import List

from game.net import INet
from game.recursive_solving import RecursiveSolvingParams, RlRunner
from .types import ValueTransition
from .model_locker import ModelLocker
from .thread_loop import ThreadLoop
from .prioritized_replay import ValuePrioritizedReplay


class CVNetBufferConnector(INet):
    MAX_SIZE = 1 << 12

    def __init__(
        self,
        model_locker: ModelLocker,
        replay_buffer: ValuePrioritizedReplay
    ) -> None:
        self._model_locker = model_locker
        self._replay_buffer = replay_buffer

    def compute_values(self, query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            size = query.size(0)
            if size > self.MAX_SIZE:
                sizes: List[int] = []
                for start in range(0, size, self.MAX_SIZE):
                    sizes.append(min(self.MAX_SIZE, size - start))
                chunks = torch.split(
                    tensor=query, split_size_or_sections=sizes, dim=0
                )
                results: List[torch.Tensor] = []
                for input_ in chunks:
                    results.append(self._model_locker.forward(query=input_))
                return torch.cat(tensors=results, dim=0)
            else:
                return self._model_locker.forward(query=query)

    def add_training_example(
        self, query: torch.Tensor, values: torch.Tensor
    ) -> None:
        transition: ValueTransition = ValueTransition(
            query=query, values=values
        )
        priority: torch.Tensor = torch.ones(size=(query.size(0), ))
        self._replay_buffer.add(sample=transition, priority=priority)
        thread = threading.current_thread()
        logging.debug(
            msg="[inet, data_gen] tid: %d, name: %s, replay buffer size: %d"
            % (thread.ident, thread.name, self._replay_buffer.size())
        )


class DataThreadLoop(ThreadLoop):
    def __init__(
        self,
        connector: CVNetBufferConnector,
        cfg: RecursiveSolvingParams,
        seed: int
    ) -> None:
        super(DataThreadLoop, self).__init__()
        self._connector = connector
        self._cfg = cfg
        self._seed = seed

    def main_loop(self):
        thread = threading.current_thread()
        logging.debug(
            msg="[thread_loop, data_gen] tid: %d, name: %s"
            % (thread.ident, thread.name)
        )
        runner = RlRunner(
            params=self._cfg, net=self._connector, seed=self._seed
        )
        while not self.terminated:
            if self.paused:
                self.wait_until_resume()
            logging.debug(
                msg="[thread_loop, data_gen] tid: %d, name: %s, go to step"
                % (thread.ident, thread.name)
            )
            runner.step()
