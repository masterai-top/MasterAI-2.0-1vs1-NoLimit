import logging
import torch.multiprocessing as mp
import torch
from typing import List

from game.net import INet
from game.recursive_solving import RecursiveSolvingParams, RlRunner
from .types import ValueTransition
from .model_locker import ModelLocker
from .process_loop import ProcessLoop
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
        proc = mp.current_process()
        logging.debug(
            msg="[inet, data_gen] pid: %d, name: %s, replay buffer size: %d"
            % (proc.pid, proc.name, self._replay_buffer.size())
        )


class DataProcessLoop(ProcessLoop):
    def __init__(
        self,
        connector: CVNetBufferConnector,
        cfg: RecursiveSolvingParams,
        seed: int
    ) -> None:
        super(DataProcessLoop, self).__init__()
        self._connector = connector
        self._cfg = cfg
        self._seed = seed

    def main_loop(self):
        proc = mp.current_process()
        logging.debug(
            msg="[proc_loop, data_gen] process id: %d, name: %s"
            % (proc.pid, proc.name)
        )
        runner = RlRunner(
            params=self._cfg, net=self._connector, seed=self._seed
        )
        while not self.terminated:
            if self.paused:
                self.wait_until_resume()
            logging.debug(
                msg="[proc_loop, data_gen] process id: %d, name: %s, go to step"
                % (proc.pid, proc.name)
            )
            runner.step()
