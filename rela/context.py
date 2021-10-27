import torch.multiprocessing as mp
from typing import List
from torch.multiprocessing import Process, Value
from typing import List

from .process_loop import ProcessLoop


class Context:
    def __init__(self) -> None:
        self._started: bool = False
        self._num_terminated_procs: int = 0
        self._loops: List[ProcessLoop] = []

    def __del__(self) -> None:
        """destructor"""
        for l in self._loops:
            l.terminate()
        for l in self._loops:
            l.join()

    def push_env_process(self, env: ProcessLoop) -> int:
        assert not self._started, "error: the context has been started"
        env.daemon = True
        self._loops.append(env)
        return len(self._loops)

    def start(self) -> None:
        self._started = True
        for env in self._loops:
            env.start()
            self._num_terminated_procs += 1

    def pause(self) -> None:
        for l in self._loops:
            l.pause()

    def resume(self) -> None:
        for l in self._loops:
            l.resume()

    def terminate(self) -> None:
        self._started = False
        for l in self._loops:
            l.terminate()

    def terminated(self) -> bool:
        print(">>> %d" % self._num_terminated_procs)
        return self._num_terminated_procs == len(self._loops)
