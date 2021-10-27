from typing import List
from torch.multiprocessing import Process, Value
from typing import List

from .process_loop import ProcessLoop


class Context:
    def __init__(self) -> None:
        self._started: bool = False
        self._num_terminated_procs: Value = Value("i", 0, lock=True)
        self._loops: List[ProcessLoop] = []
        self._procs: List[Process] = []

    def __del__(self) -> None:
        """destructor"""
        for l in self._loops:
            l.terminate()
        for p in self._procs:
            p.join()

    def push_env_process(self, env: ProcessLoop) -> int:
        assert not self._started, "error: context must be started first"
        self._loops.append(env)
        return len(self._loops)

    def start(self) -> None:
        def _proc(env: ProcessLoop) -> None:
            env.main_loop()
            with self._num_terminated_procs:
                self._num_terminated_procs.value += 1

        for env in self._loops:
            self._procs.append(Process(target=_proc, args=(env, )))
        for proc in self._procs:
            proc.start()

    def pause(self) -> None:
        for l in self._loops:
            l.pause()

    def resume(self) -> None:
        for l in self._loops:
            l.resume()

    def terminate(self) -> None:
        for l in self._loops:
            l.terminate()

    def terminated(self) -> bool:
        print(">>> %d" % self._num_terminated_procs)
        return self._num_terminated_procs.value == len(self._loops)
