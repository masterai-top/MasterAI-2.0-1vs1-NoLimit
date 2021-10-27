from typing import List
from threading import Thread
from typing import List

from .thread_loop import ThreadLoop


class Context:
    def __init__(self) -> None:
        self._started: bool = False
        self._num_terminated_thread: int = 0
        self._loops: List[ThreadLoop] = []
        self._threads: List[Thread] = []

    def __del__(self) -> None:
        """destructor"""
        for l in self._loops:
            l.terminate()
        for t in self._threads:
            t.join()

    def push_env_thread(self, env: ThreadLoop) -> int:
        assert not self._started, "error: context must be started first"
        self._loops.append(env)
        return len(self._loops)

    def start(self) -> None:
        def _thread(env: ThreadLoop) -> None:
            env.main_loop()

        for env in self._loops:
            self._num_terminated_thread += 1
            self._threads.append(Thread(target=_thread, args=(env, )))
        for proc in self._threads:
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
        # print(">>> %d" % self._num_terminated_thread)
        return self._num_terminated_thread == len(self._loops)
