from abc import abstractmethod
from torch.multiprocessing import Condition, Lock, Value


class ProcessLoop:
    def __init__(self) -> None:
        self._terminated: Value = Value("b", False, lock=True)
        self._m_paused: Lock = Lock()
        self._paused: bool = False
        self._cv_paused = Condition(lock=self._m_paused)

    def terminate(self) -> None:
        with self._terminated:
            self._terminated.value = True

    def pause(self) -> None:
        with self._m_paused:
            self._paused = True

    def resume(self) -> None:
        with self._m_paused:
            self._paused = False
            self._cv_paused.notify(n=1)

    def wait_until_resume(self) -> None:
        with self._m_paused:
            self._cv_paused.wait_for(predicate=lambda: not self._paused)

    @property
    def terminated(self) -> bool:
        return self._terminated.value

    @property
    def paused(self) -> bool:
        return self._paused

    @abstractmethod
    def main_loop(self):
        pass
