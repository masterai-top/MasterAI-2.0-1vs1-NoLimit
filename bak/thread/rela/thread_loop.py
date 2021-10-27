from abc import abstractmethod
from threading import Condition, Lock


class ThreadLoop:
    def __init__(self) -> None:
        self._terminated: bool = False
        self._m_paused: Lock = Lock()
        self._paused: bool = False
        self._cv_paused = Condition(lock=self._m_paused)

    def terminate(self) -> None:
        self._terminated = True

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
        return self._terminated

    @property
    def paused(self) -> bool:
        return self._paused

    @abstractmethod
    def main_loop(self):
        pass
