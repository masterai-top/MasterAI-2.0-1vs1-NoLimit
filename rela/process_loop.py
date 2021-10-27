import time
from abc import abstractmethod
from torch.multiprocessing import Process


class ProcessLoop(Process):
    def __init__(self) -> None:
        super(ProcessLoop, self).__init__()
        self._terminated: bool = False
        self._paused: bool = False

    def terminate(self) -> None:
        self._terminated = True

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def wait_until_resume(self) -> None:
        while not self._paused:
            time.sleep(10)

    def run(self) -> None:
        self.main_loop()

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def paused(self) -> bool:
        return self._paused

    @abstractmethod
    def main_loop(self):
        pass
