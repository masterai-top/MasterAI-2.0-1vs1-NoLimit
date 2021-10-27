import os
import sys
import time
import torch
import torch.multiprocessing as mp
sys.path.append(os.getcwd())

#
from rela.process_loop import ProcessLoop
from rela.prioritized_replay import PrioritizedReplay, ValueTransition


class Context:
    def __init__(self) -> None:
        self._started: bool = False
        self._num_terminated_procs: int = 0
        self._loops = []
        self._procs = []

    def push_env_process(self, env: ProcessLoop) -> int:
        assert not self._started, "error: the context has been started"
        self._loops.append(env)
        return len(self._loops)

    def start(self) -> None:
        # ctx = mp.get_context("spawn")
        ctx = mp.get_context("forkserver")
        self._started = True
        for env in self._loops:
            env.start()
            self._num_terminated_procs += 1

class Producer(ProcessLoop):
    def __init__(self, queue):
        super(Producer, self).__init__()
        self._queue = queue

    def main_loop(self):
        for _ in range(10):
            query = torch.rand(size=(128, 1326 * 2 + 8))
            values = torch.rand(size=(128, 1326))
            transition: ValueTransition = ValueTransition(
                query=query, values=values
            )
            priority: torch.Tensor = torch.ones(size=(query.size(0), ))
            self._queue.add(transition, priority)
            time.sleep(1)
            print("The size of queue is %s (producer)" % self._queue.size())


class Consumer(mp.Process):
    def __init__(self, queue):
        super(Consumer, self).__init__()
        self._queue = queue

    def run(self):
        while True:
            if self._queue.size() <= 0:
                print("the queue is empty")
                time.sleep(2)
            else:
                item = self._queue.sample(32, "cpu")
                print("The size of queue is %s (comsumer)" %
                      self._queue.size())
                time.sleep(1)


def test_contex() -> None:
    n_subprocs = 1
    replay_params = dict(
        capacity=2 ** 20,
        seed=10001,
        alpha=1.0,
        beta=0.4,
        prefetch=0,
        use_priority=False,
        compressed_values=False
    )
    replay_buffer = PrioritizedReplay(**replay_params)
    ctx = Context()
    ctx.push_env_process(Producer(replay_buffer))
    process_consumer = Consumer(replay_buffer)
    ctx.start()
    process_consumer.start()
    process_consumer.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    test_contex()
