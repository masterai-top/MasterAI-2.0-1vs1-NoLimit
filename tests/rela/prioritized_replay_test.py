import os
import sys
import time
import torch
import torch.multiprocessing as mp
sys.path.append(os.getcwd())

#
from rela.prioritized_replay import PrioritizedReplay, ValueTransition


class Producer(mp.Process):
    def __init__(self, queue):
        super(Producer, self).__init__()
        self._queue = queue

    def run(self):
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
                print("The size of queue is %s (comsumer)" % self._queue.size())
                time.sleep(1)


def test_replay():
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
    process_producer = Producer(replay_buffer)
    process_consumer = Consumer(replay_buffer)
    process_producer.start()
    process_consumer.start()
    process_producer.join()
    process_consumer.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    test_replay()