import torch.multiprocessing as mp
import random
import time


class Producer(mp.Process):
    def __init__(self, queue):
        super(Producer, self).__init__()
        self._queue = queue

    def run(self):
        for i in range(10):
            item = random.randint(0, 256)
            self._queue.put(item)
            print("Process Producer : item %d appended to queue %s" %
                  (item, self.name))
            time.sleep(1)
            print("The size of queue is %s" % self._queue.qsize())


class Consumer(mp.Process):
    def __init__(self, queue):
        super(Consumer, self).__init__()
        self._queue = queue

    def run(self):
        while True:
            if self._queue.empty():
                print("the queue is empty")
                break
            else:
                time.sleep(2)
                item = self._queue.get()
                print('Process Consumer : item %d popped from by %s \n' %
                      (item, self.name))
                time.sleep(1)


class Replay:
    def __init__(self) -> None:
        self._queue = mp.Queue()

    def put(self, item):
        self._queue.put(item)

    def get(self):
        return self._queue.get()

    def qsize(self):
        return self._queue.qsize()

    def empty(self):
        return self._queue.empty()


if __name__ == '__main__':
    mp.set_start_method("spawn")
    queue = Replay()
    process_producer = Producer(queue)
    process_consumer = Consumer(queue)
    process_producer.start()
    process_consumer.start()
    process_producer.join()
    process_consumer.join()
