import random
import time
import torch
from ctypes import c_bool
from concurrent.futures import Future, ThreadPoolExecutor
from torch.multiprocessing import Array, Condition, Lock, Manager, Queue, Value
from typing import List, Text, Tuple, Union

from .types import ValueTransition, dequantize, from_vector, make_batch

ExtractedData = List[torch.Tensor]
SampleWeightIds = Tuple[ValueTransition, torch.Tensor, List[int]]


class ConcurrentQueue:
    def __init__(
        self,
        capacity: int
    ) -> None:
        self._mutex = Lock()
        self._cv_size = Condition(lock=self._mutex)
        self._cv_tail = Condition(lock=self._mutex)
        # shared memory
        self._capacity: int = Value("i", capacity, lock=False)
        self._head: int = Value("i", 0, lock=False)
        self._tail: int = Value("i", 0, lock=False)
        self._size: int = Value("i", 0, lock=False)
        self._allow_write: bool = Value(c_bool, True, lock=False)
        self._safe_tail: int = Value("i", 0, lock=False)
        self._safe_size: int = Value("i", 0, lock=False)
        self._sum: float = Value("d", 0, lock=False)
        self._evicted: List[bool] = Array(
            c_bool, [False] * capacity, lock=False
        )
        self._elements: List[ValueTransition] = Manager().list(
            [None for _ in range(capacity)]
        )
        self._weights: List[float] = Array("i", [0] * capacity, lock=False)

    def safe_size(
        self, requires_sum: bool = False
    ) -> Union[int, Tuple[int, float]]:
        if requires_sum:
            with self._mutex:
                safe_size = self._safe_size.value
                sum_ = self._sum.value
            return safe_size, sum_
        else:
            with self._mutex:
                safe_size = self._safe_size.value
            return safe_size

    @property
    def size(self) -> int:
        with self._mutex:
            size = self._size.value
        return size

    def block_append(
        self, block: List[ValueTransition], weights: torch.Tensor
    ) -> None:
        block_size = len(block)
        with self._mutex:
            self._cv_size.wait_for(
                predicate=lambda:
                    self._size.value + block_size <= self._capacity.value
                    and self._allow_write.value
            )
            start = self._tail.value
            end = (self._tail.value + block_size) % self._capacity.value
            self._tail.value = end
            self._size.value += block_size
            self._check_size(
                head=self._head.value,
                tail=self._tail.value,
                size=self._size.value
            )
        sum_ = 0
        assert weights.size(0) == block_size
        for i in range(block_size):
            j = (start + i) % self._capacity.value
            self._elements[j] = block[i]
            self._weights[j] = weights[i]
            sum_ += weights[i]
        with self._mutex:
            self._cv_tail.wait_for(
                predicate=lambda: self._safe_tail.value == start)
            self._safe_tail.value = end
            self._safe_size.value += block_size
            self._sum.value += sum_
            self._check_size(
                head=self._head.value,
                tail=self._safe_tail.value,
                size=self._safe_size.value
            )
            self._cv_tail.notify_all()

    def block_pop(self, block_size: int):
        diff = 0
        head = self._head.value
        for _ in range(block_size):
            diff -= self._weights[head]
            self._evicted[head] = True
            head = (head + 1) % self._capacity.value
        with self._mutex:
            self._sum.value += diff
            self._head.value = head
            self._safe_size.value -= block_size
            self._size.value -= block_size
            assert self._safe_size.value >= 0
            self._check_size(
                head=self._head.value,
                tail=self._safe_tail.value,
                size=self._safe_size.value
            )
            self._cv_size.notify_all()

    def save(self, fpath: Text) -> None:
        with self._mutex:
            with open(file=fpath, mode="wb") as stream:
                for i in range(self._size.value):
                    self._elements[i].write(file=stream)

    def extract(self) -> ExtractedData:
        print("starting extract")
        size = self._safe_size.value
        # create data dump
        data: List[ValueTransition] = []
        weights: List[float] = []
        for i in range(size):
            index = (i + self._head) % self._capacity
            data.append(self._elements[index])
            weights.append(self._weights[index])
        # torch.tensor(): copy
        # torch.as_tensor(): reference for np array
        weights_tensor: torch.Tensor = torch.tensor(data=weights)
        batched: ExtractedData = make_batch(transitions=data, device="cpu")
        self.block_pop(size)
        batched.append(weights_tensor)
        return batched

    def update(self, ids: List[int], weights: torch.Tensor) -> None:
        diff = 0
        for i, id_ in enumerate(ids):
            if self._evicted[id_]:
                continue
            diff += weights[i] - self._weights[id_]
            self._weights[id_] = weights[i]
        with self._mutex:
            self._sum.value += diff

    def get_element_and_mark(self, idx: int) -> ValueTransition:
        """accessing elements is never locked, operate safely
        """
        id_ = (self._head.value + idx) % self._capacity.value
        self._evicted[id_] = False
        return self._elements[id_]

    def get_weight(self, idx: int) -> Tuple[float, int]:
        id_ = (self._head.value + idx) % self._capacity.value
        return self._weights[id_], id_

    def _check_size(self, head: int, tail: int, size) -> None:
        if size == 0:
            assert tail == head
        elif tail > head:
            assert tail - head == size, \
                "error: tail - head: %d vs size: %d" % (
                    tail - head, size
                )
        else:
            assert tail + self._capacity.value - head == size, \
                "error: tail - head: %d vs size: %d" % (
                    tail + self._capacity.value - head, size
                )


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta: float,
        prefetch: int,
        seed: int = None,
        use_priority: bool = True,
        compressed_values: bool = False
    ) -> None:
        """
        args:

        alpha: priority exponent
        beta: importance sampling exponent
        """
        self._alpha = alpha
        self._beta = beta
        self._prefetch = prefetch
        self._capacity = capacity
        self._use_priority = use_priority
        self._compressed_values = compressed_values
        self._storage = ConcurrentQueue(int(1.25 * capacity + 0.5))
        # shared memory
        self._num_add = Value("i", 0, lock=True)
        self._m_sampler = Lock()
        if seed is None:
            random.seed(time.time())
        else:
            random.seed(seed)
        self._sampled_ids: List[int] = []
        # self._futures = Queue()

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(
        self,
        sample: Union[ValueTransition, List[ValueTransition]],
        priority: torch.Tensor
    ) -> None:
        assert priority.dim() == 1
        weights = (
            torch.pow(input=priority, exponent=self._alpha)
            if self._use_priority
            else priority
        )
        if isinstance(sample, ValueTransition):
            sample = [sample[i] for i in range(priority.size(0))]
        self._storage.block_append(block=sample, weights=weights)
        self._num_add.value += priority.size(0)

    def sample(
        self,
        batch_size: int,
        device: Text
    ) -> Tuple[ValueTransition, torch.Tensor]:
        # temporary
        assert self._prefetch == 0, "error: async has not been supported"
        assert not self._use_priority, "error: priority has not been supported"
        # end temporary
        assert (not self._use_priority) or (not self._sampled_ids), \
            "error: previous samples' priority has not been updated."
        if self._prefetch == 0:
            batch, priority, self._sampled_ids = self._sample(
                batch_size=batch_size, device=device
            )
            return batch, priority
        # async
        # if self._futures.empty():
        #     batch, priority, self._sampled_ids = self._sample(
        #         batch_size=batch_size, device=device
        #     )
        # else:
        #     # assert self._futures.qsize() == 1
        #     future: Future = self._futures.get()
        #     batch, priority, self._sampled_ids = future.result()
        # executor = ThreadPoolExecutor(max_workers=5)
        # while self._futures.qsize() < self._prefetch:
        #     self._futures.put(
        #         executor.submit(self.sample, batch_size, device)
        #     )
        # return batch, priority

    def update_priority(self, priority: torch.Tensor) -> None:
        if priority.size(0) == 0:
            self._sampled_ids = []
            return
        assert priority.dim() == 1
        assert len(self._sampled_ids) == priority.size(0)
        weights = torch.pow(input=priority, exponent=self._alpha)
        with self._m_sampler:
            self._storage.update(ids=self._sampled_ids, weights=weights)
        self._sampled_ids = []

    def load(
        self, fpath: Text, priority: float, max_size: int, stride: int
    ) -> None:
        with open(file=fpath, mode="rb") as stream:
            priority_tensor = torch.ones(size=(1, )) * priority
            while True:
                break
        raise NotImplementedError

    def save(self, fpath: Text) -> None:
        self._storage.save(fpath=fpath)

    def extract(self) -> ExtractedData:
        """get context of the buffer as a vector of tensors

        taking just in case, if this methood is used callers are not expected
        to sample
        """
        with self._m_sampler:
            data: ExtractedData = self._storage.extract()
            data[-1] = torch.pow(input=data[-1], exponent=1 / self._alpha)
        return data

    def push(self, data: ExtractedData) -> None:
        """push content extracted from another buffer
        """
        weights: torch.Tensor = data.pop()
        elements: ValueTransition = from_vector(tensors=data)
        self.add(sample=[elements], priority=weights)

    def pop_until(self, new_size: int) -> None:
        """pop from the buffer until new_size is left
        """
        size = self._storage.size()
        if size > new_size:
            self._storage.block_pop(block_size=size - new_size)

    def size(self) -> int:
        return self._storage.safe_size(requires_sum=False)

    def num_add(self) -> int:
        return self._num_add.value

    def _sample(self, batch_size: int, device: Text) -> SampleWeightIds:
        if self._use_priority:
            return self._sample_with_priorities(
                batch_size=batch_size, device=device
            )
        else:
            return self._sample_no_priorities(
                batch_size=batch_size, device=device
            )

    def _sample_with_priorities(
        self, batch_size: int, device: Text
    ) -> SampleWeightIds:
        with self._m_sampler:
            size, sum_ = self._storage.safe_size(requires_sum=True)
            # print("size: %d, sum: %f" % (size, sum_))
            # self._storage [0, size) remains static in the subsequent section
            segment = sum_ / batch_size
            def dist(x): return random.uniform(0, x)
            samples: List[ValueTransition] = []
            weights: torch.Tensor = torch.zeros(
                size=(batch_size, ), dtype=torch.float32
            )
            ids: List[int] = []
            acc_sum = 0
            next_idx = 0
            w = 0
            id_ = 0
            for i in range(batch_size):
                rand = dist(x=segment)
                rand = min(sum_ - 0.1, rand)
                print("looking for %d-th %d sample" % (i, batch_size))
                print("\ttarget: %f" % rand)
                while next_idx <= size:
                    if (acc_sum > 0 and acc_sum >= rand) or (next_idx == size):
                        assert next_idx >= 1
                        print("\tfound: %d, %d, %f" % (
                            next_idx - 1, id_, acc_sum
                        ))
                        element = self._storage.get_element_and_mark(
                            idx=next_idx - 1
                        )
                        samples.append(element)
                        weights[i] = w
                        ids.append(id_)
                        break
                    if next_idx == size:
                        # this should never happened due to the hackky if above
                        print("next_idx: %d / %d" % (next_idx, size))
                        print("acc_sum: %f, sum: %f, rand: %f" % (
                            acc_sum, sum_, rand
                        ))
                        assert False
                    w, id_ = self._storage.get_weight(idx=next_idx)
                    acc_sum += w
                    next_idx += 1
            assert len(samples) == batch_size
            # pop storage if full
            size = self._storage.size
            if size > self._capacity:
                self._storage.block_pop(block_size=size - self._capacity)
        weights /= sum_
        weights = torch.pow(input=size * weights, exponent=-self._beta)
        weights /= weights.sum()
        if device != "cpu":
            weights = weights.to(device=torch.device(device=device))
        batch = make_batch(transitions=samples, device=device)
        if self._compressed_values:
            batch.values = dequantize(tensor=batch.values)
        return batch, weights, ids

    def _sample_no_priorities(
        self, batch_size: int, device: Text
    ) -> SampleWeightIds:
        with self._m_sampler:
            size = self._storage.safe_size()
            # print("size: %d" % size)
            # self._storage [0, size) remains static in the subsequent section
            def dist(x): return random.randrange(0, x)
            samples: List[ValueTransition] = []
            weights: torch.Tensor = torch.zeros(
                size=(batch_size, ), dtype=torch.float32
            )
            ids: List[int] = []
            for i in range(batch_size):
                idx = dist(size)
                weights[i], id_ = self._storage.get_weight(idx=idx)
                ids.append(id_)
                element = self._storage.get_element_and_mark(idx=idx)
                samples.append(element)
            assert len(samples) == batch_size
            # pop storage if full
            size = self._storage.size
            if size > self._capacity:
                self._storage.block_pop(block_size=size - self._capacity)
        batch = make_batch(transitions=samples, device=device)
        if self._compressed_values:
            batch.values = dequantize(tensor=batch.values)
        return batch, weights, ids


ValuePrioritizedReplay = PrioritizedReplay  # DataType = ValueTransition
