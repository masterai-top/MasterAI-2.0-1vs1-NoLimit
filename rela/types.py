import numpy as np
import sys
import torch
from typing import List, Text, BinaryIO, Tuple


class ValueTransition:
    def __init__(
        self,
        query: torch.Tensor = None,
        values: torch.Tensor = None
    ) -> None:
        self._query = query
        self._values = values

    def to_vector(self) -> List[torch.Tensor]:
        return self._query, self._values

    def __getitem__(self, idx: int):
        assert self._query.dim() == 2 and self._values.dim() == 2
        query = self._query[idx]
        values = self._values[idx]
        return ValueTransition(query=query, values=values)

    def pad_like():
        raise NotImplementedError

    def write(self, file: BinaryIO) -> None:
        query_size = self._query.numel()
        value_size = self._values.numel()
        write_ints(data=[query_size, value_size], file=file)
        torch.save(obj=self._query, f=file)
        torch.save(obj=self._values, f=file)

    def to_device(self, device: Text) -> None:
        d = torch.device(device=device)
        self._query.to(d)
        self._values.to(d)

    @property
    def query(self) -> torch.Tensor:
        return self._query

    @query.setter
    def query(self, value: torch.Tensor) -> None:
        self._query = value

    @property
    def values(self) -> torch.Tensor:
        return self._values

    @values.setter
    def values(self, value: torch.Tensor) -> None:
        self._values = value


def load(file: BinaryIO) -> Tuple[ValueTransition, bool]:
    success: bool = True
    result = ValueTransition()
    query_size = read_int(file=file)
    if query_size is None:
        success = False
        return result, success
    value_size = read_int(file=file)
    query = read_array(file=file, num_items=query_size)
    result.query = torch.from_numpy(ndarray=query)
    values = read_array(file=file, num_items=value_size)
    result.values = torch.from_numpy(ndarray=values)
    return result, success


def from_vector(tensors: List[torch.Tensor]) -> ValueTransition:
    assert len(tensors) == 2
    return ValueTransition(query=tensors[0], values=tensors[1])


def make_batch(transitions: ValueTransition, device: Text) -> ValueTransition:
    query_vec: List[torch.Tensor] = []
    values_vec: List[torch.Tensor] = []
    for i in range(len(transitions)):
        query_vec.append(transitions[i].query)
        values_vec.append(transitions[i].values)
    batch = ValueTransition(
        query=torch.stack(tensors=query_vec, dim=0),
        values=torch.stack(tensors=values_vec, dim=0)
    )
    if device != "cpu":
        batch.to_device(device=device)
    return batch


def read_int(
    file: BinaryIO, endianness: Text = "little", signed: bool = True
) -> int:
    """read an integer from a binary io
    """
    data = file.read(n=sys.int_info.sizeof_digit)
    if not data:
        return None
    data = int.from_bytes(
        bytes=data, byteorder=endianness, signed=signed
    )
    return data


def write_ints(
    data: List[int],
    file: BinaryIO,
    endianness: Text = "little",
    signed: bool = True
) -> None:
    for num in data:
        file.write(num.to_bytes(
            length=sys.int_info.sizeof_digit,
            byteorder=endianness,
            signed=signed
        ))


def write_floats(
    data: List[float], file: BinaryIO
) -> None:
    for num in data:
        file.write(
            s=num.to_bytes
        )
    raise NotImplementedError


def read_array(file: BinaryIO, num_items: int, dtype=np.float32):
    num_bytes = num_items * dtype().nbytes
    return np.frombuffer(buffer=file.read(n=num_bytes), dtype=dtype)


def dequantize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32) / 255
