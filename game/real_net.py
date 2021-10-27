import game
import torch
from typing import List, Text
from .net import INet
from .texas_holdem.texas_holdem_hunl import Game
from .subgame_solving import Pair, SubgameSolvingParams, build_solver, deserialize_query


class ZeroOutputNet(INet):
    def __init__(self, output_size: int, verbose: bool) -> None:
        self._output_size = output_size
        self._verbose = verbose

    def compute_values(self, query: torch.Tensor) -> torch.Tensor:
        num_queries = query.size(0)
        if self._verbose:
            print(
                "called ZeroOutputNet.handle_nn_query() with num_queries = %d"
                % num_queries
            )
        return torch.zeros(num_queries, self._output_size)

    def add_training_example(
        self,
        query: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        if self._verbose:
            print(
                "called ZeroOutputNet.cvfnet_update() with num_queries = %d"
                % query.size(0)
            )


class TorchScriptNet(INet):
    def __init__(self, path: Text, device: Text) -> None:
        self._device = device
        try:
            self._module: torch.jit.ScriptModule = torch.jit.load(
                f=path,
                map_location=None if torch.cuda.is_available()
                else torch.device("cpu")
            )
        except Exception as e:
            print("error loading the model: %s" % path)
            print(e)
            return
        print("loaded: %s" % path)
        self._module.to(device=device)

    def compute_values(self, query: torch.Tensor) -> torch.Tensor:
        input_: torch.Tensor = query.to(device=self._device)
        results: torch.Tensor = self._module.forward(input_)
        return results.to(device="cpu")

    def add_training_example(self, query: torch.Tensor, values: torch.Tensor) -> None:
        raise RuntimeError("cannot update TorchScript model, only query")


class TorchNet(INet):
    def __init__(self, path: Text, device: Text) -> None:
        self._device = device
        try:
            self._module: torch.nn.Module = torch.load(
                f=path,
                map_location=None if torch.cuda.is_available()
                else torch.device("cpu")
            )
        except Exception as e:
            print("error loading the model: %s" % path)
            print(e)
            return
        print("loaded: %s" % path)
        self._module.eval()
        self._module.to(device=device)

    def compute_values(self, query: torch.Tensor) -> torch.Tensor:
        input_: torch.Tensor = query.to(device=self._device)
        results: torch.Tensor = self._module(input_)
        return results.to(device="cpu").detach()

    def add_training_example(self, query: torch.Tensor, values: torch.Tensor) -> None:
        raise RuntimeError("cannot update Torch model, only query")


class OracleNetSolver(INet):
    def __init__(self, game: Game, params: SubgameSolvingParams) -> None:
        self._game = game
        self._params = params

    def compute_values(self, query: torch.Tensor) -> torch.Tensor:
        num_queries = query.size(0)
        values = List[torch.Tensor]
        for i in range(num_queries):
            input_: List[float] = query[i].numpy()
            values_: List[float] = self._compute_values(query=input_)
            values.append(torch.tensor(values_).to(dtype=torch.float32))
        return torch.stack(tensors=values, dim=0)

    def _compute_values(self, query: List[float]) -> List[float]:
        traverser, state, beliefs1, beliefs2 = deserialize_query(
            game=game, query=query
        )
        beliefs: Pair = [beliefs1, beliefs2]
        solver = build_solver(
            game=game, params=self._params, root=state, beliefs=beliefs
        )
        solver.multistep()
        return solver.get_hand_values(player_id=traverser)

    def add_training_example(
        self, query: torch.Tensor, values: torch.Tensor
    ) -> None:
        raise RuntimeError("not supported")


def create_zero_net(output_size: int, verbose: bool = True) -> INet:
    """creates a net that outputs zeros on query and nothing on update
    """
    return ZeroOutputNet(output_size=output_size, verbose=verbose)


def create_torchscript_net(path: Text, device: Text = "cpu") -> INet:
    """creat eval-only connector from the net in the path
    """
    return TorchScriptNet(path=path, device=device)


def create_torch_net(path: Text, device: Text = "cpu") -> INet:
    """creat eval-only connector from the net in the path
    """
    return TorchNet(path=path, device=device)


def create_oracle_value_predictor(
        game: Game, params: SubgameSolvingParams
) -> INet:
    """create virtual value net that run a solver for each query
    """
    return OracleNetSolver(game=game, params=params)
