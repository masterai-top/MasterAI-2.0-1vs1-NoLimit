# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from functools import reduce
from torch import nn


def build_mlp(
    *,
    n_in,
    n_hidden,
    n_layers,
    out_size=None,
    act=None,
    use_layer_norm=False,
    dropout=0,
) -> nn.Module:
    if act is None:
        act = GELU()
    build_norm_layer = (
        lambda: nn.LayerNorm(n_hidden) if use_layer_norm else nn.Sequential()
    )
    build_dropout_layer = (
        lambda: nn.Dropout(dropout) if dropout > 0 else nn.Sequential()
    )

    last_size = n_in
    net = []
    for _ in range(n_layers):
        net.extend(
            [
                nn.Linear(last_size, n_hidden),
                build_norm_layer(),
                act,
                build_dropout_layer(),
            ]
        )
        last_size = n_hidden
    if out_size is not None:
        net.append(nn.Linear(last_size, out_size))
    return nn.Sequential(*net)


def _comb(a, b) -> int:
    """# of combinations
    C^b_a
    """
    numerator = reduce(lambda x, y: x * y, range(a, a - b, -1), 1)
    denominator = reduce(lambda x, y: x * y, range(b, 0, -1), 1)
    return numerator // denominator


def input_size(num_hole_cards, num_deck_cards, card_emb_dim):
    """input size
    agent index         : 1
    acting agent        : 1
    pot                 : 1
    board               : 5 -> card embedding
    infostate beliefs   : 2 x C^2_52
    """
    return 3 + card_emb_dim + 2 * _comb(num_deck_cards, num_hole_cards)


def output_size(num_hole_cards, num_deck_cards, num_acitons=1):
    """output size
    # of infostate beliefs x [# of actions]
    a value net if # of actions = 1, otherwise a policy net
    """
    return num_acitons * _comb(num_deck_cards, num_hole_cards)


class Net(nn.Module):
    def __init__(
        self,
        *,
        n_hole_cards: int = 2,
        n_deck_cards: int = 52,
        n_actions: int = 1,
        n_card_suits: int = 4,
        n_card_ranks: int = 13,
        card_emb_dim: int = 64,
        n_hidden: int = 256,
        use_layer_norm: bool = False,
        dropout: int = 0,
        n_layers: int = 3,
    ) -> None:
        super(Net, self).__init__()
        self._card_emb_dim = card_emb_dim
        self._card_emb = CardEmbedding(
            n_suits=n_card_suits,
            n_ranks=n_card_ranks,
            dim=card_emb_dim
        )
        n_in = input_size(
            num_hole_cards=n_hole_cards, num_deck_cards=n_deck_cards, card_emb_dim=card_emb_dim
        )
        self._body = build_mlp(
            n_in=n_in,
            n_hidden=n_hidden,
            n_layers=n_layers,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        # for a value net, a vector of values for each possible infostate
        # of the indexed agent
        # for a policy net, a probability distribution over the legal actions
        # for each infostate
        self._output = nn.Linear(
            n_hidden if n_layers > 0 else n_in,
            output_size(n_hole_cards, n_deck_cards, n_actions)
        )
        # make initial predictions closer to 0
        with torch.no_grad():
            self._output.weight.data *= 0.01
            self._output.bias *= 0.01

    def forward(self, packed_input: torch.Tensor) -> torch.Tensor:
        """
        args:

        packed_input
        1. a probability distribution over pairs of cards for each player
        2. all public board cards
        3. the amount of money in the pot relative to the stacks of the players
        4. a flag for whether a bet has occurred on this betting round yet
        """
        state_info = packed_input[:, : 3]
        cards = packed_input[:, 3: 8].to(dtype=torch.long)
        beliefs = packed_input[:, 8:]
        card_emb = self._card_emb(cards)
        x = torch.cat([state_info, card_emb, beliefs], dim=1)
        return self._output(self._body(x))


class GELU(nn.Module):
    def forward(self, x):
        return nn.functional.gelu(x)


class CardEmbedding(nn.Module):
    def __init__(self, n_suits: int, n_ranks: int, dim: int) -> None:
        super(CardEmbedding, self).__init__()
        self._n_suits = n_suits
        self._n_ranks = n_ranks
        self._rank = nn.Embedding(num_embeddings=n_ranks, embedding_dim=dim)
        self._suit = nn.Embedding(num_embeddings=n_suits, embedding_dim=dim)
        self._card = nn.Embedding(
            num_embeddings=n_ranks * n_suits, embedding_dim=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_cards = x.shape
        x = x.reshape(shape=(-1, ))
        # shape: cards
        valid = x.ge(0).float()
        x = x.clamp(min=0)
        # shape: cards x dim
        embs = (
            self._card(x)
            + self._rank(torch.div(x, self._n_suits, rounding_mode="floor"))
            + self._suit(torch.fmod(x, self._n_suits))
        )
        embs *= valid.unsqueeze(1)
        # sum across the cards
        return embs.view(size=(batch, num_cards, -1)).sum(dim=1)


if __name__ == "__main__":
    net = Net()
    packed_input = torch.randn(size=(100, 1326 * 2 + 8))
    ret = net(packed_input)
    print(ret)
