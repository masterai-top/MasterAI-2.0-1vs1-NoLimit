from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfv.models import CardEmbedding


class StrategyNet(nn.Module):
    def __init__(
        self, n_card_types: int, n_bets: int, n_actions: int, dim: int = 256
    ) -> None:
        super(StrategyNet, self).__init__()
        # 1. card branch
        self.card_embeddings = nn.ModuleList([
            CardEmbedding(n_suits=4, n_ranks=13, dim=dim)
            for _ in range(n_card_types)
        ])
        self.card1 = nn.Linear(
            in_features=dim * n_card_types, out_features=dim
        )
        self.card2 = nn.Linear(in_features=dim, out_features=dim)
        self.card3 = nn.Linear(in_features=dim, out_features=dim)
        # 2. bet branch
        self.bet1 = nn.Linear(in_features=n_bets * 2, out_features=dim)
        self.bet2 = nn.Linear(in_features=dim, out_features=dim)
        # 3. combined trunk
        self.comb1 = nn.Linear(in_features=2 * dim, out_features=dim)
        self.comb2 = nn.Linear(in_features=dim, out_features=dim)
        self.comb3 = nn.Linear(in_features=dim, out_features=dim)
        self.action_head = nn.Linear(in_features=dim, out_features=n_actions)

    def forward(
        self, cards: List[torch.Tensor], bets: torch.Tensor
    ) -> torch.Tensor:
        """
        cards: ((Nx2), (Nx3), [(Nx1), (Nx1)]) # (hole, board, [turn, river])
        bets: N x n_bet_feats
        """
        # 1. card branch
        # embed hole, flop and optionally turn and river
        card_embs = []
        for embedding, card_group in zip(self.card_embeddings, cards):
            card_embs.append(embedding(card_group))
        card_embs = torch.cat(card_embs, dim=1)
        x = F.relu(self.card1(card_embs))
        x = F.relu(self.card2(x))
        x = F.relu(self.card3(x))
        # 2. bet branch
        bet_size = bets.clamp(min=0, max=1e6)
        bet_occurred = bets.ge(0)
        bet_feats = torch.cat([bet_size, bet_occurred.float()], dim=1)
        y = F.relu(self.bet1(bet_feats))
        y = F.relu(self.bet2(y) + y)
        # 3. combined trunk
        z = torch.cat([x, y], dim=1)
        z = F.relu(self.comb1(z))
        z = F.relu(self.comb2(z) + z)
        z = F.relu(self.comb3(z) + z)
        # (z - mean) / std
        z = F.normalize(z)
        return self.action_head(z)
