from functools import reduce
from itertools import groupby
from typing import List, Union

from .card import Card


class HandEvaluator(object):
    HIGH_CARD = 0
    ONE_PAIR = 1 << 8
    TWO_PAIR = 1 << 9
    THREE_OF_A_KIND = 1 << 10
    STRAIGHT = 1 << 11
    FLUSH = 1 << 12
    FULL_HOUSE = 1 << 13
    FOUR_OF_A_KIND = 1 << 14
    STRAIGHT_FLUSH = 1 << 15

    HAND_STRENGTH_MAP = {
        HIGH_CARD: "HIGHCARD",
        ONE_PAIR: "ONEPAIR",
        TWO_PAIR: "TWOPAIR",
        THREE_OF_A_KIND: "THREEOFAKIND",
        STRAIGHT: "STRAIGHT",
        FLUSH: "FLUSH",
        FULL_HOUSE: "FULLHOUSE",
        FOUR_OF_A_KIND: "FOUROFAKIND",
        STRAIGHT_FLUSH: "STRAIGHTFLUSH"
    }

    RANK_H_BIT = 4
    HAND_BIT = 8

    @classmethod
    def eval_hand(cls, hole_cards: List[Card], board_cards: List[Card]) -> int:
        """[Bit flg of hand][rank1 (4bit)][rank2 (4bit)]

        high card & hole card 3, 4  =>           100 0011
        one pair of rank 3          =>        1 0011 0000
        two pair of rank A, 4       =>       10 1110 0100
        three card of rank 9        =>      100 1001 0000
        straight of rank 10         =>     1000 1010 0000
        flush of rank 5             =>    10000 0101 0000
        full house of rank 3, 4     =>   100000 0011 0100
        four card of rank 2         =>  1000000 0010 0000
        straight flush of rank 7    => 10000000 0111 0000
        """
        hole_flag = cls._eval_hole_cards(hole_cards)
        hand_flag = cls._eval_hand(
            hole_cards, board_cards
        ) << HandEvaluator.HAND_BIT
        return hand_flag | hole_flag

    @classmethod
    def _eval_hand(
        cls,
        hole_cards: List[Card],
        board_cards: List[Card]
    ) -> int:
        cards = hole_cards + board_cards
        # straight flush
        is_straight_flush, rank = cls._eval_straight_flush(cards)
        if is_straight_flush:
            return HandEvaluator.STRAIGHT_FLUSH | rank
        # four card
        is_four_card, rank = cls._eval_four_card(cards)
        if is_four_card:
            return HandEvaluator.FOUR_OF_A_KIND | rank
        # full house
        is_full_house, rank = cls._eval_full_house(cards)
        if is_full_house:
            return HandEvaluator.FULL_HOUSE | rank
        # flush
        is_flush, rank = cls._eval_flush(cards)
        if is_flush:
            return HandEvaluator.FLUSH | rank
        # straight
        is_straight, rank = cls._eval_straight(cards)
        if is_straight:
            return HandEvaluator.STRAIGHT | rank
        # three card
        is_three_card, rank = cls._eval_three_card(cards)
        if is_three_card:
            return HandEvaluator.THREE_OF_A_KIND | rank
        # two pair
        is_two_pair, rank = cls._eval_two_pair(cards)
        if is_two_pair:
            return HandEvaluator.TWO_PAIR | rank
        # one pair
        is_one_pair, rank = cls._eval_one_pair(cards)
        if is_one_pair:
            return HandEvaluator.ONE_PAIR | rank
        # high card
        return HandEvaluator.HIGH_CARD | cls._eval_hole_cards(cards)

    @classmethod
    def _eval_hole_cards(cls, cards: List[Card]) -> int:
        ranks = sorted([card.rank for card in cards])
        return ranks[1] << HandEvaluator.RANK_H_BIT | ranks[0]

    @classmethod
    def _one_pair(cls, cards: List[Card]) -> int:
        rank = 0
        # memory
        # each bit presents whether a rank is in cards
        # 1, a rank is in cards; 0, otherwise
        mem = 0
        for card in cards:
            mask = 1 << card.rank
            if mem & mask != 0:
                rank = max(rank, card.rank)
            mem |= mask
        return rank

    @classmethod
    def _two_pair(cls, cards: List[Card]) -> List[int]:
        ranks = []
        mem = 0
        for card in cards:
            mask = 1 << card.rank
            if mem & mask != 0:
                ranks.append(card.rank)
            mem |= mask
        return sorted(ranks, reverse=True)[: 2]

    @classmethod
    def _three_card(cls, cards: List[Card]) -> int:
        rank = 0
        mem = reduce(
            lambda mem, card: mem + (1 << (card.rank - 1) * 3),
            cards, 0
        )
        for r in range(2, 15):
            mem >>= 3
            # get 3 bits
            count = mem & 7
            if count >= 3:
                rank = r
        return rank

    @classmethod
    def _straight(cls, cards: List[Card]) -> int:
        rank = 0
        mem = reduce(lambda mem, card: mem | 1 << card.rank, cards, 0)
        # acc & (mem >> (r + i) & 1) for i in range(5)
        # check if the 5 consecutive bits in the memory are all 1's
        def _is_straight(acc, i): return acc & (mem >> (r + i) & 1) == 1
        for r in range(2, 15):
            if reduce(_is_straight, range(5), True):
                rank = r
        return rank

    @classmethod
    def _flush(cls, cards: List[Card]) -> int:
        rank = 0
        def _rank(card): return card.rank
        def _suit(card): return card.suit
        for suit, group in groupby(sorted(cards, key=_suit), key=_suit):
            group = list(group)
            if len(group) >= 5:
                card = max(group, key=_rank)
                rank = max(rank, card.rank)
        return rank

    @classmethod
    def _full_house(cls, cards: List[Card]) -> List[Union[int, None]]:
        rank = 0
        def _rank(card): return card.rank
        three_card_ranks = []
        two_card_ranks = []
        for rank, group in groupby(sorted(cards, key=_rank), key=_rank):
            group = list(group)
            if len(group) >= 3:
                three_card_ranks.append(rank)
            elif len(group) >= 2:
                two_card_ranks.append(rank)
        if len(three_card_ranks) == 2:
            two_card_ranks.append(min(three_card_ranks))

        def max_(l): return 0 if len(l) == 0 else max(l)
        return max_(three_card_ranks), max_(two_card_ranks)

    @classmethod
    def _four_card(cls, cards: List[Card]) -> int:
        def _rank(card): return card.rank
        for rank, group in groupby(sorted(cards, key=_rank), key=_rank):
            group = list(group)
            if len(group) >= 4:
                return rank
        return 0

    @classmethod
    def _straight_flush(cls, cards: List[Card]):
        rank = 0
        def _suit(card): return card.suit
        for suit, group in groupby(sorted(cards, key=_suit), key=_suit):
            group = list(group)
            if len(group) >= 5:
                rank = cls._straight(group)
        return rank

    @classmethod
    def _eval_one_pair(cls, cards: List[Card]) -> List[Union[bool, int]]:
        rank = cls._one_pair(cards)
        return rank != 0, rank << HandEvaluator.RANK_H_BIT

    @classmethod
    def _eval_two_pair(cls, cards: List[Card]) -> List[Union[bool, int]]:
        ranks = cls._two_pair(cards)
        len_ranks = len(ranks)
        is_two_pair = len_ranks == 2
        ranks = ranks if is_two_pair else ranks + [0] * (2 - len_ranks)
        return is_two_pair, ranks[0] << HandEvaluator.RANK_H_BIT | ranks[1]

    @classmethod
    def _eval_three_card(cls, cards: List[Card]) -> List[Union[bool, int]]:
        rank = cls._three_card(cards)
        return rank != 0, rank << HandEvaluator.RANK_H_BIT

    @classmethod
    def _eval_straight(cls, cards: List[Card]) -> List[Union[bool, int]]:
        rank = cls._straight(cards)
        return rank != 0, rank << HandEvaluator.RANK_H_BIT

    @classmethod
    def _eval_flush(cls, cards: List[Card]) -> List[Union[bool, int]]:
        rank = cls._flush(cards)
        return rank != 0, rank << HandEvaluator.RANK_H_BIT

    @classmethod
    def _eval_full_house(cls, cards: List[Card]) -> List[Union[bool, int]]:
        three_card_rank, two_card_rank = cls._full_house(cards)
        return bool(three_card_rank and two_card_rank), \
            three_card_rank << HandEvaluator.RANK_H_BIT | two_card_rank

    @classmethod
    def _eval_four_card(cls, cards: List[Card]) -> List[Union[bool, int]]:
        rank = cls._four_card(cards)
        return rank != 0, rank << HandEvaluator.RANK_H_BIT

    @classmethod
    def _eval_straight_flush(cls, cards: List[Card]) -> List[Union[bool, int]]:
        rank = cls._straight_flush(cards)
        return rank != 0, rank << HandEvaluator.RANK_H_BIT
