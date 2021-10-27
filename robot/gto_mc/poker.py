# This code is from Peter Norvig #
import random


def poker(hands):
    """return a list of winning hands:
    poker([hand,...]) => [hand,...]
    """
    return allmax(hands, key=hand_rank)


def allmax(iterable, key=lambda x: x):
    "Return a list of all items equal to the max of the iterable."
    maxi = max(iterable, key=key)
    max_value = key(maxi)
    return [item for item in iterable if key(item) == max_value]


def hand_rank(hand):
    """return a value indicating how high the hand ranks
    """
    # counts is the count of each rank
    # ranks lists corresponding ranks
    # E.g. "7 T 7 9 7" => counts = (3, 1, 1); ranks = (7, 10, 9)
    groups = group(["--23456789TJQKA".index(r) for r, s in hand])
    counts, ranks = unzip(groups)
    if ranks == (14, 5, 4, 3, 2):
        ranks = (5, 4, 3, 2, 1)
    straight = len(ranks) == 5 and max(ranks) - min(ranks) == 4
    flush = len(set([s for r, s in hand])) == 1
    return (
        9 if (5, ) == counts else
        8 if straight and flush else
        7 if (4, 1) == counts else
        6 if (3, 2) == counts else
        5 if flush else
        4 if straight else
        3 if (3, 1, 1) == counts else
        2 if (2, 2, 1) == counts else
        1 if (2, 1, 1, 1) == counts else
        0), ranks


def group(items):
    """return a list of [(count, x)...]
    highest count first, the highest x first
    """
    groups = [(items.count(x), x) for x in set(items)]
    return sorted(groups, reverse=True)


def unzip(pairs):
    return list(zip(*pairs))


def card_ranks(hand):
    "Return a list of the ranks, sorted with higher first."
    # ranks = ["--23456789TJQKA".index(r) for r, s in hand]
    # ranks = [{"A":14,
    #           "K":13,
    #           "Q":12,
    #           "J":11,
    #           "T":10,
    #           }.get(r,r) for r, s in hand]
    ranks = [14 if r == "A" else
             13 if r == "K" else
             12 if r == "Q" else
             11 if r == "J" else
             10 if r == "T" else
             int(r)
             for r, s in hand]
    ranks.sort(reverse=True)
    return ranks if ranks != [14, 5, 4, 3, 2] else [5, 4, 3, 2, 1]


def straight(ranks):
    "Return True if the ordered ranks form a 5-card straight."
    return sum(ranks) - min(ranks)*5 == 10


def flush(hand):
    "Return True if all the cards have the same suit."
    suits = [s for r, s in hand]
    return len(set(suits)) == 1


def two_pair(ranks):
    """If there are two pair, return the two ranks as a
    tuple: (highest, lowest); otherwise return None."""
    result = [r for r in set(ranks) if ranks.count(r) == 2]
    if len(result) == 2:
        return (max(result), min(result))


def kind(n, ranks):
    """Return the first rank that this hand has exactly n of.
    Return None if there is no n-of-a-kind in the hand."""
    for r in set(ranks):
        if ranks.count(r) == n:
            return r
    return None


deck = [r+s for r in "23456789TJQKA" for s in "SHDC"]


def deal(numhands, n=5, deck=[r+s for r in "23456789TJQKA" for s in "SHDC"]):
    "Return a list of numhands hands consisting of n cards each"
    random.shuffle(deck)
    deck = iter(deck)
    return [[next(deck) for card in range(n)] for hand in range(numhands)]


def test():
    "Test cases for the functions in poker program"
    sf = "6C 7C 8C 9C TC".split()  # Straight Flush
    fk = "9D 9H 9S 9C 7D".split()  # Four of a Kind
    fh = "TD TC TH 7C 7D".split()  # Full House
    s1 = "AS 2S 3S 4S 5C".split()  # A-5 straight
    s2 = "2C 3C 4C 5S 6S".split()  # 2-6 straight
    s3 = "TC JC QC KS AS".split()  # 10-A straight
    tp = "5S 5D 9H 9C 6S".split()  # two pair
    ah = "AS 2S 3S 4S 6C".split()  # A high
    sh = "2S 3S 4S 6C 7D".split()  # 7 high
    assert poker([sf, fk, fh]) == [sf]
    assert poker([fk, fh]) == [fk]
    assert poker([fh, fh]) == [fh, fh]
    assert poker([sf]) == [sf]
    assert poker([sf] + 99*[fh]) == [sf]
    assert poker([s1, s2]) == [s2]
    assert poker([s1, tp]) == [s1]

    # assert hand_rank(sf) == (8, 10)
    # assert hand_rank(fk) == (7, 9, 7)
    # assert hand_rank(fh) == (6, 10, 7)
    # assert hand_rank(s1) == (4, 5)
    # assert hand_rank(s3) == (4, 14)

    assert card_ranks(sf) == [10, 9, 8, 7, 6]
    assert card_ranks(fk) == [9, 9, 9, 9, 7]
    assert card_ranks(fh) == [10, 10, 10, 7, 7]
    assert card_ranks(["AC", "3D", "4S", "KH"]) == [14, 13, 4, 3]

    # Ace-high beats 7-high
    assert (card_ranks(["AS", "2C", "3D", "4H", "6S"]) >
            card_ranks(["2D", "3S", "4C", "6H", "7D"]))
    # 5-straight loses to 6-straight
    assert (card_ranks(["AS", "2C", "3D", "4H", "5S"]) <
            card_ranks(["2D", "3S", "4C", "5H", "6D"]))

    fkranks = card_ranks(fk)
    tpranks = card_ranks(tp)

    assert kind(4, fkranks) == 9
    assert kind(3, fkranks) == None
    assert kind(2, fkranks) == None
    assert kind(1, fkranks) == 7

    assert two_pair(tpranks) == (9, 5)
    assert two_pair([10, 10, 5, 5, 2]) == (10, 5)

    assert straight([9, 8, 7, 6, 5]) == True
    assert straight([9, 8, 8, 6, 5]) == False

    assert flush(sf) == True
    assert flush(fk) == False

    return "tests pass"
