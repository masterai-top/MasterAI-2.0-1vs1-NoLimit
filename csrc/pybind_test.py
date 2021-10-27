#!/usr/bin/env python
# -- coding utf-8 --

import sys
import os
import random
import time

path_current = os.path.dirname(os.path.abspath(__file__))
path_build = os.path.join(path_current, "build")
sys.path.append(path_build)

import numpy as np
import mc as test

if __name__ == "__main__":
    #
    random.seed(time.time())
    #
    lookup_table = "./csrc/config/lookup_tablev3.bin"

    #
    suit = ["c", "d", "h", "s"]
    rank = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    cards = []
    for s in suit:
        for r in rank:
            c = str(r + s)
            cards.append(c)
    # print(cards)

    sim = test.Simulator(lookup_table)
    for i in range(len(cards)):
        slice = random.sample(cards, 5)
        # print("random cards: %s" % (str(slice)))

        try:
            test_num = 10000
            players_unknown = 1
            known_hands = [[slice[0], slice[1]]]
            known_hands = [["6s", "6c"]]
            print("known_hands: %s" % (known_hands))
            comm_hand = [slice[2], slice[3], slice[4]]
            comm_hand = []
            print("comm_hansd: %s" % (comm_hand))
            result = sim.compute_probabilities(test_num, comm_hand, known_hands, players_unknown)
            start = time.time()
            print("{} {}".format(str(known_hands[0]), result[0]))
            print("['??', '??] {}".format(result[1]))
            over = time.time()
            print("take time: %ld" % int(round((over - start) * 1000000)))
        except:
            print("test failed:  cards=%s" % str(slice))
