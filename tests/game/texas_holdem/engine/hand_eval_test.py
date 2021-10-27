from game.texas_holdem.engine.deck import Deck
from game.texas_holdem.engine.hand_eval import HandEvaluator

def test_hand_eval():
    deck = Deck()
    num_sims = int(1e3)
    num_player_1_wins = 0
    num_player_2_wins = 0
    num_ties = 0
    for _ in range(num_sims):
        print("--->")
        deck.restore()
        deck.shuffle()
        hole_cards_1 = deck.draw_cards(2)
        hole_cards_2 = deck.draw_cards(2)
        board_cards = deck.draw_cards(5)
        print("player 1 hole cards: %s %s" % tuple(hole_cards_1))
        print("player 2 hole cards: %s %s" % tuple(hole_cards_2))
        print("board cards: %s %s %s %s %s" % tuple(board_cards))

        player_1_hand = HandEvaluator.eval_hand(hole_cards_1, board_cards)
        player_2_hand = HandEvaluator.eval_hand(hole_cards_2, board_cards)
        if player_1_hand == player_2_hand:
            print("tie")
            num_ties += 1
        else:
            print(
                "player 1 wins"
                if player_1_hand > player_2_hand else "player 2 wins"
            )
            if player_1_hand > player_2_hand:
                num_player_1_wins += 1
            else:
                num_player_2_wins += 1

    print("--->")
    print("player 1 wins prob: %.2f%%, player 2 wins prob: %.2f%%" % (
        num_player_1_wins / num_sims * 100, num_player_2_wins / num_sims * 100
    ))
    print("tie prob: %.2f%%" % (num_ties / num_sims * 100))
