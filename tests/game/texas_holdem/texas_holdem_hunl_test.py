import game
from game.texas_holdem.texas_holdem_hunl import Game, PartialPublicState

game = Game(
    num_hole_cards=2,
    num_deck_cards=52,
    stack_size=10,
    small_blind=1,
    big_blind=2
)
root = game.get_initial_state()


class TestGame:

    def test_unpack_action(self):
        pass

    def test_act(self):
        pass

    def test_root(self):
        assert root.player_id == game.BANKER
        assert game.action_id2str[root.last_action] == game.INITIAL_ACTION
        assert root.player_bet_to == game.small_blind
        assert root.oppo_bet_to == game.big_blind
        assert len(root.board_cards) == game.NUM_BOARD_CARDS_PER_STREET[0]
        assert root.is_first_action
        assert not root.is_terminal

    def test_get_feasible_actions(self):
        feasible_actions = [True] * game.num_actions
        feasible_actions[2] = False
        assert feasible_actions == game.get_feasible_actions(root)
