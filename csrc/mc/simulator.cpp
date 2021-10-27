#include "simulator.h"
#include "tools.h"
#include "samples.h"
#include "cards.h"
using namespace std;

#define ROLLLOG_DEBUG cout
#define ROLLLOG_ERROR cerr

map<int, string> mSuitTran =
{
    { 0, "d"},
    { 16, "c"},
    { 32, "h"},
    { 48, "s"},
};

std::vector<short> vCards =
{
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
};

Simulator::Simulator(string &file)
{
    init(file);
}

Simulator::~Simulator()
{

}

int Simulator::init(string &algFile)
{
    //
    cTable = gen_combo_table(52, 5);
    ROLLLOG_DEBUG << "cTable size:" << cTable.size() << endl;
    //
    replace =
    {
        0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 0, 4,
        6, 3, 5, 2, 4, 1, 3, 1, 1, 2, 3, 3, 4,
        4, 5, 2, 2, 4, 6, 3, 5, 3, 3, 4, 5, 4, 4
    };
    table = read_vect(algFile.c_str());
    if (table.empty())
    {
        return -1;
    }

    return 0;
}

int Simulator::to_ckey(const vector<int> &hand)
{
    int key = 0;
    for (uint32_t i = 0; i < hand.size(); ++i)
    {
        key += cTable[(i + 1) * 53 + hand[i]];
    }

    return key;
}

vector<int> Simulator::get_remaining(vector<int> &comm_hand, vector<vector<int>> &known_hands)
{
    vector<int> remaining;
    vector<int> filled(52);

    for (int card : comm_hand)
    {
        card = card <= 0 ? 1 : card;
        filled[card - 1] = 1;
    }

    for (vector<int> known_hand : known_hands)
    {
        for (int card : known_hand)
        {
            card = card <= 0 ? 1 : card;
            filled[card - 1] = 1;
        }
    }

    for (int i = 0; i < 52; ++i)
    {
        if (!filled[i])
        {
            remaining.push_back(i);
        }
    }

    return remaining;
}

vector<vector<int>> Simulator::fill_empty(int N, vector<int> &comm_hand, vector<vector<int>> &known_hands, int players_unknown)
{
    vector<int> remaining = get_remaining(comm_hand, known_hands);
    int c = players_unknown * 2 + 5 - comm_hand.size();
    vector<vector<int>> samples = gen_samples(N, c, remaining.size());
    for (vector<int> &sample : samples)
    {
        for (int i = 0; i < c; ++i)
        {
            sample[i] = remaining[(unsigned int)sample[i] > remaining.size() - 1 ?  remaining.size() - 1 : sample[i]];
        }
    }

    return samples;
}

int Simulator::evaluate_selection(vector<int> selection)
{
    sort(selection.begin(), selection.end());

    vector<int> hand(5);
    for (int i = 0; i < 5; ++i)
    {
        hand[i] = selection[i + 2];
    }

    int key = (unsigned int)to_ckey(hand) > table.size() - 1 ? table.size() - 1 : to_ckey(hand);
    if(key < 0 || table.size() < 1) return 0;
    int max_score = table[key];
    for (int j = 0; j < 40; j += 2)
    {
        int ix_k = (replace[j] + 1) * 53;
        key += cTable[ix_k + selection[replace[j + 1]]] - cTable[ix_k + hand[replace[j]]];
        hand[replace[j]] = selection[replace[j + 1]];
        key = (key < 0 || key > (int)table.size() - 1) ? (int)table.size() - 1 : key;
        if(key < 0) return max_score;
        max_score = max(max_score, table[key]);
    }

    return max_score;
}

void Simulator::update_winners(int my_val, int &max_val, int ix, vector<int> &winners)
{
    if (my_val > max_val)
    {
        winners = {ix};
        max_val = my_val;
    }
    else if (my_val == max_val)
    {
        winners.push_back(ix);
    }
}

vector<int> Simulator::simulate(vector<int> &selection, vector<vector<int>> &known_hands, vector<int> &sample, int start)
{
    vector<int> winners;
    int max_val = 0;

    for (uint32_t i = 0; i < known_hands.size(); i++)
    {
        selection[5] = known_hands[i][0];
        selection[6] = known_hands[i][1];
        update_winners(evaluate_selection(selection), max_val, i, winners);
    }

    // int opp_max_val = 0;
    // int opp_max = 0;
    for (uint32_t i = 0; start + i * 2 < sample.size(); i ++)
    {
        selection[5] = sample[start + i * 2];
        selection[6] = sample[start + 1 + i * 2];
        update_winners(evaluate_selection(selection), max_val, known_hands.size() + i, winners);
    }

    return winners;
}

vector<vector<int>> Simulator::calculate(int N, vector<int>comm_hand, vector<vector<int>>known_hands, int players_unknown)
{
    vector<vector<int>> samples = fill_empty(N, comm_hand, known_hands, players_unknown);
    vector<int> selection(7);
    vector<vector<int>> results(known_hands.size() + players_unknown, vector<int>(2, 0));

    for (uint32_t i = 0; i < comm_hand.size(); ++i)
    {
        selection[i] = comm_hand[i];
    }

    int s_comm = 5 - (int)comm_hand.size();
    int cnt = 0;
    for (vector<int> sample : samples)
    {
        cnt++;

        for (int i = 0; i < s_comm; ++i)
        {
            selection[4 - i] = sample[i];
        }

        vector<int> winners = simulate(selection, known_hands, sample, s_comm);
        if (winners.size() == 1)
        {
            results[winners[0]][0]++;
        }
        else
        {
            for (int winner : winners)
            {
                results[winner][1]++;
            }
        }
    }

    return results;
}

void Simulator::print_results(int N, vector<vector<int>> hands, vector<vector<int>> results, std::vector<float> &vWins)
{
    for (uint32_t i = 0; i < hands.size(); i++)
    {
        format_result(N, results[i], vWins);
    }

    int unknown = results.size() - hands.size();
    if (unknown > 0)
    {
        float wins = 0;
        float factor = 100.0 / N / unknown;
        for (uint32_t i = hands.size(); i < results.size(); i++)
        {
            wins += results[i][0];
        }

        vWins.push_back( wins * factor);
    }
}

void Simulator::format_result(int N, vector<int> result, std::vector<float> &vWins)
{
    vWins.push_back(result[0] * 100.0 / N);
}

std::vector<float> Simulator::compute_probabilities(int N, vector<string> comm_hand, vector<vector<string>> known_hands, int players_unknown)
{
    std::vector<vector<int>> vKnownHands;
    for (vector<string> k : known_hands)
    {
        vKnownHands.push_back(convert_hand(k));
    }

    std::vector<int> vCommHands = convert_hand(comm_hand);

    std::vector<float> v;
    std::vector<vector<int>> results = calculate(N, vCommHands, vKnownHands, players_unknown);
    print_results(N, vKnownHands, results, v);
    return v;
}
