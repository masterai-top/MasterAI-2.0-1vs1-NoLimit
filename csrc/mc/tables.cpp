#include "tables.h"
#include "tools.h"
#include "cards.h"

bool replace(vector<int> &arr1, vector<int> &arr2)
{
    return arr1[0] < arr2[0] || (arr1[0] == arr2[0] && arr1[1] <= arr2[1]);
}

vector<vector<int>> retrieve_top(const vector<int> &hand)
{
    vector<vector<int>> top2(2, vector<int>(2));
    int prev = 0;
    for (uint32_t i = 1; i < hand.size(); ++i)
    {
        if (hand[i] != hand[i - 1])
        {
            vector<int> arr = {(int)(i - prev), hand[(int)(i - 1)]};
            if (replace(top2[0], arr))
            {
                top2[1] = top2[0];
                top2[0] = arr;
            }
            else if (replace(top2[1], arr))
            {
                top2[1] = arr;
            }

            prev = i;
        }
    }

    return top2;
}

int numerize_top(const vector<vector<int>> &top2)
{
    int a = top2[0][0];
    int b = top2[1][0];
    int m = 0;

    if (a == 4)
    {
        m = 7;
    }
    else if (a == 3)
    {
        if (b == 2)
        {
            m = 6;
        }
        else
        {
            m = 3;
        }
    }
    else if (a == 2)
    {
        if (b == 2)
        {
            m = 2;
        }
        else
        {
            m = 1;
        }
    }

    return m * 13 * 13 + top2[0][1] * 13 + ((m == 2 || m == 6) ? top2[1][1] : 0);
}

int numerize_global(const vector<int> &hand, const vector<int> &suit)
{
    bool is_flush = true;
    bool is_straight = true;
    for (uint32_t i = 1; i < hand.size(); ++i)
    {
        is_straight &= (hand[i - 1] + 1 == hand[i]);
        is_flush &= (suit[i - 1] == suit[i]);
    }

    int m = 0;
    if (is_straight && is_flush)
    {
        if (hand.back() == 12)
        {
            m = 9;
        }
        else
        {
            m = 8;
        }
    }
    else if (is_straight)
    {
        m = 4;
    }
    else if (is_flush)
    {
        m = 5;
    }

    return m * 13 * 13;
}

int evaluate(const vector<int> &hand2)
{
    vector<int> hand1 (hand2);
    sort(hand1.begin(), hand1.end(), [](int a, int b)
    {
        return a % 13 < b % 13;
    });

    vector<int> hand;
    vector<int> suit;
    for (uint32_t j = 0; j < hand1.size(); ++j)
    {
        hand.push_back(hand1[j] % 13);
        suit.push_back(hand1[j] / 13);
    }

    hand.push_back(-1);
    vector<vector<int>> top2 = retrieve_top(hand);
    hand.pop_back();

    int max_head = max(numerize_top(top2), numerize_global(hand, suit));
    int hand_val = 0;
    int factor = 1;
    for (uint32_t i = 0; i < hand.size(); ++i)
    {
        hand_val += hand[i] * factor;
        factor *= 13;
    }

    return max_head * factor + hand_val;
}

vector<int> pow52 = {1, 52, 52 * 52, 52 * 52 * 52, 52 * 52 * 52 * 52};

int to_key(const vector<int> &hand)
{
    int ret = 0;

    for (uint32_t i = 0; i < hand.size(); ++i)
    {
        ret += pow52[i] * hand[i];
    }

    return ret;
}

int to_ckey(const vector<int> &c_table, const vector<int> &hand)
{
    int key = 0;

    for (uint32_t i = 0; i < hand.size(); ++i)
    {
        key += c_table[(i + 1) * 53 + hand[i]];
    }

    return key;
}

void fill_lookup_table(const vector<int> &c_table, vector<int> &table, vector<int> &hand)
{
    if (hand.size() == 5)
    {
        print(hand);
        cout << endl;
        table[to_ckey(c_table, hand)] = evaluate(hand);
    }
    else
    {
        for (uint32_t i = (hand.empty() ? 0 : (hand.back() + 1)); i <= 52 - ( 5 - hand.size()); ++i)
        {
            hand.push_back(i);
            fill_lookup_table(c_table, table, hand);
            hand.pop_back();
        }
    }
}

vector<int> gen_lookup_table()
{
    vector<int> c_table = gen_combo_table(52, 5);
    vector<int> table(c_table[5 * 53 + 52]);
    vector<int> hand;
    fill_lookup_table(c_table, table, hand);
    return table;
}



