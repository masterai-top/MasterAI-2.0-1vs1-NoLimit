#include "samples.h"

//shuffled sequence containing integers 0, 1, ..., n-1 each l times
vector<int> gen_seq(int &l, int &n)
{
    vector<int> seq = vector<int>(l * n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < l; ++j)
        {
            seq.at(i * l + j) = i;
        }
    }

    random_device rd;
    mt19937 g(rd());
    shuffle(seq.begin(), seq.end(), g);
    return seq;
}

//N = 30 random samples | C = choose 6 | n = 13 elements
vector<vector<int> > gen_samples(int N, int C, int n)
{
    int l = 1 + (int) (N * C) / n;
    vector<int> seq = gen_seq(l, n);
    vector<int> heads(n, 0);
    vector<int> size(n, 0);
    int filled = 0;
    vector<vector<int> > samps(N + 1, vector<int>());

    int c = 0;
    for (int x : seq)
    {
        c++;
        if (heads[x] < filled)
        {
            heads[x] = filled;
        }

        //cout << to_string(c) + ' ' + to_string(heads[x]) << endl;

        if (heads[x] > N)
        {
            heads[x] = N;
        }
        samps[heads[x]].push_back(x);
        if (samps[heads[x]].size() == (size_t)C)
        {
            filled++;
        }

        if (filled == N)
        {
            break;
        }

        heads[x]++;
    }

    samps.pop_back();
    return samps;
}
