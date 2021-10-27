#ifndef TABLES_H
#define TABLES_H

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

//
int evaluate(const vector<int> &hand1);
//
vector<int> gen_lookup_table();
//
int to_key(const vector<int> &hand);
//
int to_ckey(const vector<int> &c_table, const vector<int> &hand);

#endif
