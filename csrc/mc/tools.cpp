#include "tools.h"

void print(vector<int> const &input)
{
    std::copy(input.begin(), input.end(), std::ostream_iterator<int>(std::cout, " "));
}

ifstream::pos_type filesize(const char *filename)
{
    ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    ifstream::pos_type ret =  in.tellg();
    return ret;
}

void write_vect(vector<int> vect, string name)
{
    fstream f;
    int *arr = &vect[0];
    f.open(name, ios::out | ios::binary);
    if (f)
    {
        f.write(reinterpret_cast<char *>(arr), vect.size() * 4);
        f.close();
    }
    else
    {
        cout << "Error" << endl;
    }
}

vector<int> read_vect(const char *name)
{
    fstream f;
    string loc = name;
    f.open(loc, ios::in | ios::binary);
    int fsize = filesize(loc.c_str());
    vector<int> vect(fsize / 4);
    int *arr = &vect[0];
    if (f)
    {
        f.read(reinterpret_cast<char *>(arr), fsize);
        f.close();
    }
    return vect;
}
