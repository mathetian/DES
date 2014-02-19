#include <iostream>
#include <queue>
using namespace std;

#include <stdint.h>

typedef pair<RainbowChain, int> PPR;

class RainbowChain
{
public:
    uint64_t nStartKey, nEndKey;
    bool operator < (const RainbowChain &m) const;
};

struct cmp
{
    bool operator()(const PPR &a, const PPR &b)
    {
        RainbowChain  r1 = a.first;
        RainbowChain  r2 = b.first;
        if(r1.nEndKey < r2.nEndKey)
            return true;
        return false;
    }
};

// 95a7 cf4e 2000 0000 210f f300 8000 0000
// ca95 a983 b100 0000 d110 f400 8000 0000
// 8b1c 469a 1d00 0000 0b61 f400 8000 0000
// 38bf 7805 6600 0000 4569 f400 8000 0000
// 570e 67e5 4c00 0000 99f3 f400 8000 0000
// 0446 98ec 6d00 0000 c17a f500 8000 0000
// d8ff 7965 fd00 0000 677b f500 8000 0000
// dbe1 1a6a 4100 0000 0ba3 f500 8000 0000
// 3697 a03c 2700 0000 53d7 f500 8000 0000
// d2ed 7171 6900 0000 1ce2 f500 8000 0000
// 91dc 556f 7800 0000 94ac f600 8000 0000
// e57c cada a700 0000 acd3 f600 8000 0000
// 332f aeab a600 0000 0eff f600 8000 0000
// 52f9 cd7e 5600 0000 f0ad f700 8000 0000
// f914 80c4 6900 0000 9ce4 f700 8000 0000
// ca33 6679 2a00 0000 83be f800 8000 0000
// 5f2d 73d3 7300 0000 34f1 f800 8000 0000

priority_queue<PPR, vector<PPR>, cmp> chainPQ;

int main()
{
	RainbowChain chain;
	chain.nEndKey = 0x31e5a0e3bc;
	chainPQ.push(make_pair(chain,10));
	chain.nEndKey = 0xd2ed717169;
	chainPQ.push(make_pair(chain,9));
	chain.nEndKey = 0x332faeaba6;
	chainPQ.push(make_pair(chain,11));

	while(!chainPQ.empty())
	{
		cout<<chainPQ.top().first.nEndKey<<" "<<chainPQ.top().second<<endl;
		chainPQ.pop();
	}
	return 0;
}