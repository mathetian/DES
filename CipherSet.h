#ifndef _HASH_SET_H
#define _HASH_SET_H

#include <vector>
using namespace std;

#include <stdint.h>

class CipherSet{
public:
	CipherSet();
	virtual ~ CipherSet();

private:
	vector<uint64_t> m_vHash;
	vector<pair<uint64_t,uint64_t> > m_vFound;
	int index;
	int solve;
public:
	void AddHash(uint64_t cipherText);
	bool AnyHashLeft();
	uint64_t GetLeftHash();
	void AddResult(uint64_t cipherText,uint64_t key);
	void Done();
	bool solved();
public:
	int GetKeyFoundNum();
};
#endif